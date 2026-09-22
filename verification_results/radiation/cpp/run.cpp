#include "common.hpp"
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
namespace rr
{
namespace
{
struct Lock {
	int fd = -1;
	explicit Lock(const fs::path &p)
	{
		fd = open(p.c_str(), O_CREAT | O_RDWR, 0600);
		require(fd >= 0, "Cannot lock batch");
		if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
			::close(fd);
			fd = -1;
			throw std::runtime_error("Another runner owns this batch");
		}
	}
	~Lock()
	{
		if (fd >= 0)
			::close(fd);
	}
};
Json scaledConfig(Json config)
{
	double old = numeric(config.at("mesh.scale"));
	require(old > 0, "Invalid template mesh.scale");
	double stretch = halfWidth / old;
	if (config.contains("units.centimeters") || config.contains("units.seconds") ||
		config.contains("units.grams")) {
		require(
			close(cCgs * numeric(config.value("units.seconds", "1")) /
					numeric(config.value("units.centimeters", "1")),
				  1, 1e-12),
			"Use original regression templates, not generated run.ini");
	}
	config["mesh.scale"] = number(halfWidth);
	config["units.centimeters"] = "1";
	config["units.seconds"] = "1";
	config["units.grams"] = "1";
	config["radiation.test.extinction"] =
		number(numeric(config.at("radiation.test.extinction")) / stretch);
	config["radiation.test.width"] =
		number(numeric(config.at("radiation.test.width")) * stretch);
	config["radiation.test.luminosity"] = number(
		numeric(config.at("radiation.test.luminosity")) * cCgs * stretch * stretch);
	require(numeric(config.at("radiation.test.extinction")) >= 0 &&
				numeric(config.at("radiation.test.width")) > 0 &&
				numeric(config.at("radiation.test.width")) <= halfWidth / 2,
			"Invalid template opacity/width");
	for (auto key : {"radiation.test.luminosity", "radiation.test.background",
			 "radiation.test.amplitude"}) {
		require(numeric(config.at(key)) > 0, "Invalid test parameter");
	}
	double cap = numeric(config.value("timestep.fixed", "0"));
	require(cap >= 0, "Negative timestep.fixed");
	if (cap > 0) {
		config["timestep.fixed"] = number(cap * stretch / cCgs);
	}
	return config;
}
Json signature(Json cfg)
{
	for (auto k : {"mesh.level.maximum", "mesh.level.minimum", "output.directory",
			 "radiation.test.reference", "output.disabled", "output.interval"}) {
		cfg.erase(k);
	}
	return cfg;
}
bool configEqual(Json a, Json b)
{
	for (auto k : {"output.directory", "radiation.test.reference"}) {
		a.erase(k);
		b.erase(k);
	}
	if (a.size() != b.size())
		return false;
	for (auto it = a.begin(); it != a.end(); ++it) {
		if (!b.contains(it.key()))
			return false;
		if (it.value() == b[it.key()])
			continue;
		try {
			if (!close(numeric(it.value()), numeric(b[it.key()]), 1e-14))
				return false;
		} catch (const std::exception &) {
			return false;
		}
	}
	return true;
}
void ensureCgs(const Options &o)
{
	auto p = sourceRoot(o.root) / "src/physcon.cpp";
	auto s = readText(p);
	std::regex re(
		R"((else if\s*\(opts\(\)\.radiation\)\s*\{[^{}]*?)(\bt\s*=\s*opts\(\)\.code_to_cm\s*/\s*2\.99792458e\+?10\s*;))");
	auto count = std::distance(std::sregex_iterator(s.begin(), s.end(), re), std::sregex_iterator());
	if (!count)
		return;
	require(count == 1, "Ambiguous legacy unit override");
	if (o.dryRun) {
		std::cout << "Would correct legacy c=1 override with backup\n";
		return;
	}
	require(!o.noBuild, "Source forces c=1; run without --no-build to correct and rebuild");
	auto backup = p;
	backup += ".before-cgs-" + stamp();
	fs::copy_file(p, backup);
	atomicText(p, std::regex_replace(s, re, "$1t = opts().code_to_s;"));
}
} // namespace
int run(Options o)
{
	Json saved;
	bool resumed = !o.resume.empty();
	if (resumed) {
		for (auto key : {"--output", "--time", "--snapshots", "--build", "--exe", "--odt", "--hard-dt",
						 "--rad-reconstruction", "positionals"})
			require(!o.explicitOptions.contains(key),
					"--resume uses saved cases, levels, build, executable, time and snapshot count");
		o.output = rr::absolute(o.resume);
		saved = readJson(o.output / "batch.json");
		require(saved.at("units") == cgsUnits, "Resume requires a CGS batch");
		auto workflow = saved.value("workflow", "");
		require(workflow == "run_live.py" || workflow == "radiation-results", "Unsupported batch workflow");
		o.root = saved.at("root").get<std::string>();
		o.build = saved.at("build").get<std::string>();
		o.exe = saved.at("executable").get<std::string>();
		o.selectedCases = saved.at("cases").get<std::vector<std::string>>();
		o.levels = saved.at("levels").get<std::vector<int>>();
		o.time = saved.at("time");
		o.snapshots = saved.at("snapshots");
		o.noBuild = true;
		if (saved.contains("options")) {
			auto &a = saved["options"];
			o.odt = a.value("odt", 0.);
			o.hard_dt = a.value("hard_dt", 0.);
			o.noSilo = a.value("no_silo", false);
			o.reconstruction = a.value("reconstruction", "");
			if (o.generator.empty())
				o.generator = a.value("generator", "");
		}
	}
	o.root = rr::absolute(o.root);
	o.build = rr::absolute(o.build.is_absolute() ? o.build : o.root / o.build);
	auto source = sourceRoot(o.root);
	auto tools = toolsRoot(o.root);
	require(fs::is_regular_file(source / "CMakeLists.txt"),
			"Cannot find Octo-TIGER source directory: " + source.string());
	if (o.exe.empty())
		o.exe = o.build / "octotiger";
	o.exe = rr::absolute(o.exe);
	if (!o.generator.empty())
		o.generator = rr::absolute(o.generator);
	if (o.output.empty())
		o.output = o.root / "verification_results/results" / ((o.command == "live" ? "live-" : "") + stamp());
	o.output = rr::absolute(o.output);
	int inx = 0;
	std::istringstream cache(readText(o.build / "CMakeCache.txt"));
	std::string line;
	std::smatch match;
	std::regex rx(R"(^OCTOTIGER_WITH_GRIDDIM:[^=]+=([0-9]+)\s*$)");
	while (std::getline(cache, line))
		if (std::regex_match(line, match, rx))
			inx = std::stoi(match[1]);
	require(inx > 0 && inx <= 1024, "Cannot determine valid INX from CMakeCache.txt");
	require(readText(source / "src/grid.cpp").find("RADIATION_PLOT_EXPORT_BEGIN") != std::string::npos,
			"Install slice export first: radiation-results install --root PROJECT");
	Json plans = Json::array();
	for (int level : o.levels)
		for (auto &name : o.selectedCases) {
			require(level >= 0 && level <= 9 && std::find(cases.begin(), cases.end(), name) != cases.end(),
					"Invalid saved case/level");
			int n = inx * (1 << level);
			require(name != "gaussian_pulse" || (n >= 4 && n <= 512 && n % 2 == 0),
					"Gaussian reference requires even N from 4 through 512");
			auto cfg = scaledConfig(readConfig(tools / "configs" / (name + ".ini")));
			auto cap = cadence(o.time, o.snapshots, numeric(cfg.at("hydro.cfl")),
							   o.odt > 0 ? std::optional<double>(o.odt) : std::nullopt);
			double dt = cap.at("hard_dt");
			if (cfg.contains("timestep.fixed") && numeric(cfg["timestep.fixed"]) > 0) {
				dt = std::min(dt, numeric(cfg["timestep.fixed"]));
			}
			if (o.hard_dt > 0) {
				dt = std::min(dt, o.hard_dt);
			}
			cap["hard_dt"] = dt;
			cfg.update({{"mesh.level.maximum", std::to_string(level)},
						{"mesh.level.minimum", std::to_string(level)},
						{"runtime.stop_time", number(o.time)},
						{"output.interval", number(cap.at("odt"))},
						{"timestep.fixed", number(dt)},
						{"output.disabled", o.noSilo ? "on" : "off"},
						{"runtime.disable_diagnostics", "on"},
						{"problem.disable_analytic", "off"},
						{"hydro.species.count", "1"},
						{"hydro.species.atomic_mass", "1"},
						{"hydro.species.atomic_number", "1"}});
			if (!o.reconstruction.empty()) {
				cfg["rad_reconstruction"] = o.reconstruction;
			}
			plans.push_back(
				{{"case", name}, {"level", level}, {"cells", n}, {"config", cfg}, {"capture", cap}});
			std::cout << name << " level=" << level << " N=" << n << " t=" << number(o.time)
					  << " s odt=" << number(cap.at("odt")) << " s hard_dt=" << number(dt) << " s\n";
		}
	if (o.dryRun) {
		ensureCgs(o);
		return 0;
	}
	bool wantsMovies = !o.noMovies && !o.noSilo;
	checkDependencies(o, wantsMovies);
	if (!resumed)
		require(!fs::exists(o.output) || (fs::is_directory(o.output) && fs::is_empty(o.output)),
				"Output must be a new empty directory");
	fs::create_directories(o.output);
	Lock lock(o.output / ".runner.lock");
	if (resumed) {
		require(sha256(o.exe) == saved.at("executable_sha256").get<std::string>(),
				"Executable changed since batch began; use a new batch");
		for (auto it = saved.at("source_sha256").begin(); it != saved.at("source_sha256").end(); ++it)
			require(sha256(source / it.key()) == it.value().get<std::string>(),
					"Source changed since batch began: " + it.key());
		auto old = readJson(o.output / "live.json");
		require(!old.value("active", false),
				"Batch is marked active; stop its existing runner before resuming");
		require(old.at("runs").size() == plans.size(), "Saved batch plan changed");
		for (std::size_t i = 0; i < plans.size(); ++i)
			for (auto k : {"case", "level", "cells"})
				require(old["runs"][i][k] == plans[i][k], "Saved batch plan changed");
	}
	Json state{{"active", true},
			   {"time", o.time},
			   {"snapshots", o.snapshots},
			   {"threads", o.threads},
			   {"units", cgsUnits},
			   {"length", 2 * halfWidth},
			   {"c", cCgs},
			   {"message", "Preparing batch"},
			   {"runs", Json::array()}};
	for (auto &p : plans)
		state["runs"].push_back(
			{{"case", p["case"]}, {"level", p["level"]}, {"cells", p["cells"]}, {"stage", "Waiting"}});
	publish(o.output, state);
	std::cout << "Results: " << o.output / "index.html" << std::endl;
	Json meta;
	fs::path folder;
	std::optional<std::size_t> current;
	auto stage = [&](std::string s) {
		state["message"] = s;
		if (current)
			state["runs"][*current]["stage"] = s;
		publish(o.output, state);
	};
	try {
		if (!o.noOpen) {
			try {
				executeDetached({"xdg-open", (o.output / "index.html").string()}, o.output);
			} catch (const std::exception &e) {
				std::cerr << "Open the page manually: " << e.what() << '\n';
			}
		}
		ensureCgs(o);
		bool gaussian = std::find(o.selectedCases.begin(), o.selectedCases.end(), "gaussian_pulse") !=
						o.selectedCases.end();
		if (!o.noBuild) {
			stage("Building");
			std::vector<std::string> cmd{"cmake", "--build", o.build.string(), "--target", "octotiger"};
			cmd.insert(cmd.end(), {"-j", std::to_string(o.jobs)});
			execute(cmd, source, o.output / "build.log");
		}
		executable(o.exe.string());
		if (gaussian) {
			if (o.generator.empty()) {
				o.generator = fs::canonical("/proc/self/exe").parent_path() / "gen_radiation_reference";
				require(fs::is_regular_file(o.generator),
						"Missing bundled Gaussian reference generator; run verification_results/radiation/build_cpp.sh "
						"or select an existing generator with --generator PATH");
			}
			executable(o.generator.string());
		}
		auto hash = sha256(o.exe);
		if (!resumed) {
			Json sources;
			for (auto s :
				 {"src/grid.cpp", "src/physcon.cpp", "src/radiation/rad_grid.cpp",
				  "octotiger/test_problems/radiation/profiles.hpp",
				  "octotiger/test_problems/radiation/plot_output.hpp", "octotiger/radiation/conservation.hpp",
				  "octotiger/radiation/rad_grid.hpp", "octotiger/radiation/grey_opacity.hpp",
				  "octotiger/radiation/opacities.hpp", "octotiger/runReporter.hpp",
				  "src/options_processing.cpp", "src/node_server_actions_3.cpp", "src/runReporter.cpp"})
				sources[s] = sha256(source / s);
			saved = {{"created_utc", stamp()},
					 {"root", o.root.string()},
					 {"build", o.build.string()},
					 {"executable", o.exe.string()},
					 {"executable_sha256", hash},
					 {"source_sha256", sources},
					 {"origin", "Octo-TIGER application"},
					 {"cases", o.selectedCases},
					 {"levels", o.levels},
					 {"time", o.time},
					 {"snapshots", o.snapshots},
					 {"workflow", "radiation-results"},
					 {"units", cgsUnits},
					 {"length", 2 * halfWidth},
					 {"c", cCgs},
					 {"options",
					  {{"odt", o.odt},
					   {"hard_dt", o.hard_dt},
					   {"no_silo", o.noSilo},
					   {"reconstruction", o.reconstruction},
					   {"generator", o.generator.string()}}}};
			writeJson(o.output / "batch.json", saved);
		}
		for (std::size_t index = 0; index < plans.size(); ++index) {
			current = index;
			meta = Json();
			auto plan = plans[index];
			auto name = plan.at("case").get<std::string>();
			int level = plan.at("level"), cells = plan.at("cells");
			auto cfg = plan.at("config");
			folder = o.output / name / ("l" + std::to_string(level));
			Json expected{{"case", name},
						  {"level", level},
						  {"inx", inx},
						  {"cells", cells},
						  {"length", 2 * halfWidth},
						  {"dx", 2 * halfWidth / cells},
						  {"time", o.time},
						  {"c", cCgs},
						  {"units", cgsUnits},
						  {"origin", "Octo-TIGER application"},
						  {"background", numeric(cfg.at("radiation.test.background"))},
						  {"executable_sha256", hash},
						  {"movie_capture", plan["capture"]}};
			bool existing = resumed && fs::exists(folder);
			if (existing) {
				stage("Checking saved run");
				auto candidate = readJson(folder / "run.json");
				auto status = candidate.value("status", "");
				bool eligible =
					status == "complete" ||
					(status == "failed" &&
					 candidate.value("error", "")
						 .starts_with("Could not verify the running solver's CGS units and light speed;"));
				require(eligible, "Existing run is incomplete; use a new batch. Its files are retained: " +
									  folder.string());
				for (auto it = expected.begin(); it != expected.end(); ++it)
					require(candidate.value(it.key(), Json()) == it.value(),
							"Saved run metadata differs: " + it.key());
				require(readConfig(folder / "run.ini") == candidate.at("config"),
						"Saved run.ini differs from run.json");
				require(configEqual(candidate.at("config"), cfg),
						"Saved configuration differs from current templates");
				verifyCgsSummary(folder / "runSummary.json");
				candidate["norms"] = readNorms(folder, candidate);
				readSlice(folder, candidate);
				auto b = readConservation(folder, candidate);
				if (b)
					candidate["conservation"] = b->summary;
				if (wantsMovies)
					numericalSilos(folder);
				if (name == "gaussian_pulse")
					require(sha256(candidate.at("config").at("radiation.test.reference").get<std::string>()) ==
								candidate.at("reference_sha256").get<std::string>(),
							"Gaussian reference changed");
				candidate["status"] = "complete";
				candidate.erase("error");
				meta = candidate;
				writeJson(folder / "run.json", meta);
			} else {
				fs::create_directories(folder / "radiation-slices");
				cfg["output.directory"] = folder.string() + "/";
				cfg["output.results_file"] = (folder / "runSummary.json").string();
				meta = expected;
                meta["opacity"] = opacityMetadata(cfg);
				meta["status"] = "running";
				writeJson(folder / "run.json", meta);
				if (name == "gaussian_pulse") {
					stage("Generating reference");
					auto ref = folder / "reference.bin";
					execute({o.generator.string(), "--output", ref.string(), "--cells", std::to_string(cells),
							 "--length", number(2 * halfWidth), "--c", number(cCgs), "--chi",
							 cfg.at("radiation.test.extinction"), "--width",
							 cfg.at("radiation.test.width"), "--background",
							 cfg.at("radiation.test.background"), "--amplitude",
							 cfg.at("radiation.test.amplitude"),
							 "--time", number(o.time)},
							folder, folder / "reference.log");
					cfg["radiation.test.reference"] = ref.string();
					meta["reference_sha256"] = sha256(ref);
				}
				writeConfig(folder / "run.ini", cfg);
				meta["config"] = cfg;
				// Preserve numeric string spelling from older Python metadata when extending a batch.
				if (resumed)
					for (const auto &[oldFolder, oldMeta] : completedRuns(o.output)) {
						if (oldMeta.at("case") != name)
							continue;
						auto prior = oldMeta.at("config");
						if (configEqual(signature(prior), signature(cfg))) {
							const auto comparable = signature(prior);
							for (auto it = comparable.begin(); it != comparable.end(); ++it)
								cfg[it.key()] = it.value();
							writeConfig(folder / "run.ini", cfg);
							meta["config"] = cfg;
						}
						break;
					}
				meta["comparison_signature"] = signature(cfg);
				std::vector<std::string> cmd{
					o.exe.string(), "--runtime.config_file=" + (folder / "run.ini").string(),
					"--hpx:threads=" + std::to_string(o.threads)};
				meta["command"] = cmd;
				writeJson(folder / "run.json", meta);
				stage("Simulating");
				execute(cmd, folder, folder / "run.log");
				verifyCgsSummary(folder / "runSummary.json");
				meta["norms"] = readNorms(folder, meta);
				readSlice(folder, meta);
				auto b = readConservation(folder, meta);
				require(bool(b), "Missing radiation-conservation.csv; rebuild with diagnostics");
				meta["conservation"] = b->summary;
				meta["status"] = "complete";
				writeJson(folder / "run.json", meta);
			}
			stage("Making plots");
			render(o.output, o);
			if (wantsMovies) {
				stage("Rendering movie");
				auto movieOpts = o;
				movieOpts.reuseFrames = false;
				auto key = o.field + "-" + o.view + (o.view == "slice" ? "-" + o.axis : "");
				auto cached = folder / "movies" / key / "render.json";
				if (resumed && fs::is_regular_file(cached)) {
					auto oldRender = readJson(cached);
					movieOpts.reuseFrames =
						oldRender.at("signature").value("renderer", "") == sha256("/proc/self/exe");
				}
				auto movie = makeMovie(folder, movieOpts);
				movieIndex(o.output);
				state["runs"][index]["source_snapshots"] = movie.at("source_snapshots");
			}
			stage("Ready");
			current.reset();
			meta = Json();
		}
		state["active"] = false;
		state["message"] = "All requested simulations and postprocessing are complete";
		publish(o.output, state);
		return 0;
	} catch (const std::exception &e) {
		if (!meta.is_null() && meta.value("status", "") != "complete") {
			meta["status"] = "failed";
			meta["error"] = e.what();
			writeJson(folder / "run.json", meta);
		}
		if (current)
			state["runs"][*current]["stage"] =
				"Failed during " + state["runs"][*current]["stage"].get<std::string>();
		state["active"] = false;
		state["error"] = e.what();
		state["message"] = "Batch stopped; completed results retained";
		publish(o.output, state);
		throw;
	}
}
} // namespace rr
