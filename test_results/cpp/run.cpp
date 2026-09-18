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
Json scaled_config(Json config)
{
	double old = numeric(config.at("xscale"));
	require(old > 0, "Invalid template xscale");
	double stretch = half_width / old;
	if (config.contains("code_to_cm") || config.contains("code_to_s") || config.contains("code_to_g"))
		require(
			close(c_cgs * numeric(config.value("code_to_s", "1")) / numeric(config.value("code_to_cm", "1")),
				  1, 1e-12),
			"Use original regression templates, not generated run.ini");
	config["xscale"] = number(half_width);
	config["code_to_cm"] = "1";
	config["code_to_s"] = "1";
	config["code_to_g"] = "1";
	config["rad_test_chi"] = number(numeric(config.at("rad_test_chi")) / stretch);
	config["rad_test_width"] = number(numeric(config.at("rad_test_width")) * stretch);
	config["rad_test_luminosity"] =
		number(numeric(config.at("rad_test_luminosity")) * c_cgs * stretch * stretch);
	require(numeric(config.at("rad_test_chi")) >= 0 && numeric(config.at("rad_test_width")) > 0 &&
				numeric(config.at("rad_test_width")) <= half_width / 2,
			"Invalid template opacity/width");
	for (auto key : {"rad_test_luminosity", "rad_test_background", "rad_test_amplitude"})
		require(numeric(config.at(key)) > 0, "Invalid test parameter");
	double cap = numeric(config.value("hard_dt", "0"));
	require(cap >= 0, "Negative hard_dt");
	if (cap > 0)
		config["hard_dt"] = number(cap * stretch / c_cgs);
	return config;
}
Json signature(Json cfg)
{
	for (auto k : {"max_level", "min_level", "datadir", "rad_reference", "disable_output", "odt"})
		cfg.erase(k);
	return cfg;
}
bool config_equal(Json a, Json b)
{
	for (auto k : {"datadir", "rad_reference"}) {
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
void ensure_cgs(const Options &o)
{
	auto p = o.root / "src/physcon.cpp";
	auto s = read_text(p);
	std::regex re(
		R"((else if\s*\(opts\(\)\.radiation\)\s*\{[^{}]*?)(\bt\s*=\s*opts\(\)\.code_to_cm\s*/\s*2\.99792458e\+?10\s*;))");
	auto count = std::distance(std::sregex_iterator(s.begin(), s.end(), re), std::sregex_iterator());
	if (!count)
		return;
	require(count == 1, "Ambiguous legacy unit override");
	if (o.dry_run) {
		std::cout << "Would correct legacy c=1 override with backup\n";
		return;
	}
	require(!o.no_build, "Source forces c=1; run without --no-build to correct and rebuild");
	auto backup = p;
	backup += ".before-cgs-" + stamp();
	fs::copy_file(p, backup);
	atomic_text(p, std::regex_replace(s, re, "$1t = opts().code_to_s;"));
}
} // namespace
int run(Options o)
{
	Json saved;
	bool resumed = !o.resume.empty();
	if (resumed) {
		for (auto key : {"--output", "--time", "--snapshots", "--build", "--exe", "--odt", "--hard-dt",
						 "--rad-reconstruction", "positionals"})
			require(!o.explicit_options.contains(key),
					"--resume uses saved cases, levels, build, executable, time and snapshot count");
		o.output = rr::absolute(o.resume);
		saved = read_json(o.output / "batch.json");
		require(saved.at("units") == cgs_units, "Resume requires a CGS batch");
		auto workflow = saved.value("workflow", "");
		require(workflow == "run_live.py" || workflow == "radiation-results", "Unsupported batch workflow");
		o.root = saved.at("root").get<std::string>();
		o.build = saved.at("build").get<std::string>();
		o.exe = saved.at("executable").get<std::string>();
		o.selected_cases = saved.at("cases").get<std::vector<std::string>>();
		o.levels = saved.at("levels").get<std::vector<int>>();
		o.time = saved.at("time");
		o.snapshots = saved.at("snapshots");
		o.no_build = true;
		if (saved.contains("options")) {
			auto &a = saved["options"];
			o.odt = a.value("odt", 0.);
			o.hard_dt = a.value("hard_dt", 0.);
			o.no_silo = a.value("no_silo", false);
			o.reconstruction = a.value("reconstruction", "");
			if (o.generator.empty())
				o.generator = a.value("generator", "");
		}
	}
	o.root = rr::absolute(o.root);
	o.build = rr::absolute(o.build.is_absolute() ? o.build : o.root / o.build);
	if (o.exe.empty())
		o.exe = o.build / "octotiger";
	o.exe = rr::absolute(o.exe);
	if (!o.generator.empty())
		o.generator = rr::absolute(o.generator);
	if (o.output.empty())
		o.output = o.root / "test_results/results" / ((o.command == "live" ? "live-" : "") + stamp());
	o.output = rr::absolute(o.output);
	int inx = 0;
	std::istringstream cache(read_text(o.build / "CMakeCache.txt"));
	std::string line;
	std::smatch match;
	std::regex rx(R"(^OCTOTIGER_WITH_GRIDDIM:[^=]+=([0-9]+)\s*$)");
	while (std::getline(cache, line))
		if (std::regex_match(line, match, rx))
			inx = std::stoi(match[1]);
	require(inx > 0 && inx <= 1024, "Cannot determine valid INX from CMakeCache.txt");
	require(read_text(o.root / "src/grid.cpp").find("RADIATION_PLOT_EXPORT_BEGIN") != std::string::npos,
			"Install slice export first: radiation-results install --root CHECKOUT");
	Json plans = Json::array();
	for (int level : o.levels)
		for (auto &name : o.selected_cases) {
			require(level >= 0 && level <= 9 && std::find(cases.begin(), cases.end(), name) != cases.end(),
					"Invalid saved case/level");
			int n = inx * (1 << level);
			require(name != "gaussian_pulse" || (n >= 4 && n <= 512 && n % 2 == 0),
					"Gaussian reference requires even N from 4 through 512");
			auto cfg = scaled_config(read_config(o.root / "test_results/configs" / (name + ".ini")));
			auto cap = cadence(o.time, o.snapshots, numeric(cfg.at("cfl")),
							   o.odt > 0 ? std::optional<double>(o.odt) : std::nullopt);
			double dt = cap.at("hard_dt");
			if (cfg.contains("hard_dt") && numeric(cfg["hard_dt"]) > 0)
				dt = std::min(dt, numeric(cfg["hard_dt"]));
			if (o.hard_dt > 0)
				dt = std::min(dt, o.hard_dt);
			cap["hard_dt"] = dt;
			cfg.update({{"max_level", std::to_string(level)},
						{"min_level", std::to_string(level)},
						{"stop_time", number(o.time)},
						{"odt", number(cap.at("odt"))},
						{"hard_dt", number(dt)},
						{"disable_output", o.no_silo ? "on" : "off"},
						{"disable_diagnostics", "on"},
						{"disable_analytic", "off"},
						{"n_species", "1"},
						{"atomic_mass", "1"},
						{"atomic_number", "1"}});
			if (!o.reconstruction.empty())
				cfg["rad_reconstruction"] = o.reconstruction;
			plans.push_back(
				{{"case", name}, {"level", level}, {"cells", n}, {"config", cfg}, {"capture", cap}});
			std::cout << name << " level=" << level << " N=" << n << " t=" << number(o.time)
					  << " s odt=" << number(cap.at("odt")) << " s hard_dt=" << number(dt) << " s\n";
		}
	if (o.dry_run) {
		ensure_cgs(o);
		return 0;
	}
	bool wants_movies = !o.no_movies && !o.no_silo;
	check_dependencies(o, wants_movies);
	if (!resumed)
		require(!fs::exists(o.output) || (fs::is_directory(o.output) && fs::is_empty(o.output)),
				"Output must be a new empty directory");
	fs::create_directories(o.output);
	Lock lock(o.output / ".runner.lock");
	if (resumed) {
		require(sha256(o.exe) == saved.at("executable_sha256").get<std::string>(),
				"Executable changed since batch began; use a new batch");
		for (auto it = saved.at("source_sha256").begin(); it != saved.at("source_sha256").end(); ++it)
			require(sha256(o.root / it.key()) == it.value().get<std::string>(),
					"Source changed since batch began: " + it.key());
		auto old = read_json(o.output / "live.json");
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
			   {"units", cgs_units},
			   {"length", 2 * half_width},
			   {"c", c_cgs},
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
		if (!o.no_open) {
			try {
				execute({"xdg-open", (o.output / "index.html").string()}, o.output, {}, {}, false);
			} catch (const std::exception &e) {
				std::cerr << "Open the page manually: " << e.what() << '\n';
			}
		}
		ensure_cgs(o);
		bool gaussian = std::find(o.selected_cases.begin(), o.selected_cases.end(), "gaussian_pulse") !=
						o.selected_cases.end();
		if (!o.no_build) {
			stage("Building");
			std::vector<std::string> cmd{"cmake", "--build", o.build.string(), "--target", "octotiger"};
			cmd.insert(cmd.end(), {"-j", std::to_string(o.jobs)});
			execute(cmd, o.root, o.output / "build.log");
		}
		executable(o.exe.string());
		if (gaussian) {
			if (o.generator.empty()) {
				o.generator = fs::canonical("/proc/self/exe").parent_path() / "gen_radiation_reference";
				require(fs::is_regular_file(o.generator),
						"Missing bundled Gaussian reference generator; run test_results/build_cpp.sh "
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
				  "octotiger/radiation/rad_grid.hpp", "src/node_server_actions_3.cpp"})
				sources[s] = sha256(o.root / s);
			saved = {{"created_utc", stamp()},
					 {"root", o.root.string()},
					 {"build", o.build.string()},
					 {"executable", o.exe.string()},
					 {"executable_sha256", hash},
					 {"source_sha256", sources},
					 {"origin", "Octo-TIGER application"},
					 {"cases", o.selected_cases},
					 {"levels", o.levels},
					 {"time", o.time},
					 {"snapshots", o.snapshots},
					 {"workflow", "radiation-results"},
					 {"units", cgs_units},
					 {"length", 2 * half_width},
					 {"c", c_cgs},
					 {"options",
					  {{"odt", o.odt},
					   {"hard_dt", o.hard_dt},
					   {"no_silo", o.no_silo},
					   {"reconstruction", o.reconstruction},
					   {"generator", o.generator.string()}}}};
			write_json(o.output / "batch.json", saved);
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
						  {"length", 2 * half_width},
						  {"dx", 2 * half_width / cells},
						  {"time", o.time},
						  {"c", c_cgs},
						  {"units", cgs_units},
						  {"origin", "Octo-TIGER application"},
						  {"background", numeric(cfg.at("rad_test_background"))},
						  {"executable_sha256", hash},
						  {"movie_capture", plan["capture"]}};
			bool existing = resumed && fs::exists(folder);
			if (existing) {
				stage("Checking saved run");
				auto candidate = read_json(folder / "run.json");
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
				require(read_config(folder / "run.ini") == candidate.at("config"),
						"Saved run.ini differs from run.json");
				require(config_equal(candidate.at("config"), cfg),
						"Saved configuration differs from current templates");
				verify_cgs_log(folder / "run.log");
				candidate["norms"] = read_norms(folder, candidate);
				read_slice(folder, candidate);
				auto b = read_conservation(folder, candidate);
				if (b)
					candidate["conservation"] = b->summary;
				if (wants_movies)
					numerical_silos(folder);
				if (name == "gaussian_pulse")
					require(sha256(candidate.at("config").at("rad_reference").get<std::string>()) ==
								candidate.at("reference_sha256").get<std::string>(),
							"Gaussian reference changed");
				candidate["status"] = "complete";
				candidate.erase("error");
				meta = candidate;
				write_json(folder / "run.json", meta);
			} else {
				fs::create_directories(folder / "radiation-slices");
				cfg["datadir"] = folder.string() + "/";
				meta = expected;
				meta["status"] = "running";
				write_json(folder / "run.json", meta);
				if (name == "gaussian_pulse") {
					stage("Generating reference");
					auto ref = folder / "reference.bin";
					execute({o.generator.string(), "--output", ref.string(), "--cells", std::to_string(cells),
							 "--length", number(2 * half_width), "--c", number(c_cgs), "--chi",
							 cfg.at("rad_test_chi"), "--width", cfg.at("rad_test_width"), "--background",
							 cfg.at("rad_test_background"), "--amplitude", cfg.at("rad_test_amplitude"),
							 "--time", number(o.time)},
							folder, folder / "reference.log");
					cfg["rad_reference"] = ref.string();
					meta["reference_sha256"] = sha256(ref);
				}
				write_config(folder / "run.ini", cfg);
				meta["config"] = cfg;
				// Preserve numeric string spelling from older Python metadata when extending a batch.
				if (resumed)
					for (const auto &[old_folder, old_meta] : completed_runs(o.output)) {
						if (old_meta.at("case") != name)
							continue;
						auto prior = old_meta.at("config");
						if (config_equal(signature(prior), signature(cfg))) {
							const auto comparable = signature(prior);
							for (auto it = comparable.begin(); it != comparable.end(); ++it)
								cfg[it.key()] = it.value();
							write_config(folder / "run.ini", cfg);
							meta["config"] = cfg;
						}
						break;
					}
				meta["comparison_signature"] = signature(cfg);
				std::vector<std::string> cmd{o.exe.string(), "--config_file=" + (folder / "run.ini").string(),
											 "--hpx:threads=" + std::to_string(o.threads)};
				meta["command"] = cmd;
				write_json(folder / "run.json", meta);
				stage("Simulating");
				CGSCheck checker;
				execute(cmd, folder, folder / "run.log", [&](auto &line) { checker.feed(line); });
				checker.finish();
				meta["norms"] = read_norms(folder, meta);
				read_slice(folder, meta);
				auto b = read_conservation(folder, meta);
				require(bool(b), "Missing radiation-conservation.csv; rebuild with diagnostics");
				meta["conservation"] = b->summary;
				meta["status"] = "complete";
				write_json(folder / "run.json", meta);
			}
			stage("Making plots");
			render(o.output, o);
			if (wants_movies) {
				stage("Rendering movie");
				auto movie_opts = o;
				movie_opts.reuse_frames = false;
				auto key = o.field + "-" + o.view + (o.view == "slice" ? "-" + o.axis : "");
				auto cached = folder / "movies" / key / "render.json";
				if (resumed && fs::is_regular_file(cached)) {
					auto old_render = read_json(cached);
					movie_opts.reuse_frames =
						old_render.at("signature").value("renderer", "") == sha256("/proc/self/exe");
				}
				auto movie = make_movie(folder, movie_opts);
				movie_index(o.output);
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
			write_json(folder / "run.json", meta);
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
