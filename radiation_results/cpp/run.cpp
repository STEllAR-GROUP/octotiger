#include <stddef.h>
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
		if (fd >= 0) {
      ::close(fd);
    }
  }
};
using StudyRows = std::map<std::pair<std::string, int>, Json>;

StudyRows validate_study(const Json &study)
{
    const auto &document = study.at("document");
    require(document.is_object() && document.at("schema").is_number_integer() &&
        document.at("schema") == 1 && document.at("amr_plan").is_string() &&
        !document.at("amr_plan").get<std::string>().empty(), "Invalid study configuration schema or AMR plan path");
    const auto factors = document.value("reduced_c_factors", Json::object());
    require(factors.is_object(), "Study reduced_c_factors must be an object");
    for (auto it = factors.begin(); it != factors.end(); ++it) {
        require(coupled_case(it.key()) && it.value().is_number(),
            "Study reduced_c_factors may only name coupled cases with numeric factors");
        const auto factor = it.value().get<double>();
        require(std::isfinite(factor) && factor > 0 && factor <= 1,
            "Study reduced light-speed factors must be in (0, 1]");
    }
    const auto &rows = study.at("amr_plan").at("rows");
    require(rows.is_array() && (rows.size() == 28 || rows.size() == 32),
        "Study AMR plan must contain the seven legacy cases, optionally plus Marshak, at levels 2..5");
    StudyRows result;
    for (const auto &row : rows) {
        require(row.at("case").is_string() && row.at("finest_level").is_number_integer() &&
            row.at("finest_level") >= 2 && row.at("finest_level") <= 5,
            "Invalid case or finest level in study AMR plan");
        const auto name = row.at("case").get<std::string>();
        const int level = row.at("finest_level").get<int>();
        require(std::find(cases.begin(), cases.end(), name) != cases.end() &&
            row.at("base_level").is_number_integer() && row.at("base_level") == level - 1 &&
            (row.at("field") == "density" || row.at("field") == "radiation_energy") &&
            (row.at("mode") == "absolute" || row.at("mode") == "relative") &&
            row.at("buffer").is_number_integer() && row.at("buffer") >= 0 && row.at("buffer") <= 2,
            "Invalid adjacent-level gradient settings in study AMR plan");
        const auto coarsen = row.value("coarsen_factor", Json(.8));
        for (const auto &value : {row.at("selected").at("threshold"), row.at("floor"), coarsen}) {
            require(value.is_number() && std::isfinite(value.get<double>()) && value.get<double>() > 0,
                "Study AMR threshold, floor and coarsening factor must be finite and positive");
        }
        require(coarsen.get<double>() <= 1, "Study AMR coarsening factor must be at most 1");
        require(result.emplace(std::make_pair(name, level), row).second,
            "Duplicate case/level in study AMR plan");
    }
    for (const auto &name : cases) {
        int count = 0;
        for (int level = 2; level <= 5; ++level) {
            count += result.count({name, level});
        }
        require(count == 4 || (name == "marshak" && count == 0),
            "Study AMR plan requires all levels 2..5 for every included case");
    }
    return result;
}

Json load_study(const fs::path &path)
{
    const auto config_path = rr::absolute(path);
    const auto document = read_json(config_path);
    require(document.is_object() && document.contains("amr_plan") && document.at("amr_plan").is_string(),
        "Study configuration requires an amr_plan path");
    fs::path plan_path = document.at("amr_plan").get<std::string>();
    require(!plan_path.empty(), "Study AMR plan path must not be empty");
    plan_path = rr::absolute(plan_path.is_absolute() ? plan_path : config_path.parent_path() / plan_path);
    return {{"path", config_path.string()}, {"document", document},
        {"amr_plan_path", plan_path.string()}, {"amr_plan", read_json(plan_path)}};
}

Options study_controls(Options controls, const StudyRows &rows, const Json &study,
    const std::string &name, int level)
{
    if (rows.empty()) {
        return controls;
    }
    const auto found = rows.find({name, level});
    require(found != rows.end(), "Study AMR plan does not contain selected case/level");
    const auto &row = found->second;
    controls.min_level = level - 1;
    controls.gradient_field = row.at("field");
    controls.gradient_mode = row.at("mode");
    // The sampler's transport half-width is 1, while the CGS runner uses 3e10 cm.
    // Both gradient modes divide by this length exactly once; coupled rows are CGS.
    controls.gradient_threshold = row.at("selected").at("threshold").get<double>() /
        (coupled_case(name) || name == "marshak" ? 1. : half_width);
    controls.gradient_floor = row.at("floor");
    controls.gradient_buffer = row.at("buffer");
    controls.gradient_coarsen_factor = row.value("coarsen_factor", .8);
    controls.reduced_c_factor = study.at("document").value("reduced_c_factors", Json::object()).value(name, 1.);
    return controls;
}

Json scaled_config(Json config)
{
	double old = numeric(config.at("xscale"));
	require(old > 0, "Invalid template xscale");
	double stretch = half_width / old;
	if (config.contains("code_to_cm") || config.contains("code_to_s") || config.contains("code_to_g")) {
    require(
			close(c_cgs * numeric(config.value("code_to_s", "1")) / numeric(config.value("code_to_cm", "1")),
				  1, 1e-12),
			"Use original regression templates, not generated run.ini");
  }
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
	for (auto key : {"rad_test_luminosity", "rad_test_background", "rad_test_amplitude"}) {
    require(numeric(config.at(key)) > 0, "Invalid test parameter");
  }
  double cap = numeric(config.value("hard_dt", "0"));
	require(cap >= 0, "Negative hard_dt");
	if (cap > 0) {
    config["hard_dt"] = number(cap * stretch / c_cgs);
  }
  return config;
}
Json signature(Json cfg)
{
    cfg["radiation_algorithm_version"] = 1;
	for (auto k : {"max_level", "min_level", "datadir", "rad_reference", "disable_output", "odt", "radiation_backend"}) {
    cfg.erase(k);
  }
  return cfg;
}
bool config_equal(Json a, Json b)
{
    // Backend selects execution, not physics. Legacy runs used the CPU path.
    for (Json* config : {&a, &b}) {
        if (!config->contains("radiation_backend")) {
            (*config)["radiation_backend"] = "cpu";
        }
        if (!config->contains("radiation_model")) {
            (*config)["radiation_model"] = "m1";
        }
        if (!config->contains("eos")) {
            (*config)["eos"] = "IDEAL";
        }
        if (!config->contains("radiation_integrator")) {
            (*config)["radiation_integrator"] = "euler";
        }
    }
    // Legacy batches omitted this option; absence means physical-light-speed evolution.
    if (!a.contains("radiation_reduced_c_factor")) {
        a["radiation_reduced_c_factor"] = "1";
    }
    if (!b.contains("radiation_reduced_c_factor")) {
        b["radiation_reduced_c_factor"] = "1";
    }
	for (auto k : {"datadir", "rad_reference"}) {
		a.erase(k);
		b.erase(k);
	}
	if (a.size() != b.size()) {
    return false;
  }
  for (auto it = a.begin(); it != a.end(); ++it) {
		if (!b.contains(it.key())) {
      return false;
    }
    if (it.value() == b[it.key()]) {
      continue;
    }
    try {
			if (!close(numeric(it.value()), numeric(b[it.key()]), 1e-14)) {
        return false;
      }
    } catch (const std::exception &) {
			return false;
		}
	}
	return true;
}
void ensure_cgs(const Options &o)
{
	auto p = source_root(o.root) / "src/physcon.cpp";
	auto s = read_text(p);
	std::regex re(
		R"((else if\s*\(opts\(\)\.radiation\)\s*\{[^{}]*?)(\bt\s*=\s*opts\(\)\.code_to_cm\s*/\s*2\.99792458e\+?10\s*;))");
	auto count = std::distance(std::sregex_iterator(s.begin(), s.end(), re), std::sregex_iterator());
	if (!count) {
    return;
  }
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
	Json saved, study;
	bool resumed = !o.resume.empty();
	if (resumed) {
		for (auto key : {"--output", "--time", "--snapshots", "--build", "--exe", "--odt", "--hard-dt", "--study-config",
						 "--rad-reconstruction", "--launcher", "--radiation-reduced-c-factor", "--min-level",
                         "--gradient-refine-field", "--gradient-refine-mode", "--gradient-refine-threshold",
                         "--gradient-refine-floor", "--gradient-refine-buffer", "--gradient-refine-coarsen-factor", "positionals"}) {
      require(!o.explicit_options.contains(key),
					"--resume uses saved cases, levels, build, executable, time and snapshot count");
    }
    o.output = rr::absolute(o.resume);
		saved = read_json(o.output / "batch.json");
        const auto saved_backend = saved.value("options", Json::object()).value("radiation_backend", "cpu");
        require(!o.explicit_options.contains("--radiation-backend") || o.radiation_backend == saved_backend,
            "--resume uses the saved radiation backend; select a new batch to change --radiation-backend");
        o.radiation_backend = saved_backend;
        const auto saved_model = saved.value("options", Json::object()).value("radiation_model", "m1");
        require(saved_model == "m1" || saved_model == "fld", "Invalid saved radiation_model");
        require(!o.explicit_options.contains("--radiation-model") || o.radiation_model == saved_model,
            "--resume uses the saved radiation model; choose a new batch to change it");
        o.radiation_model = saved_model;
        const auto saved_integrator = saved.value("options", Json::object()).value("radiation_integrator", "euler");
        require(!o.explicit_options.contains("--radiation-integrator") || o.radiation_integrator == "auto" ||
            o.radiation_integrator == saved_integrator,
            "--resume uses the saved radiation integrator; choose a new batch to change it");
        o.radiation_integrator = saved_integrator;
        const auto saved_options = saved.value("options", Json::object());
        if (saved_options.contains("radiation_algorithm_version")) {
            require(saved_options.at("radiation_algorithm_version") == 1,
                "Unsupported saved radiation algorithm version; use a new batch");
        }
        if (saved_options.contains("radiation_flux_semantics")) {
            require(saved_options.at("radiation_flux_semantics") ==
                (saved_model == "fld" ? "derived_fld_flux" : "evolved_m1_flux"),
                "Saved radiation flux semantics do not match the model");
        }
        if (saved_model == "fld") {
            require(saved_options.value("radiation_limiter", "") == "levermore_pomraning" &&
                saved_options.value("radiation_flux_semantics", "") == "derived_fld_flux",
                "FLD resume requires explicit limiter and derived-flux metadata");
        }
        require((saved_model == "fld" && saved_integrator == "backward_euler" &&
            saved.value("options", Json::object()).value("radiation_algorithm_version", 0) == 1) ||
            (saved_model == "m1" && saved_integrator == "euler"),
            "This batch uses a retired radiation method or invalid model/integrator pair; use a new M1 RK1 or FLD batch");
        if (saved.contains("study_config")) {
            study = saved.at("study_config");
            require(saved.at("study_config_sha256") == sha256_text(study.dump()),
                "Frozen study configuration digest differs");
        }
		require(saved.at("units") == cgs_units || saved.at("units").value("system", "") == "mixed",
            "Resume requires recognized batch units");
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
            o.launcher = a.value("launcher", "");
            o.reduced_c_factor = a.value("radiation_reduced_c_factor", 1.);
            o.min_level = a.value("min_level", -1);
            o.gradient_field = a.value("gradient_refine_field", "none");
            o.gradient_mode = a.value("gradient_refine_mode", "relative");
            o.gradient_threshold = a.value("gradient_refine_threshold", 1.);
            o.gradient_floor = a.value("gradient_refine_floor", 1.e-30);
            o.gradient_buffer = a.value("gradient_refine_buffer", 0);
            o.gradient_coarsen_factor = a.value("gradient_refine_coarsen_factor", .8);
			if (o.generator.empty()) {
        o.generator = a.value("generator", "");
      }
    }
	}
	if (!resumed && !o.study_config.empty()) {
        study = load_study(o.study_config);
    }
    const auto study_rows = study.is_null() ? StudyRows{} : validate_study(study);
    const auto study_hash = study.is_null() ? std::string{} : sha256_text(study.dump());
	require(std::isfinite(o.reduced_c_factor) && o.reduced_c_factor > 0 && o.reduced_c_factor <= 1,
        "Invalid saved radiation_reduced_c_factor");
    require(std::set<std::string>{"cpu", "cuda", "kokkos"}.contains(o.radiation_backend),
        "Invalid saved radiation_backend");
    require(o.radiation_model == "m1" || o.radiation_model == "fld", "Invalid saved radiation_model");
    if (o.radiation_integrator == "auto") {
        o.radiation_integrator = o.radiation_model == "fld" ? "backward_euler" : "euler";
    }
    require((o.radiation_model == "m1" && o.radiation_integrator == "euler") ||
        (o.radiation_model == "fld" && o.radiation_integrator == "backward_euler"),
        "M1 requires Euler/RK1 and FLD requires backward_euler; use auto for the model default");
    const bool fld = o.radiation_model == "fld";
    if (fld) {
        require(o.radiation_backend == "cpu" && o.reduced_c_factor == 1.,
            "FLD requires the CPU backend and physical light speed (radiation-reduced-c-factor=1)");
        for (const auto &name : o.selected_cases) {
            require(fld_case(name), "FLD is available only for gaussian_pulse and marshak; unsupported case: " + name);
        }
    }
	o.root = rr::absolute(o.root);
	o.build = rr::absolute(o.build.is_absolute() ? o.build : o.root / o.build);
	auto source = source_root(o.root);
	auto tools = tools_root(o.root);
	require(fs::is_regular_file(source / "CMakeLists.txt"),
			"Cannot find Octo-TIGER source directory: " + source.string());
	if (o.exe.empty()) {
    o.exe = o.build / "octotiger";
  }
  o.exe = rr::absolute(o.exe);
    if (!o.launcher.empty()) {
        o.launcher = rr::absolute(o.launcher);
    }
	if (!o.generator.empty()) {
    o.generator = rr::absolute(o.generator);
  }
  if (o.output.empty()) {
    o.output = o.root / "radiation_results/results" / ((o.command == "live" ? "live-" : "") + stamp());
  }
  o.output = rr::absolute(o.output);
	int inx = 0;
	std::istringstream cache(read_text(o.build / "CMakeCache.txt"));
	std::string line;
	std::smatch match;
	std::regex rx(R"(^OCTOTIGER_WITH_GRIDDIM:[^=]+=([0-9]+)\s*$)");
	while (std::getline(cache, line)) {
    if (std::regex_match(line, match, rx)) {
      inx = std::stoi(match[1]);
    }
  }
  require(inx > 0 && inx <= 1024, "Cannot determine valid INX from CMakeCache.txt");
	require(read_text(source / "src/grid.cpp").find("RADIATION_PLOT_EXPORT_BEGIN") != std::string::npos,
			"Install slice export first: radiation-results install --root PROJECT");
	Json plans = Json::array();
    const bool rsla_metadata = !resumed || (saved.contains("options") &&
        saved.at("options").contains("radiation_reduced_c_factor"));
    // Old batches retain their recorded snapshot-derived cap when resumed.
    const bool no_silo_uncapped = o.no_silo && (!resumed ||
        saved.value("options", Json::object()).value("no_silo_ignores_snapshot_cadence", false));
	for (int level : o.levels) {
    for (auto &name : o.selected_cases) {
			require(level >= 0 && level <= 9 && std::find(cases.begin(), cases.end(), name) != cases.end(),
					"Invalid saved case/level");
			int n = inx * (1 << level);
			require(name != "gaussian_pulse" || (n >= 4 && n <= 512 && n % 2 == 0),
					"Gaussian reference requires even N from 4 through 512");
			auto cfg = read_config(tools / "configs" / (name + ".ini"));
            cfg["radiation_backend"] = o.radiation_backend;
            cfg["radiation_model"] = o.radiation_model;
            cfg["radiation_integrator"] = o.radiation_integrator;
            if (fld_case(name)) {
                cfg["eos"] = cfg.value("eos", "IDEAL");
            }
            const auto controls = study_controls(o, study_rows, study, name, level);
            if (fld) {
                require(controls.reduced_c_factor == 1., "FLD study settings require physical light speed");
                const auto off = [&](const char *key) {
                    const auto value = lower(cfg.value(key, "off"));
                    return value == "off" || value == "false" || value == "0";
                };
                require(off("hydro") && off("gravity") && numeric(cfg.value("omega", "0")) == 0. &&
                    lower(cfg.value("eos", "ideal")) == "ideal",
                    "FLD templates require stationary gas: hydro/gravity off, omega=0, EOS=IDEAL");
                cfg.update({{"radiation_limiter", "levermore_pomraning"}, {"rad_implicit", "on"},
                    {"eos", "IDEAL"}, {"hydro", "off"}, {"gravity", "off"}, {"omega", "0"}});
            }
            const bool adaptive = controls.gradient_field != "none";
            const int min_level = controls.min_level >= 0 ? controls.min_level : (adaptive ? level - 1 : level);
            require(min_level >= 0 && min_level <= level && (adaptive || min_level == level),
                "--min-level must be within [0,max level]; mixed levels require --gradient-refine-field");
            if (adaptive) {
                require(min_level == level - 1, "Gradient runs require exactly two adjacent allowed leaf levels");
                cfg.update({{"unigrid", "off"}, {"gradient_refine_field", controls.gradient_field},
                    {"gradient_refine_mode", controls.gradient_mode}, {"gradient_refine_threshold", number(controls.gradient_threshold)},
                    {"gradient_refine_floor", number(controls.gradient_floor)}, {"gradient_refine_buffer", std::to_string(controls.gradient_buffer)},
                    {"gradient_refine_coarsen_factor", number(controls.gradient_coarsen_factor)}});
            }
            const bool coupled = coupled_case(name);
            if (rsla_metadata) {
                cfg["radiation_reduced_c_factor"] = number(coupled ? controls.reduced_c_factor : 1.);
            }
            if (!coupled && name != "marshak") {
                cfg = scaled_config(cfg);
            }
            if (fld && name == "gaussian_pulse") {
                require(numeric(cfg.at("rad_test_chi")) > 0,
                    "The Gaussian FLD diffusion-limit reference requires positive scattering opacity");
            }
            const bool custom_time = o.explicit_options.contains("--time") ||
                (resumed && saved.value("custom_time", false));
            const double run_time = (coupled || name == "marshak") && !custom_time ? numeric(cfg.at("stop_time")) : o.time;
			auto cap = cadence(run_time, o.snapshots, numeric(cfg.at("cfl")),
							   o.odt > 0 ? std::optional<double>(o.odt) : std::nullopt);
			double dt = no_silo_uncapped ? 0. : cap.at("hard_dt").get<double>();
			if (cfg.contains("hard_dt") && numeric(cfg["hard_dt"]) > 0) {
        const double bound = numeric(cfg["hard_dt"]);
        dt = dt > 0 ? std::min(dt, bound) : bound;
      }
      if (o.hard_dt > 0) {
        dt = dt > 0 ? std::min(dt, o.hard_dt) : o.hard_dt;
      }
      cap["hard_dt"] = dt;
			cfg.update({{"max_level", std::to_string(level)},
						{"min_level", std::to_string(min_level)},
						{"stop_time", number(run_time)},
						{"odt", number(cap.at("odt"))},
						{"hard_dt", number(dt)},
						{"disable_output", o.no_silo ? "on" : "off"},
						{"disable_diagnostics", "on"},
						{"disable_analytic", coupled || fld ? "on" : "off"},
						{"n_species", "1"},
						{"atomic_mass", coupled ? cfg.value("atomic_mass", "1") : "1"},
						{"atomic_number", coupled ? cfg.value("atomic_number", "1") : "1"}});
			if (!o.reconstruction.empty()) {
        cfg["rad_reconstruction"] = o.reconstruction;
      }
      plans.push_back(
				{{"case", name}, {"level", level}, {"cells", n}, {"config", cfg}, {"capture", cap}});
			std::cout << name << " level=" << level << " N=" << n << " t=" << number(run_time)
					  << (name == "marshak" ? " normalized odt=" : " s odt=") << number(cap.at("odt")) << " hard_dt=" << number(dt)
                      << " alpha=" << cfg.value("radiation_reduced_c_factor", "1")
                      << " backend=" << o.radiation_backend
                      << " model=" << o.radiation_model
                      << " integrator=" << o.radiation_integrator << "\n";
            if (adaptive) {
                std::cout << "  adaptive leaves=" << min_level << ".." << level << " field=" << controls.gradient_field
                    << " mode=" << controls.gradient_mode << " threshold=" << number(controls.gradient_threshold) << "\n";
            }
		}
  }
  if (o.dry_run) {
		ensure_cgs(o);
		return 0;
	}
	bool wants_movies = !o.no_movies && !o.no_silo;
	check_dependencies(o, wants_movies);
	if (!resumed) {
    require(!fs::exists(o.output) || (fs::is_directory(o.output) && fs::is_empty(o.output)),
				"Output must be a new empty directory");
  }
  fs::create_directories(o.output);
	Lock lock(o.output / ".runner.lock");
	if (resumed) {
		require(sha256(o.exe) == saved.at("executable_sha256").get<std::string>(),
				"Executable changed since batch began; use a new batch");
		for (auto it = saved.at("source_sha256").begin(); it != saved.at("source_sha256").end(); ++it) {
      require(sha256(source / it.key()) == it.value().get<std::string>(),
					"Source changed since batch began: " + it.key());
    }
        if (!o.launcher.empty()) {
            require(sha256(o.launcher) == saved.at("options").at("launcher_sha256").get<std::string>(),
                "Launcher changed since batch began; use a new batch");
        }
    auto old = read_json(o.output / "live.json");
		require(!old.value("active", false),
				"Batch is marked active; stop its existing runner before resuming");
		require(old.at("runs").size() == plans.size(), "Saved batch plan changed");
		for (size_t i = 0; i < plans.size(); ++i) {
      for (auto k : {"case", "level", "cells"}) {
        require(old["runs"][i][k] == plans[i][k], "Saved batch plan changed");
      }
    }
  }
	const Json batch_units = std::find(o.selected_cases.begin(), o.selected_cases.end(), "marshak") != o.selected_cases.end()
        ? Json{{"system", "mixed"}, {"note", "Each run records its units; Marshak is normalized and the other cases use CGS"}} : cgs_units;
    Json state{{"active", true},
			   {"time", o.time},
			   {"snapshots", o.snapshots},
			   {"threads", o.threads},
			   {"units", batch_units},
			   {"length", 2 * half_width},
			   {"c", c_cgs},
			   {"message", "Preparing batch"},
			   {"runs", Json::array()}};
	for (auto &p : plans) {
    state["runs"].push_back(
			{{"case", p["case"]}, {"level", p["level"]}, {"cells", p["cells"]}, {"stage", "Waiting"}});
  }
  publish(o.output, state);
	std::cout << "Results: " << o.output / "index.html" << std::endl;
	Json meta;
	fs::path folder;
	std::optional<size_t> current;
	auto stage = [&](std::string s) {
		state["message"] = s;
		if (current) {
      state["runs"][*current]["stage"] = s;
    }
    publish(o.output, state);
	};
	try {
		if (!o.no_open) {
			try {
				execute_detached({"xdg-open", (o.output / "index.html").string()}, o.output);
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
			execute(cmd, source, o.output / "build.log");
		}
		executable(o.exe.string());
		if (gaussian) {
			if (o.generator.empty()) {
				o.generator = fs::canonical("/proc/self/exe").parent_path() / "gen_radiation_reference";
				require(fs::is_regular_file(o.generator),
						"Missing bundled Gaussian reference generator; run radiation_results/build_cpp.sh "
						"or select an existing generator with --generator PATH");
			}
			executable(o.generator.string());
		}
		auto hash = sha256(o.exe);
		if (!resumed) {
			Json sources;
			for (auto s :
				 {"src/grid.cpp", "src/physcon.cpp", "src/radiation/rad_grid.cpp",
                  "src/radiation/iteration.cpp", "src/radiation/fld.cpp",
                  "octotiger/radiation/fld.hpp", "octotiger/radiation/fld_runtime.hpp",
                  "octotiger/elliptic/composite.hpp",
                  "octotiger/radiation/ap_flux.hpp", "octotiger/radiation/implicit_source.hpp",
                  "octotiger/radiation/marshak_boundary.hpp",
				  "octotiger/test_problems/radiation/profiles.hpp",
				  "octotiger/test_problems/radiation/plot_output.hpp", "octotiger/radiation/conservation.hpp",
				  "octotiger/radiation/rad_grid.hpp", "src/node_server_actions_3.cpp",
                  "src/node_server_actions_1.cpp", "src/node_server_actions_2.cpp", "octotiger/node_server.hpp",
                  "src/test_problems/radiation/radiation.cpp", "src/test_problems/marshak/marshak.cpp",
                  "octotiger/test_problems/radiation/hydro_output.hpp",
                  "test_problems/radiation/hydro_benchmarks.hpp",
                  "octotiger/radiation/m1.hpp", "octotiger/radiation/opacities.hpp",
                  "test_problems/radiation/profiles.hpp", "test_problems/radiation/reference.hpp",
                  "octotiger/options.hpp", "src/options_processing.cpp",
                  "test_problems/radiation.hpp", "octotiger/amr/gradient.hpp",
                  "src/io/silo_in.cpp", "src/io/silo_out.cpp"}) {
        sources[s] = sha256(source / s);
      }
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
                     {"custom_time", o.explicit_options.contains("--time")},
					 {"units", batch_units},
					 {"length", 2 * half_width},
					 {"c", c_cgs},
					 {"options",
					  {{"odt", o.odt},
					   {"hard_dt", o.hard_dt},
					   {"no_silo", o.no_silo},
                       {"no_silo_ignores_snapshot_cadence", no_silo_uncapped},
					   {"reconstruction", o.reconstruction},
					   {"generator", o.generator.string()},
                       {"launcher", o.launcher.string()},
                       {"radiation_reduced_c_factor", o.reduced_c_factor},
                       {"radiation_backend", o.radiation_backend},
                       {"radiation_model", o.radiation_model},
                       {"radiation_integrator", o.radiation_integrator},
                       {"radiation_algorithm_version", 1},
                       {"radiation_flux_semantics", fld ? "derived_fld_flux" : "evolved_m1_flux"},
                       {"min_level", o.min_level}, {"gradient_refine_field", o.gradient_field},
                       {"gradient_refine_mode", o.gradient_mode}, {"gradient_refine_threshold", o.gradient_threshold},
                       {"gradient_refine_floor", o.gradient_floor}, {"gradient_refine_buffer", o.gradient_buffer},
                       {"gradient_refine_coarsen_factor", o.gradient_coarsen_factor},
                       {"launcher_sha256", o.launcher.empty() ? "" : sha256(o.launcher)}}}};
            if (fld) {
                saved["options"]["radiation_limiter"] = "levermore_pomraning";
            }
            if (!study.is_null()) {
                saved["study_config"] = study;
                saved["study_config_sha256"] = study_hash;
            }
			write_json(o.output / "batch.json", saved);
		}
		for (size_t index = 0; index < plans.size(); ++index) {
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
						  {"length", 2 * numeric(cfg.at("xscale"))},
                          {"dx", 2 * numeric(cfg.at("xscale")) / cells},
                          {"time", numeric(cfg.at("stop_time"))},
						  {"c", name == "marshak" ? 1. : c_cgs},
						  {"units", name == "marshak" ? marshak_units : cgs_units},
						  {"origin", "Octo-TIGER application"},
                          {"radiation_backend", o.radiation_backend},
                          {"radiation_model", o.radiation_model},
                          {"radiation_integrator", o.radiation_integrator},
                          {"radiation_algorithm_version", 1},
                          {"radiation_flux_semantics", fld ? "derived_fld_flux" : "evolved_m1_flux"},
						  {"background", numeric(cfg.value("rad_test_background", "0"))},
						  {"executable_sha256", hash},
						  {"movie_capture", plan["capture"]}};
            if (!study.is_null()) {
                expected["study_config_sha256"] = study_hash;
            }
            if (rsla_metadata) {
                const double factor = numeric(cfg.at("radiation_reduced_c_factor"));
                expected["radiation_reduced_c_factor"] = factor;
                expected["physical_c"] = name == "marshak" ? 1. : c_cgs;
                expected["transport_c"] = factor * expected["physical_c"].get<double>();
                expected["radiation_equations"] = fld ? "stationary physical-c FLD" :
                    (factor == 1. ? "physical-c" : "Skinner-Ostriker RSLA");
            }
            if (cfg.value("unigrid", "on") == "off") {
                expected["adaptive"] = true;
                expected["min_level"] = int(numeric(cfg.at("min_level")));
                expected["max_level"] = level;
                expected["mesh_description"] = "Adaptive Cartesian leaves; cells/dx denote configured finest equivalent resolution";
            }
            if (name == "marshak") {
                expected["reference_kind"] = "Su-Olson nonequilibrium diffusion reference";
                expected["validation_scope"] = "Normalized stationary material, equal radiation/material energy capacities. Analytic diffusion comparison is not an exact solution of nonlinear M1 transport.";
                expected["marshak_capacity_ratio"] = 1.;
                expected["marshak_opacity"] = 100.;
            }
            if (coupled_case(name)) {
                const auto original = read_config(tools / "configs" / (name + ".ini"));
                const double benchmark_time = numeric(original.at("stop_time"));
                expected["benchmark_time"] = benchmark_time;
                expected["reference_kind"] = "numerical-only";
                expected["validation_scope"] = numeric(cfg.at("stop_time")) < benchmark_time ?
                    "Early-time implementation check. This run stops before the configured benchmark time and does not validate the developed shock or blast." :
                    "Numerical Cartesian adaptation. No independent reference profile or quantitative agreement with the original spherical benchmark is established.";
            }
            if (fld) {
                expected["radiation_limiter"] = "levermore_pomraning";
                expected["reference_kind"] = "linear-diffusion-limit";
                expected["reference_model"] = name == "marshak" ?
                    "su-olson-linear-halfspace" : "periodic-constant-diffusion";
                expected["reference_exact_for_solver"] = false;
                expected["reference_sampling"] = name == "marshak" ? "cell-center point values" : "cell averages";
                expected["validation_scope"] = name == "marshak" ?
                    "Su-Olson linear nonequilibrium half-space diffusion reference, with equal radiation/material capacities. Nonlinear flux limiting and the finite far boundary can differ from the reference; reported errors are slice differences, not full-volume convergence evidence." :
                    "Periodic constant-D heat-equation reference for the represented initial energy, D=c/(3 chi), with derived F=-D grad(E). This is a diffusion-limit reference for nonlinear Levermore-Pomraning FLD; finite-gradient limiter differences remain even with vanishing numerical error. Reported errors are slice differences.";
                if (name == "gaussian_pulse") {
                    expected["reference_generator_sha256"] = sha256(o.generator);
                    expected["reference_diffusion_coefficient"] = expected.at("c").get<double>() /
                        numeric(cfg.at("rad_test_chi")) / 3.;
                }
            }
			bool existing = resumed && fs::exists(folder);
			if (existing) {
				stage("Checking saved run");
				auto candidate = read_json(folder / "run.json");
                if (!candidate.contains("radiation_backend")) {
                    candidate["radiation_backend"] = "cpu";
                }
                if (!candidate.contains("radiation_integrator")) {
                    candidate["radiation_integrator"] = "euler";
                }
                if (!candidate.contains("radiation_model")) {
                    candidate["radiation_model"] = "m1";
                }
                if (candidate.at("radiation_model") == "m1") {
                    if (!candidate.contains("radiation_flux_semantics")) {
                        candidate["radiation_flux_semantics"] = "evolved_m1_flux";
                    }
                    if (candidate.at("radiation_integrator") == "euler" &&
                        !candidate.contains("radiation_algorithm_version")) {
                        candidate["radiation_algorithm_version"] = 1;
                    }
                }
				auto status = candidate.value("status", "");
				bool eligible =
					status == "complete" ||
					(status == "failed" &&
					 candidate.value("error", "")
						 .starts_with("Could not verify the running solver's CGS units and light speed;"));
				require(eligible, "Existing run is incomplete; use a new batch. Its files are retained: " +
									  folder.string());
				for (auto it = expected.begin(); it != expected.end(); ++it) {
          require(candidate.value(it.key(), Json()) == it.value(),
							"Saved run metadata differs: " + it.key());
        }
        require(read_config(folder / "run.ini") == candidate.at("config"),
						"Saved run.ini differs from run.json");
				require(config_equal(candidate.at("config"), cfg),
						"Saved configuration differs from current templates");
				verify_cgs_log(folder / "run.log", name == "marshak" ? 1. : c_cgs);
                adopt_checkpoint_provenance(folder, candidate);
                if (fld) {
                    candidate["fld_diagnostics"] = read_fld_diagnostics(folder, candidate);
                    candidate.erase("norms");
                    read_slice(folder, candidate);
                    if (name == "marshak") {
                        read_marshak_material(folder, candidate);
                    }
                } else if (coupled_case(name)) {
                    read_hydro_profile(folder, candidate);
                } else {
                    candidate["norms"] = read_norms(folder, candidate);
                    read_slice(folder, candidate);
                }
				auto b = read_conservation(folder, candidate);
                if (candidate.value("adaptive", false)) {
                    candidate["mesh"] = slice_mesh_summary(folder, candidate);
                }
				if (b) {
          candidate["conservation"] = b->summary;
        }
        if (wants_movies) {
          numerical_silos(folder);
        }
        if (name == "gaussian_pulse") {
          require(sha256(candidate.at("config").at("rad_reference").get<std::string>()) ==
								candidate.at("reference_sha256").get<std::string>(),
							"Gaussian reference changed");
        }
        candidate["status"] = "complete";
				candidate.erase("error");
				meta = candidate;
				write_json(folder / "run.json", meta);
			} else {
				fs::create_directories(folder / (coupled_case(name) ? "radiation-hydro-slices" : "radiation-slices"));
                if (name == "marshak") {
                    fs::create_directories(folder / "marshak-material-slices");
                }
				cfg["datadir"] = folder.string() + "/";
				meta = expected;
				meta["status"] = "running";
				write_json(folder / "run.json", meta);
				if (name == "gaussian_pulse") {
					stage("Generating reference");
					auto ref = folder / "reference.bin";
					execute({o.generator.string(), "--output", ref.string(), "--cells", std::to_string(cells),
							 "--model", fld ? "fld" : "m1",
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
				if (resumed) {
          for (const auto &[old_folder, old_meta] : completed_runs(o.output)) {
						if (old_meta.at("case") != name) {
              continue;
            }
            auto prior = old_meta.at("config");
						if (config_equal(signature(prior), signature(cfg))) {
							const auto comparable = signature(prior);
							for (auto it = comparable.begin(); it != comparable.end(); ++it) {
                if (cfg.contains(it.key())) {
                    cfg[it.key()] = it.value();
                }
              }
              write_config(folder / "run.ini", cfg);
							meta["config"] = cfg;
						}
						break;
					}
        }
        meta["comparison_signature"] = signature(cfg);
                if (!study.is_null()) {
                    // A frozen study deliberately tunes gradients separately at each
                    // resolution. Keep every actual control in config and compare the
                    // shared study identity for the aggregate adaptive result pages.
                    for (auto key : {"gradient_refine_field", "gradient_refine_mode", "gradient_refine_threshold",
                            "gradient_refine_floor", "gradient_refine_buffer", "gradient_refine_coarsen_factor"}) {
                        meta["comparison_signature"].erase(key);
                    }
                    meta["comparison_signature"]["study_config_sha256"] = study_hash;
                }
				std::vector<std::string> cmd{o.exe.string(), "--config_file=" + (folder / "run.ini").string(),
											 "--hpx:threads=" + std::to_string(o.threads)};
                if (!o.launcher.empty()) {
                    cmd.insert(cmd.begin(), o.launcher.string());
                }
				meta["command"] = cmd;
				write_json(folder / "run.json", meta);
				stage("Simulating");
				CGSCheck checker;
                checker.expected_c = name == "marshak" ? 1. : c_cgs;
				execute(cmd, folder, folder / "run.log", [&](auto &line) { checker.feed(line); });
				checker.finish();
                adopt_checkpoint_provenance(folder, meta);
                if (fld) {
                    meta["fld_diagnostics"] = read_fld_diagnostics(folder, meta);
                    read_slice(folder, meta);
                    if (name == "marshak") {
                        read_marshak_material(folder, meta);
                    }
                } else if (coupled_case(name)) {
                    read_hydro_profile(folder, meta);
                    meta["reference_kind"] = "numerical-only";
                } else {
                    meta["norms"] = read_norms(folder, meta);
                    read_slice(folder, meta);
                }
				auto b = read_conservation(folder, meta);
                if (meta.value("adaptive", false)) {
                    meta["mesh"] = slice_mesh_summary(folder, meta);
                }
				require(bool(b) || coupled_case(name), "Missing radiation-conservation.csv; rebuild with diagnostics");
                if (b) {
                    meta["conservation"] = b->summary;
                }
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
		if (current) {
      state["runs"][*current]["stage"] =
				"Failed during " + state["runs"][*current]["stage"].get<std::string>();
    }
    state["active"] = false;
		state["error"] = e.what();
		state["message"] = "Batch stopped; completed results retained";
		publish(o.output, state);
		throw;
	}
}
} // namespace rr
