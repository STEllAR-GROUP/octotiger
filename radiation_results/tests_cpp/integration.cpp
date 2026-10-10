#include "common.hpp"
using namespace rr;
int main(int argc, char **argv)
{
	fs::path tmp;
	try {
		require(argc == 6, "Usage: rr-integration RUNNER FIXTURE_EXE SOURCE_DIR GNUPLOT CMAKE");
		auto runner = rr::absolute(argv[1]);
		tmp = fs::temp_directory_path() / ("rr integration " + stamp());
		fs::create_directories(tmp);
		// Exercise shared filesystems whose visible mount path is a directory symlink.
		const auto tool_alias = tmp / "tools alias";
		fs::create_directory_symlink(runner.parent_path(), tool_alias);
		runner = tool_alias / runner.filename();
		auto root = tmp / "project", source = root / "src/octotiger",
			 build = root / "build/octotiger/release", batch = tmp / "batch";
		fs::create_directories(build);
		for (auto name :
				 {"src/grid.cpp", "src/physcon.cpp", "src/radiation/rad_grid.cpp",
				  "octotiger/test_problems/radiation/profiles.hpp",
				  "octotiger/test_problems/radiation/plot_output.hpp", "octotiger/radiation/conservation.hpp",
				  "octotiger/radiation/rad_grid.hpp", "src/node_server_actions_3.cpp",
                  "src/node_server_actions_1.cpp", "src/node_server_actions_2.cpp", "octotiger/node_server.hpp",
                  "src/test_problems/radiation/radiation.cpp",
                  "octotiger/test_problems/radiation/hydro_output.hpp",
                  "test_problems/radiation/hydro_benchmarks.hpp",
                  "octotiger/radiation/m1.hpp", "octotiger/radiation/opacities.hpp",
                  "test_problems/radiation/profiles.hpp", "test_problems/radiation/reference.hpp",
                  "octotiger/options.hpp", "src/options_processing.cpp",
                  "test_problems/radiation.hpp", "octotiger/amr/gradient.hpp",
                  "src/io/silo_in.cpp", "src/io/silo_out.cpp"}) {
      atomic_text(source / name, "// RADIATION_PLOT_EXPORT_BEGIN: integration fixture\n");
    }
    atomic_text(source / "CMakeLists.txt",
					"cmake_minimum_required(VERSION 3.16)\n"
					"project(runner_fixture NONE)\nset(OCTOTIGER_WITH_GRIDDIM 4 CACHE STRING "
					"\"Grid size\")\n"
					"add_custom_target(octotiger COMMAND ${CMAKE_COMMAND} -E echo \"Mock solver "
					"already installed\")\n");
		execute({argv[5], "-S", source.string(), "-B", build.string()}, tmp);
		for (const auto &name : cases) {
      atomic_text(source / "radiation_results/configs" / (name + ".ini"),
						read_text(fs::path(argv[3]) / "configs" / (name + ".ini")));
    }
    auto solver = build / "octotiger";
		fs::copy_file(rr::absolute(argv[2]), solver);
		std::vector<std::string> start{
			runner.string(), "live",	  "wave",		  "0",		   "1",	   "release",	  "--root",
			root.string(),	 "--output",  batch.string(), "--threads", "2",	   "--snapshots", "3",
			"--no-build",	 "--no-open", "--no-movies",  "--gnuplot", argv[4]};
		execute(start, tmp, tmp / "start.log");
		int checks = 0;
		auto expect = [&](bool ok, const std::string &message) {
			require(ok, "TEST FAILED: " + message);
			++checks;
		};
        // Exercise the launcher against the real planner, without launching any
        // full-duration physics or rebuilding the helper tools inside this fixture.
        const auto launcher_dir = source / "radiation_results";
        for (const auto &name : {"run_ensman.sh", "run.sh"}) {
            fs::copy_file(fs::path(argv[3]) / name, launcher_dir / name);
        }
        atomic_text(launcher_dir / "results.sh", "#!/bin/sh\nexec " +
            shell_quote(runner.string()) + " \"$@\"\n");
        fs::permissions(launcher_dir / "results.sh", fs::perms::owner_exec, fs::perm_options::add);
        const auto launcher = launcher_dir / "run_ensman.sh";
        const auto planned_batch = tmp / "ensman dry run";
        execute({launcher.string(), "--build", build.string(), "--output", planned_batch.string(),
            "--dry-run"}, tmp, tmp / "ensman-plan.log");
        const auto plan_text = read_text(tmp / "ensman-plan.log");
        execute({launcher.string(), "--build", build.string(), "--output", planned_batch.string(),
            "--min-level", "1", "--gradient-refine-field", "radiation_energy", "--gradient-refine-mode", "absolute",
            "--gradient-refine-threshold", "1e-10", "--dry-run"}, tmp, tmp / "ensman-amr-plan.log");
        expect(read_text(tmp / "ensman-amr-plan.log").find("adaptive leaves=1..2") != std::string::npos,
            "Ensman wrapper forwards value-taking AMR options without treating their values as levels");
        const auto blast_position = plan_text.find("radiating_blast level=2 N=16 t=22000");
        const auto super_position = plan_text.find("supercritical_shock level=2 N=16 t=7500");
        const auto sub_position = plan_text.find("subcritical_shock level=2 N=16 t=38300");
        expect(blast_position != std::string::npos && super_position != std::string::npos &&
            sub_position != std::string::npos && blast_position < super_position &&
            super_position < sub_position && plan_text.find("streaming_wave level=") == std::string::npos,
            "Ensman launcher selects only three full-duration cases in the intended order");
        expect(!fs::exists(planned_batch), "Ensman dry-run does not create or run a batch");
        expect(execute({launcher.string(), "--threads", "13", "--dry-run"}, tmp,
            tmp / "ensman-over-budget.log", {}, false) != 0,
            "Ensman launcher rejects worker counts above the local resource limit");
		expect(!read_json(batch / "live.json").at("active").get<bool>(), "batch completed");
		expect(fs::file_size(batch / "plots/streaming_wave/convergence.png") > 100, "convergence rendered");
		for (auto &name : cases) {
      expect(fs::is_regular_file(batch / (name + ".html")), "dedicated problem page: " + name);
    }
    expect(read_text(batch / "gaussian_pulse.html").find("telegraph equation") != std::string::npos,
			   "Gaussian page identifies the telegraph reference before results exist");
		auto wave_page = read_text(batch / "streaming_wave.html");
		expect(wave_page.find("id=\"level-0\"") != std::string::npos &&
				   wave_page.find("id=\"level-1\"") != std::string::npos &&
				   read_json(batch / "plots/streaming_wave/errors.json").size() == 24,
			   "problem page includes every resolution and all four field norms");
		auto plot_time = fs::last_write_time(batch / "plots/streaming_wave/l0/slice_er.png");
		execute({runner.string(), "pages", batch.string(), "--gnuplot", "/missing/gnuplot", "--ffmpeg",
				 "/missing/ffmpeg"},
				tmp, tmp / "pages.log");
		expect(fs::last_write_time(batch / "plots/streaming_wave/l0/slice_er.png") == plot_time,
			   "HTML-only update needs no plotting tools and does not regenerate images");
		auto executions = [&] {
			for (int level : {0, 1}) {
        expect(read_text(batch / "streaming_wave" / ("l" + std::to_string(level)) /
								 "solver-executions") == "run\n",
					   "completed solver executed exactly once");
      }
    };
		executions();
		std::vector<std::string> resume{runner.string(), "live", "--resume",  batch.string(),
										"--threads",	 "3",	 "--no-open", "--no-movies",
										"--gnuplot",	 argv[4]};
		execute(resume, tmp, tmp / "resume.log");
		executions();
		// Older Python batches use this workflow name and omit the new options object.
		auto saved = read_json(batch / "batch.json");
		saved["workflow"] = "run_live.py";
		saved.erase("options");
		write_json(batch / "batch.json", saved);
		execute(resume, tmp, tmp / "legacy-resume.log");
		executions();
		auto radiation_source = source / "src/radiation/rad_grid.cpp";
		auto original = read_text(radiation_source);
		atomic_text(radiation_source, original + "// changed\n");
		expect(execute(resume, tmp, tmp / "changed-source.log", {}, false) != 0 &&
				   read_text(tmp / "changed-source.log").find("Source changed") != std::string::npos,
			   "changed source rejected");
		atomic_text(radiation_source, original);
		{
			std::ofstream out(solver, std::ios::app);
			out << "changed";
		}
		expect(execute(resume, tmp, tmp / "changed-exe.log", {}, false) != 0 &&
				   read_text(tmp / "changed-exe.log").find("Executable changed") != std::string::npos,
			   "changed executable rejected");
		fs::copy_file(rr::absolute(argv[2]), solver, fs::copy_options::overwrite_existing);
		auto run_file = batch / "streaming_wave/l0/run.json";
		auto meta = read_json(run_file);
		meta["time"] = 5.;
		write_json(run_file, meta);
		expect(execute(resume, tmp, tmp / "changed-meta.log", {}, false) != 0 &&
				   read_text(tmp / "changed-meta.log").find("Saved run metadata differs: time") !=
					   std::string::npos,
			   "mismatched run rejected");
		executions();
		// Exercise the actual failing path: build all cases in an application project
		// with ONLY an octotiger target, then invoke the real bundled FFTW generator.
		auto all_batch = tmp / "all batch";
		execute({runner.string(), "live", "all", "0", "release", "--root", root.string(), "--output",
				 all_batch.string(), "--threads", "2", "--snapshots", "3", "--no-open", "--no-movies",
				 "--gnuplot", argv[4], "--radiation-reduced-c-factor", "0.02"},
				tmp, tmp / "all.log");
		expect(!read_json(all_batch / "live.json").at("active").get<bool>(), "all-case batch completed");
		for (const auto &name : cases) {
      expect(read_json(all_batch / name / "l0/run.json").at("status") == "complete",
				   "all-case run completed: " + name);
      expect(close(read_json(all_batch / name / "l0/run.json").at("radiation_reduced_c_factor"),
          coupled_case(name) ? .02 : 1.),
          "runner applies RSLA only to coupled cases: " + name);
    }
        for (const auto &name : {"subcritical_shock", "supercritical_shock", "radiating_blast"}) {
            auto page = read_text(all_batch / (std::string(name) + ".html"));
            expect(page.find("without a supplied analytic reference") != std::string::npos &&
                page.find("Full-volume norms and orders") == std::string::npos,
                std::string(name) + " numerical-only page has no invented accuracy norms");
            expect(fs::file_size(all_batch / "plots" / name / "l0/profiles.png") > 100,
                std::string(name) + " hydro profiles rendered");
            auto run = read_json(all_batch / name / "l0/run.json");
            auto config = read_config(fs::path(argv[3]) / "configs" / (std::string(name) + ".ini"));
            expect(close(run.at("time"), numeric(config.at("stop_time"))) &&
                close(run.at("length"), 2 * numeric(config.at("xscale"))),
                std::string(name) + " preserves physical domain and template duration");
        }
    auto gaussian = all_batch / "gaussian_pulse/l0";
		auto display = read_json(all_batch / "plots/gaussian_pulse/l0/display.json");
		expect(display.at("energy_scale") == "log" && display.at("energy_floor").get<double>() > 0 &&
				   display.at("energy_floor").get<double>() < display.at("energy_maximum").get<double>(),
			   "Gaussian energy renders a finite logarithmic scale despite nonpositive excess samples");
		expect(fs::file_size(gaussian / "reference.bin") == 76 + 64 * 4 * 4 * 4,
			   "real generator wrote the full reference file");
		expect(read_text(all_batch / "build.log").find("gen_radiation_reference") == std::string::npos,
			   "application build does not request reference-generator target");
		expect(fs::equivalent(
				   read_json(all_batch / "batch.json").at("options").at("generator").get<std::string>(),
				   runner.parent_path() / "gen_radiation_reference"),
			   "bundled generator selected");
        const auto short_batch = tmp / "coupled short batch";
        const auto solver_launcher = tmp / "solver launcher.sh";
        atomic_text(solver_launcher, "#!/bin/sh\nprintf 'launcher ran\\n' >> launcher-executions\nexec \"$@\"\n");
        fs::permissions(solver_launcher, fs::perms::owner_exec, fs::perm_options::add);
        execute({runner.string(), "run", "subcritical", "0", "release", "--root", root.string(),
            "--output", short_batch.string(), "--time", "0.01", "--threads", "2", "--snapshots", "3",
            "--no-build", "--no-open", "--no-movies", "--no-silo", "--gnuplot", argv[4],
            "--launcher", solver_launcher.string(), "--radiation-reduced-c-factor", "0.001",
            "--hard-dt", "0.005"}, tmp, tmp / "short.log");
        const auto short_run = read_json(short_batch / "subcritical_shock/l0/run.json");
        expect(close(numeric(short_run.at("config").at("hard_dt")), .005),
            "no-Silo run retains explicit hard_dt without the smaller snapshot-derived cap");
        expect(close(short_run.at("radiation_reduced_c_factor"), .001) &&
            close(short_run.at("physical_c"), c_cgs) && close(short_run.at("transport_c"), .001 * c_cgs) &&
            numeric(short_run.at("config").at("radiation_reduced_c_factor")) == .001 &&
            read_text(short_batch / "subcritical_shock.html").find("Skinner-Ostriker") != std::string::npos &&
            read_text(short_batch / "plots/index.html").find("modified radiation equations") != std::string::npos,
            "RSLA factor is retained in configuration/provenance and explicitly labeled in reports");
        expect(execute({runner.string(), "run", "--radiation-reduced-c-factor", "0", "--dry-run"},
            tmp, tmp / "bad-factor.log", {}, false) != 0,
            "zero reduced light-speed factor is rejected");
        expect(short_run.at("executable_sha256") == sha256(solver) &&
            short_run.at("command").at(0) == solver_launcher.string() &&
            read_text(short_batch / "subcritical_shock/l0/launcher-executions") == "launcher ran\n",
            "launcher wraps solver execution while provenance identifies the real solver");
        expect(close(short_run.at("time"), .01) && short_run.at("benchmark_time").get<double>() > .01 &&
            short_run.at("validation_scope").get<std::string>().starts_with("Early-time"),
            "custom short run retains actual time and explicit incomplete benchmark label");
        execute({launcher.string(), "--resume", short_batch.string(), "--threads", "2", "--no-open",
            "--no-movies", "--gnuplot", argv[4]}, tmp, tmp / "short-resume.log");
        expect(read_text(short_batch / "subcritical_shock/l0/solver-executions") == "run\n",
            "custom-duration coupled resume does not rerun the solver");
        expect(close(read_json(short_batch / "subcritical_shock/l0/run.json").at("radiation_reduced_c_factor"), .001),
            "resume preserves the saved reduced light-speed factor");
        atomic_text(solver_launcher, read_text(solver_launcher) + "# changed launcher\n");
        expect(execute({launcher.string(), "--resume", short_batch.string(), "--no-open", "--no-movies",
            "--gnuplot", argv[4]}, tmp, tmp / "launcher-changed.log", {}, false) != 0 &&
            read_text(tmp / "launcher-changed.log").find("Launcher changed") != std::string::npos,
            "resume refuses a changed solver launcher");
        execute({runner.string(), "check", short_batch.string()}, tmp, tmp / "short-check.log");
        for (const auto &name : {"streaming_front", "subcritical_shock"}) {
            const auto adaptive_batch = tmp / (std::string(name) + " adaptive batch");
            execute({runner.string(), "run", name, "2", "release", "--root", root.string(),
                "--output", adaptive_batch.string(), "--min-level", "1", "--gradient-refine-field", "radiation_energy",
                "--gradient-refine-mode", "absolute", "--gradient-refine-threshold", "1e-10",
                "--gradient-refine-floor", "1e-20", "--gradient-refine-buffer", "2",
                "--no-build", "--no-open", "--no-movies", "--no-silo", "--gnuplot", argv[4]}, tmp, tmp / (std::string(name)+"-adaptive.log"));
            const auto folder = adaptive_batch / name / "l2";
            const auto adaptive_run = read_json(folder / "run.json");
            expect(numeric(adaptive_run.at("config").at("hard_dt")) == 0.,
                "no-Silo run without an explicit cap leaves timestep selection to the solver");
            expect(adaptive_run.at("adaptive") == true && adaptive_run.at("min_level") == 1 &&
                adaptive_run["config"]["unigrid"] == "off" &&
                adaptive_run["config"]["gradient_refine_threshold"] == "1e-10" &&
                adaptive_run["mesh"]["full_mesh"]["leaf_cells"] == 2304,
                "AMR controls and actual full-mesh census retained in provenance");
            const auto page = read_text(adaptive_batch / (std::string(name) + ".html"));
            expect(page.find("AMR levels") != std::string::npos && page.find("not a common cell-center plane") != std::string::npos,
                "adaptive plots explicitly describe actual mesh and leaf-average projection");
            execute({runner.string(), "run", "--resume", adaptive_batch.string(), "--no-open", "--no-movies", "--gnuplot", argv[4]},
                tmp, tmp / (std::string(name)+"-adaptive-resume.log"));
            expect(read_text(folder / "solver-executions") == "run\n", "AMR resume preserves saved controls and completed result");
        }
		fs::remove_all(tmp);
		std::cout << checks << " integration checks passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\nTest files retained at " << tmp << '\n';
		return 1;
	}
}
