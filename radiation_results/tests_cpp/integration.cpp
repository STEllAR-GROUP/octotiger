#include "common.hpp"
using namespace rr;
int main(int argc, char **argv)
{
	fs::path tmp;
	try {
		require(argc == 6 || (argc == 7 && std::string(argv[6]) == "--checkpoint-only"),
            "Usage: rr-integration RUNNER FIXTURE_EXE SOURCE_DIR GNUPLOT CMAKE [--checkpoint-only]");
        const bool checkpoint_only = argc == 7;
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
                  "src/radiation/iteration.cpp", "src/radiation/fld.cpp",
                  "octotiger/radiation/fld.hpp", "octotiger/radiation/fld_runtime.hpp",
                  "octotiger/radiation/fld_benchmarks.hpp",
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
			"--no-build",	 "--no-open", "--no-movies", "--radiation-integrator", "euler", "--gnuplot", argv[4]};
        if (!checkpoint_only) {
		    execute(start, tmp, tmp / "start.log");
        }
		int checks = 0;
		auto expect = [&](bool ok, const std::string &message) {
			require(ok, "TEST FAILED: " + message);
			++checks;
		};
        {
            const auto continued = tmp / "checkpoint batch";
            ::setenv("RR_TEST_CHECKPOINT", "1", 1);
            execute({runner.string(), "live", "subcritical", "0", "--root", root.string(),
                "--build", build.string(), "--output", continued.string(), "--threads", "1", "--time", "4",
                "--snapshots", "3", "--no-build", "--no-open", "--no-movies", "--gnuplot", argv[4]},
                tmp, tmp / "checkpoint.log");
            ::unsetenv("RR_TEST_CHECKPOINT");
            const auto folder = continued / "subcritical_shock/l0";
            const auto m = read_json(folder / "run.json");
            expect(m.at("status") == "complete" && m.at("checkpoint_resume").at("checkpoint_time") == 2 &&
                m.at("conservation")[0].at("normalization_time") == 2 && m.at("performance_eligible") == false,
                "full C++ live pipeline accepts and qualifies valid checkpoint segment before report");
            const auto page = read_text(continued / "subcritical_shock.html");
            expect(page.find("Checkpoint continuation conservation segment") != std::string::npos &&
                page.find("conservation-segments/original-attempt.csv") != std::string::npos &&
                page.find("tail after checkpoint is abandoned") != std::string::npos,
                "public report labels resumed segment and links preserved original raw ledger");
            const auto original = read_text(folder / "conservation-segments/original-attempt.csv");
            execute({runner.string(), "live", "--resume", continued.string(), "--no-open", "--no-movies", "--gnuplot", argv[4]},
                tmp, tmp / "checkpoint-resume.log");
            expect(read_text(folder / "solver-executions") == "run\n" &&
                read_text(folder / "conservation-segments/original-attempt.csv") == original,
                "completed checkpoint report recovery preserves ledger and does not execute solver again");
        }
        if (checkpoint_only) {
            std::cout << checks << " checkpoint pipeline checks passed; evidence " << tmp << '\n';
            return 0;
        }
        for (const auto& method : {"euler", "rk1"}) {
            const auto marshak_batch = tmp / (std::string("marshak-") + method);
            execute({runner.string(), "run", "marshak", "0", "--root", root.string(), "--build", build.string(),
                "--output", marshak_batch.string(), "--threads", "2", "--snapshots", "3", "--no-build", "--no-open", "--no-movies",
                "--radiation-integrator", method, "--gnuplot", argv[4]}, tmp, tmp / (std::string("marshak-") + method + ".log"));
            const auto m = read_json(marshak_batch / "marshak/l0/run.json");
            expect(m.at("radiation_integrator") == "euler" &&
                m.at("config").at("radiation_integrator") == "euler" &&
                m.at("comparison_signature").at("radiation_integrator") == "euler" &&
                read_json(marshak_batch / "batch.json").at("options").at("radiation_integrator") == "euler",
                std::string(method) + " selects the same canonical Euler method in all provenance");
            expect(m.at("units") == marshak_units && m.at("c") == 1. && m.at("time") == 1. && m.at("length") == 2.,
                "Marshak retains normalized defaults instead of CGS rescaling");
            expect(m.at("norms").contains("material_egas") && fs::is_regular_file(marshak_batch / "plots/marshak/l0/material_profiles.png"),
                "Marshak has material norms and an analytic comparison plot for each method");
            expect(read_text(marshak_batch / "plots/marshak/l0/slice.csv").find("material_egas_ref") != std::string::npos,
                "Marshak exported paired CSV includes the material reference");
            execute({runner.string(), "run", "--resume", marshak_batch.string(), "--no-open", "--no-movies", "--gnuplot", argv[4]}, tmp,
                tmp / (std::string("marshak-resume-") + method + ".log"));
            expect(read_text(marshak_batch / "marshak/l0/solver-executions") == "run\n", "Marshak resume reuses completed results");
        }
        for (const auto &name : {"gaussian_pulse", "marshak", "equilibrium_sphere", "streaming_wave", "streaming_front"}) {
            const bool references = fld_reference_case(name);
            const auto fld_batch = tmp / (std::string("fld-") + name);
            execute({runner.string(), "run", name, "0", "--root", root.string(), "--build", build.string(),
                "--output", fld_batch.string(), "--radiation-model", "fld", "--time", ".25", "--hard-dt", ".03125",
                "--threads", "2", "--snapshots", "3", "--no-build", "--no-open", "--no-movies",
                "--gnuplot", argv[4]}, tmp, tmp / (std::string("fld-") + name + ".log"));
            const auto folder = fld_batch / name / "l0";
            const auto m = read_json(folder / "run.json");
            const auto options = read_json(fld_batch / "batch.json").at("options");
            expect(m.at("status") == "complete" && m.at("radiation_model") == "fld" &&
                m.at("radiation_integrator") == "backward_euler" && m.at("radiation_algorithm_version") == 1 &&
                m.at("radiation_flux_semantics") == "derived_fld_flux" &&
                m.at("radiation_limiter") == "levermore_pomraning" &&
                options.at("radiation_model") == "fld" && options.at("radiation_integrator") == "backward_euler" &&
                options.at("radiation_flux_semantics") == "derived_fld_flux",
                std::string(name) + " FLD auto method and derived-flux provenance are explicit");
            expect(m.at("config").at("radiation_model") == "fld" &&
                m.at("config").at("disable_analytic") == "on" &&
                m.at("comparison_signature").at("radiation_model") == "fld" &&
                m.at("comparison_signature").at("radiation_algorithm_version") == 1 &&
                m.at("reference_kind") == (references ? "linear-diffusion-limit" : "unavailable") && !m.contains("norms") &&
                m.at("reference_exact_for_solver") == false &&
                m.at("fld_diagnostics").contains(references ? "slice_reference_norms" : "analytic_invariants"),
                "FLD verifies implicit diagnostics and uses only scientifically applicable references");
            expect(close(numeric(m.at("config").at("hard_dt")), .03125) &&
                m.at("config").at("disable_output") == "off" &&
                options.at("fld_output_each_step") == true &&
                m.at("movie_capture").at("steps_per_output_check") == 1,
                "FLD with snapshots retains the requested timestep without an implicit five-step output cap");
            const auto page = read_text(fld_batch / (std::string(name) + ".html"));
            expect(page.find("FLD") != std::string::npos &&
                page.find("Full-volume norms and orders") == std::string::npos &&
                (read_text(fld_batch / "plots" / name / "l0/slice.csv").find("er_ref") != std::string::npos) == references,
                "FLD report includes analytic columns only when a valid qualified reference is supplied");
            if (!references) {
                expect(page.find("No analytic spatial FLD solution is supplied") != std::string::npos &&
                    page.find("Projected-slice differences from the analytic reference") == std::string::npos &&
                    !fs::exists(fld_batch / "plots" / name / "l0/reference-differences.csv") &&
                    fs::is_regular_file(fld_batch / "plots" / name / "l0/analytic-invariants.json"),
                    "vacuum FLD reports analytic energy integrals without fabricated profile errors");
            }
            if (std::string(name) == "equilibrium_sphere") {
                expect(m.at("reference_model") == "steady-gaussian-emitter" &&
                    m.at("reference_sampling") == "cell averages" &&
                    page.find("Dirichlet") != std::string::npos,
                    "FLD sphere records its steady diffusion-limit reference and imposed energy boundary");
            }
            if (std::string(name) == "gaussian_pulse") {
                expect(fs::is_regular_file(folder / "reference.bin") && m.contains("reference_sha256") &&
                    m.contains("reference_generator_sha256") &&
                    m.at("reference_model") == "periodic-constant-diffusion" &&
                    close(m.at("reference_diffusion_coefficient"), c_cgs / numeric(m.at("config").at("rad_test_chi")) / 3.),
                    "Gaussian FLD records its constant-D reference equation and generator provenance");
            }
            execute({runner.string(), "run", "--resume", fld_batch.string(), "--radiation-integrator", "auto",
                "--no-open", "--no-movies", "--gnuplot", argv[4]}, tmp, tmp / (std::string("fld-resume-") + name + ".log"));
            expect(read_text(folder / "solver-executions") == "run\n",
                "FLD resume restores the model and reuses validated completed results");
            expect(execute({runner.string(), "run", "--resume", fld_batch.string(), "--radiation-model", "m1",
                "--dry-run"}, tmp, tmp / "fld-resume-mismatch.log", {}, false) != 0 &&
                read_text(tmp / "fld-resume-mismatch.log").find("saved radiation model") != std::string::npos,
                "FLD resume refuses to reinterpret the batch as M1");
        }
        for (const auto &bad_options : std::vector<std::vector<std::string>>{
                {"subcritical"}, {"supercritical"}, {"blast"},
                {"gaussian", "--radiation-integrator", "euler"},
                {"gaussian", "--radiation-backend", "cuda"},
                {"gaussian", "--radiation-reduced-c-factor", ".1"}}) {
            std::vector<std::string> cmd{runner.string(), "run", "--radiation-model", "fld", "--dry-run",
                "--root", root.string(), "--build", build.string(), "--output", (tmp / "invalid-fld-plan").string()};
            cmd.insert(cmd.end(), bad_options.begin(), bad_options.end());
            expect(execute(cmd, tmp, tmp / "invalid-fld-plan.log", {}, false) != 0 &&
                !fs::exists(tmp / "invalid-fld-plan"),
                "unsupported FLD case or physics is rejected before creating a batch");
        }
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
            "--gradient-refine-threshold", "1e-10", "--radiation-backend", "cuda",
            "--radiation-integrator", "euler", "--dry-run"},
            tmp, tmp / "ensman-amr-plan.log");
        expect(read_text(tmp / "ensman-amr-plan.log").find("adaptive leaves=1..2") != std::string::npos,
            "Ensman wrapper forwards value-taking AMR options without treating their values as levels");
        expect(read_text(tmp / "ensman-amr-plan.log").find("backend=cuda") != std::string::npos,
            "Ensman wrapper forwards the value-taking backend option");
        expect(read_text(tmp / "ensman-amr-plan.log").find("integrator=euler") != std::string::npos,
            "Ensman wrapper forwards the value-taking integrator option");
        const auto blast_position = plan_text.find("radiating_blast level=2 N=16 t=22000");
        const auto super_position = plan_text.find("supercritical_shock level=2 N=16 t=7500");
        const auto sub_position = plan_text.find("subcritical_shock level=2 N=16 t=38300");
        expect(blast_position != std::string::npos && super_position != std::string::npos &&
            sub_position != std::string::npos && blast_position < super_position &&
            super_position < sub_position && plan_text.find("streaming_wave level=") == std::string::npos,
            "Ensman launcher selects only three full-duration cases in the intended order");
        expect(!fs::exists(planned_batch), "Ensman dry-run does not create or run a batch");
        expect(execute({launcher.string(), "--radiation-model", "fld", "--dry-run"}, tmp,
            tmp / "ensman-fld.log", {}, false) != 0 &&
            read_text(tmp / "ensman-fld.log").find("stationary FLD is unavailable") != std::string::npos,
            "Ensman launcher refuses unsupported moving-material FLD cases before invoking the runner");
        expect(execute({launcher.string(), "--radiation-integrator", "ader2", "--dry-run"}, tmp,
            tmp / "ensman-ader.log", {}, false) != 0,
            "Ensman launcher exposes only the retained M1 Euler/RK1 scheme");
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
		expect(saved.at("options").at("radiation_backend") == "cpu", "runner defaults to the CPU backend");
        const auto new_wave = read_json(batch / "streaming_wave/l0/run.json");
        expect(saved.at("options").at("stationary_output_each_step") == true &&
            new_wave.at("movie_capture").at("steps_per_output_check") == 1 &&
            close(numeric(new_wave.at("config").at("hard_dt")), .4) &&
            close(new_wave.at("movie_capture").at("hard_dt"), .4) &&
            close(new_wave.at("movie_capture").at("odt"), 2.),
            "new stationary M1 checks output each step while retaining the historical odt/5 timestep cap");
        expect(saved.at("source_sha256").contains("src/radiation/iteration.cpp") &&
            saved.at("source_sha256").contains("src/radiation/fld.cpp") &&
            saved.at("source_sha256").contains("octotiger/radiation/fld.hpp") &&
            saved.at("source_sha256").contains("octotiger/radiation/fld_runtime.hpp") &&
            saved.at("source_sha256").contains("octotiger/elliptic/composite.hpp") &&
            !saved.at("source_sha256").contains("src/radiation/ader.cpp") &&
            !saved.at("source_sha256").contains("src/radiation/ader_predictor.cpp") &&
            !saved.at("source_sha256").contains("octotiger/radiation/relaxation_predictor.hpp"),
            "new provenance hashes current runtime sources without depending on removed ADER files");
		saved["workflow"] = "run_live.py";
		saved.erase("options");
		write_json(batch / "batch.json", saved);
        for (int level : {0, 1}) {
            const auto folder = batch / "streaming_wave" / ("l" + std::to_string(level));
            auto legacy = read_json(folder / "run.json");
            legacy.erase("radiation_backend");
            legacy.erase("radiation_model");
            legacy.erase("radiation_algorithm_version");
            legacy.erase("radiation_flux_semantics");
            legacy["config"].erase("radiation_backend");
            legacy["config"].erase("radiation_model");
            // This fixture now represents a pre-flag batch: those drivers
            // checked output after a five-step refinement batch. Its timestep
            // cap already matches the old policy and must remain unchanged.
            legacy["movie_capture"]["steps_per_output_check"] = 5;
            write_config(folder / "run.ini", legacy.at("config"));
            write_json(folder / "run.json", legacy);
        }
		execute(resume, tmp, tmp / "legacy-resume.log");
		executions();
        expect(read_json(batch / "streaming_wave/l0/run.json").at("radiation_backend") == "cpu",
            "legacy metadata and configs without a backend resume as CPU");
        expect(read_json(batch / "streaming_wave/l0/run.json").at("movie_capture").at("steps_per_output_check") == 5,
            "legacy resume preserves its recorded batched output cadence");
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
            "--radiation-backend", "cuda", "--hard-dt", "0.005"}, tmp, tmp / "short.log");
        const auto short_run = read_json(short_batch / "subcritical_shock/l0/run.json");
        expect(short_run.at("radiation_backend") == "cuda" && short_run.at("config").at("radiation_backend") == "cuda" &&
            read_config(short_batch / "subcritical_shock/l0/run.ini").at("radiation_backend") == "cuda" &&
            read_json(short_batch / "batch.json").at("options").at("radiation_backend") == "cuda" &&
            !short_run.at("comparison_signature").contains("radiation_backend"),
            "backend is frozen in batch/run provenance and solver config without changing the physics signature");
        expect(short_run.at("radiation_integrator") == "euler" &&
            short_run.at("config").at("radiation_integrator") == "euler" &&
            short_run.at("comparison_signature").at("radiation_integrator") == "euler" &&
            read_config(short_batch / "subcritical_shock/l0/run.ini").at("radiation_integrator") == "euler" &&
            read_json(short_batch / "batch.json").at("options").at("radiation_integrator") == "euler",
            "default Euler integrator is retained in run/batch/config and numerical comparisons");
        expect(short_run.at("radiation_model") == "m1" &&
            short_run.at("radiation_algorithm_version") == 1 &&
            short_run.at("radiation_flux_semantics") == "evolved_m1_flux" &&
            short_run.at("comparison_signature").at("radiation_model") == "m1" &&
            !short_run.at("comparison_signature").contains("radiation_ader_version") &&
            read_json(short_batch / "batch.json").at("options").at("radiation_algorithm_version") == 1,
            "new M1 results record model, algorithm version and evolved flux semantics");
        expect(read_text(short_batch / "subcritical_shock.html").find("M1 RK1") != std::string::npos,
            "generated report identifies the retained first-order integration method");
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
        expect(execute({runner.string(), "run", "--radiation-backend", "unknown", "--dry-run"},
            tmp, tmp / "bad-backend.log", {}, false) != 0 &&
            read_text(tmp / "bad-backend.log").find("--radiation-backend must be cpu, cuda, or kokkos") != std::string::npos,
            "invalid backend is rejected before planning or execution");
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
        expect(read_json(short_batch / "subcritical_shock/l0/run.json").at("radiation_backend") == "cuda",
            "resume restores the saved backend without requiring the flag again");
        execute({runner.string(), "run", "--resume", short_batch.string(), "--radiation-backend", "cuda",
            "--no-open", "--no-movies", "--gnuplot", argv[4]}, tmp, tmp / "backend-matching-resume.log");
        expect(read_text(short_batch / "subcritical_shock/l0/solver-executions") == "run\n",
            "matching explicit backend is accepted on resume without rerunning");
        expect(execute({runner.string(), "run", "--resume", short_batch.string(), "--radiation-backend", "cpu",
            "--dry-run"}, tmp, tmp / "backend-mismatch.log", {}, false) != 0 &&
            read_text(tmp / "backend-mismatch.log").find("uses the saved radiation backend") != std::string::npos,
            "resume refuses a different backend");
        expect(read_json(short_batch / "subcritical_shock/l0/run.json").at("radiation_integrator") == "euler",
            "resume restores the saved integrator without requiring the flag again");
        const auto current_euler_batch = read_json(short_batch / "batch.json");
        for (const std::string retired_method : {"ader2", "pdars", "imex2"}) {
            auto retired_batch = current_euler_batch;
            retired_batch["options"]["radiation_integrator"] = retired_method;
            write_json(short_batch / "batch.json", retired_batch);
            const auto log = tmp / (retired_method + "-resume.log");
            const int status = execute({runner.string(), "run", "--resume", short_batch.string(),
                "--dry-run"}, tmp, log, {}, false);
            expect(read_json(short_batch / "batch.json") == retired_batch,
                "retired batch rejection preserves original provenance");
            write_json(short_batch / "batch.json", current_euler_batch);
            expect(status != 0 && read_text(log).find("retired radiation method") != std::string::npos,
                retired_method + " saved batch is rejected");
            expect(execute({runner.string(), "run", "--radiation-integrator", retired_method, "--dry-run"},
                tmp, tmp / (retired_method + "-fresh.log"), {}, false) != 0,
                retired_method + " fresh selection is rejected instead of silently renamed");
        }
        execute({runner.string(), "run", "--resume", short_batch.string(), "--radiation-integrator", "rk1", "--dry-run"},
            tmp, tmp / "rk1-matching-resume.log");
        expect(read_json(short_batch / "batch.json") == current_euler_batch &&
            read_text(short_batch / "subcritical_shock/l0/solver-executions") == "run\n",
            "RK1 alias matches saved Euler provenance without rerunning the solver");
        expect(execute({runner.string(), "run", "--resume", short_batch.string(), "--radiation-integrator", "ader2",
            "--dry-run"}, tmp, tmp / "integrator-mismatch.log", {}, false) != 0,
            "resume refuses a different integrator");
        expect(execute({runner.string(), "run", "--radiation-integrator", "ader2", "--dry-run"},
            tmp, tmp / "ader-fresh.log", {}, false) != 0 &&
            read_text(tmp / "ader-fresh.log").find("ADER2 has been removed") != std::string::npos,
            "new ADER invocation is rejected before planning or solver execution");
        expect(execute({runner.string(), "run", "--radiation-integrator", "unknown", "--dry-run"},
            tmp, tmp / "integrator-invalid.log", {}, false) != 0,
            "unknown integrator is rejected");
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
        // A study resolves its relative AMR plan once, then freezes both JSON
        // documents so an aggregate adaptive batch can be resumed independently.
        const auto study_dir = tmp / "study inputs";
        const auto study_file = study_dir / "local-study.json";
        // This tests the retained historical AMR-study reader independently of
        // the current local M1/FLD preparation preset, which has another schema.
        const Json study_document{{"schema", 1}, {"amr_plan", "amr/local-thresholds.json"},
            {"reduced_c_factors", {{"subcritical_shock", .001}, {"supercritical_shock", .001},
                {"radiating_blast", .001}}}};
        const fs::path study_plan_relative = study_document.at("amr_plan").get<std::string>();
        expect(study_plan_relative.is_relative(), "study fixture exercises a relative AMR plan path");
        const auto study_plan = read_json(fs::path(argv[3]) / study_plan_relative);
        write_json(study_file, study_document);
        write_json(study_dir / study_plan_relative, study_plan);
        const auto study_batch = tmp / "study batch";
        execute({runner.string(), "live", "wave", "2", "3", "release", "--root", root.string(),
            "--output", study_batch.string(), "--study-config", study_file.string(), "--threads", "1",
            "--snapshots", "3", "--no-build", "--no-open", "--no-movies", "--no-silo",
            "--radiation-backend", "kokkos", "--gnuplot", argv[4]}, tmp, tmp / "study.log");
        const auto frozen_batch = read_json(study_batch / "batch.json");
        expect(frozen_batch.at("study_config").at("document") == study_document &&
            frozen_batch.at("study_config").at("amr_plan") == study_plan &&
            frozen_batch.at("study_config_sha256").get<std::string>().size() == 64,
            "batch freezes the complete study and AMR plan with a digest");
        double prior_threshold = 0;
        for (int level : {2, 3}) {
            const auto folder = study_batch / "streaming_wave" / ("l" + std::to_string(level));
            const auto run = read_json(folder / "run.json");
            double sampled_threshold = 0;
            for (const auto &row : study_plan.at("rows")) {
                if (row.at("case") == "streaming_wave" && row.at("finest_level") == level) {
                    sampled_threshold = row.at("selected").at("threshold");
                }
            }
            const double threshold = numeric(run.at("config").at("gradient_refine_threshold"));
            expect(sampled_threshold > 0 && close(threshold * half_width, sampled_threshold, 1e-12) &&
                threshold != prior_threshold && run.at("adaptive") == true && run.at("min_level") == level - 1 &&
                close(run.at("radiation_reduced_c_factor"), 1.) &&
                run.at("radiation_backend") == "kokkos" && run.at("config").at("radiation_backend") == "kokkos" &&
                run.at("study_config_sha256") == frozen_batch.at("study_config_sha256"),
                "study preserves distinct per-level AMR controls and converts the transport gradient once");
            expect(read_text(folder / "solver-executions") == "run\n", "study solver executed once per level");
            prior_threshold = threshold;
        }
        const auto study_page = read_text(study_batch / "streaming_wave.html");
        expect(study_page.find("id=\"level-2\"") != std::string::npos &&
            study_page.find("id=\"level-3\"") != std::string::npos &&
            fs::file_size(study_batch / "plots/streaming_wave/l2/slice_er.png") > 100 &&
            fs::file_size(study_batch / "plots/streaming_wave/l3/slice_er.png") > 100 &&
            !read_json(study_batch / "live.json").at("active").get<bool>(),
            "one completed study page aggregates both adaptive resolutions with different thresholds");
        expect(execute({runner.string(), "live", "wave", "2", "release", "--root", root.string(),
            "--study-config", study_file.string(), "--gradient-refine-threshold", "1", "--dry-run"},
            tmp, tmp / "study-conflict.log", {}, false) != 0 &&
            read_text(tmp / "study-conflict.log").find("cannot be combined with direct AMR") != std::string::npos,
            "study rejects a conflicting direct AMR flag");
        expect(execute({runner.string(), "live", "wave", "2", "release", "--root", root.string(),
            "--study-config", study_file.string(), "--radiation-reduced-c-factor", "0.5", "--dry-run"},
            tmp, tmp / "study-rsla-conflict.log", {}, false) != 0 &&
            read_text(tmp / "study-rsla-conflict.log").find("cannot be combined with direct AMR") != std::string::npos,
            "study rejects a conflicting direct reduced-light-speed flag");
        fs::remove_all(study_dir);
        const std::vector<std::string> study_resume{runner.string(), "live", "--resume", study_batch.string(),
            "--threads", "1", "--no-open", "--no-movies", "--gnuplot", argv[4]};
        execute(study_resume, tmp, tmp / "study-resume.log");
        expect(read_json(study_batch / "batch.json").at("study_config") == frozen_batch.at("study_config"),
            "resume uses the frozen study after its external configuration and AMR plan are removed");
        expect(read_json(study_batch / "batch.json").at("options").at("radiation_backend") == "kokkos",
            "study backend remains an independent frozen execution option");
        for (int level : {2, 3}) {
            expect(read_text(study_batch / "streaming_wave" / ("l" + std::to_string(level)) /
                "solver-executions") == "run\n", "study resume reuses every completed level without rerunning");
        }
        auto corrupted_study = read_json(study_batch / "batch.json");
        corrupted_study["study_config"]["amr_plan"]["rows"][0]["selected"]["threshold"] = 123.;
        write_json(study_batch / "batch.json", corrupted_study);
        expect(execute(study_resume, tmp, tmp / "study-corrupted.log", {}, false) != 0 &&
            read_text(tmp / "study-corrupted.log").find("Frozen study configuration digest differs") != std::string::npos,
            "resume rejects mutation of the frozen study before accepting its saved runs");
		fs::remove_all(tmp);
		std::cout << checks << " integration checks passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\nTest files retained at " << tmp << '\n';
		return 1;
	}
}
