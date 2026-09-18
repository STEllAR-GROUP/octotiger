#include "common.hpp"
using namespace rr;
int main(int argc, char **argv)
{
	fs::path tmp;
	try {
		require(argc == 6, "Usage: rr-integration RUNNER FIXTURE_EXE SOURCE_DIR GNUPLOT CMAKE");
		auto runner = rr::absolute(argv[1]);
		tmp = fs::temp_directory_path() / ("rr integration " + stamp());
		auto root = tmp / "checkout", batch = tmp / "batch";
		fs::create_directories(root / "release");
		for (auto name :
			 {"src/grid.cpp", "src/physcon.cpp", "src/radiation/rad_grid.cpp",
			  "octotiger/test_problems/radiation/profiles.hpp",
			  "octotiger/test_problems/radiation/plot_output.hpp", "octotiger/radiation/conservation.hpp",
			  "octotiger/radiation/rad_grid.hpp", "src/node_server_actions_3.cpp"})
			atomic_text(root / name, "// RADIATION_PLOT_EXPORT_BEGIN: integration fixture\n");
		atomic_text(root / "CMakeLists.txt",
					"cmake_minimum_required(VERSION 3.16)\n"
					"project(runner_fixture NONE)\nset(OCTOTIGER_WITH_GRIDDIM 4 CACHE STRING "
					"\"Grid size\")\n"
					"add_custom_target(octotiger COMMAND ${CMAKE_COMMAND} -E echo \"Mock solver "
					"already installed\")\n");
		execute({argv[5], "-S", root.string(), "-B", (root / "release").string()}, tmp);
		for (const auto &name : cases)
			atomic_text(root / "test_results/configs" / (name + ".ini"),
						read_text(fs::path(argv[3]) / "configs" / (name + ".ini")));
		auto solver = root / "release/octotiger";
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
		expect(!read_json(batch / "live.json").at("active").get<bool>(), "batch completed");
		expect(fs::file_size(batch / "plots/streaming_wave/convergence.png") > 100, "convergence rendered");
		for (auto &name : cases)
			expect(fs::is_regular_file(batch / (name + ".html")), "dedicated problem page: " + name);
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
			for (int level : {0, 1})
				expect(read_text(batch / "streaming_wave" / ("l" + std::to_string(level)) /
								 "solver-executions") == "run\n",
					   "completed solver executed exactly once");
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
		auto source = root / "src/radiation/rad_grid.cpp";
		auto original = read_text(source);
		atomic_text(source, original + "// changed\n");
		expect(execute(resume, tmp, tmp / "changed-source.log", {}, false) != 0 &&
				   read_text(tmp / "changed-source.log").find("Source changed") != std::string::npos,
			   "changed source rejected");
		atomic_text(source, original);
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
				 "--gnuplot", argv[4]},
				tmp, tmp / "all.log");
		expect(!read_json(all_batch / "live.json").at("active").get<bool>(), "all-case batch completed");
		for (const auto &name : cases)
			expect(read_json(all_batch / name / "l0/run.json").at("status") == "complete",
				   "all-case run completed: " + name);
		auto gaussian = all_batch / "gaussian_pulse/l0";
		auto display = read_json(all_batch / "plots/gaussian_pulse/l0/display.json");
		expect(display.at("energy_scale") == "log" && display.at("energy_floor").get<double>() > 0 &&
				   display.at("energy_floor").get<double>() < display.at("energy_maximum").get<double>(),
			   "Gaussian energy renders a finite logarithmic scale despite nonpositive excess samples");
		expect(fs::file_size(gaussian / "reference.bin") == 76 + 64 * 4 * 4 * 4,
			   "real generator wrote the full reference file");
		expect(read_text(all_batch / "build.log").find("gen_radiation_reference") == std::string::npos,
			   "application build does not request reference-generator target");
		expect(read_json(all_batch / "batch.json").at("options").at("generator") ==
				   (runner.parent_path() / "gen_radiation_reference").string(),
			   "bundled generator selected");
		fs::remove_all(tmp);
		std::cout << checks << " integration checks passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\nTest files retained at " << tmp << '\n';
		return 1;
	}
}
