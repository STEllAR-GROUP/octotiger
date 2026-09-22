#include "common.hpp"
using namespace rr;
int main(int argc, char **argv)
{
	fs::path tmp;
	try {
		require(argc == 6, "Usage: rr-integration RUNNER FIXTURE_EXE SOURCE_DIR GNUPLOT CMAKE");
		auto runner = rr::absolute(argv[1]);
		tmp = fs::temp_directory_path() / ("rr integration " + stamp());
		auto root = tmp / "project", source = root / "src/octotiger",
			 build = root / "build/octotiger/release", batch = tmp / "batch";
		fs::create_directories(build);
		for (auto name :
				 {"src/grid.cpp", "src/physcon.cpp", "src/radiation/rad_grid.cpp",
				  "octotiger/test_problems/radiation/profiles.hpp",
				  "octotiger/test_problems/radiation/plot_output.hpp", "octotiger/radiation/conservation.hpp",
				  "octotiger/radiation/rad_grid.hpp", "src/node_server_actions_3.cpp"})
			atomicText(source / name, "// RADIATION_PLOT_EXPORT_BEGIN: integration fixture\n");
		atomicText(source / "CMakeLists.txt",
					"cmake_minimum_required(VERSION 3.16)\n"
					"project(runner_fixture NONE)\nset(OCTOTIGER_WITH_GRIDDIM 4 CACHE STRING "
					"\"Grid size\")\n"
					"add_custom_target(octotiger COMMAND ${CMAKE_COMMAND} -E echo \"Mock solver "
					"already installed\")\n");
		execute({argv[5], "-S", source.string(), "-B", build.string()}, tmp);
		for (const auto &name : cases)
			atomicText(source / "verification_results/radiation/configs" / (name + ".ini"),
						readText(fs::path(argv[3]) / "configs" / (name + ".ini")));
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
		expect(!readJson(batch / "live.json").at("active").get<bool>(), "batch completed");
		expect(fs::file_size(batch / "plots/streaming_wave/convergence.png") > 100, "convergence rendered");
		for (auto &name : cases)
			expect(fs::is_regular_file(batch / (name + ".html")), "dedicated problem page: " + name);
		expect(readText(batch / "gaussian_pulse.html").find("telegraph equation") != std::string::npos,
			   "Gaussian page identifies the telegraph reference before results exist");
		auto wavePage = readText(batch / "streaming_wave.html");
		expect(wavePage.find("id=\"level-0\"") != std::string::npos &&
				   wavePage.find("id=\"level-1\"") != std::string::npos &&
				   readJson(batch / "plots/streaming_wave/errors.json").size() == 24,
			   "problem page includes every resolution and all four field norms");
		auto plotTime = fs::last_write_time(batch / "plots/streaming_wave/l0/slice_er.png");
		execute({runner.string(), "pages", batch.string(), "--gnuplot", "/missing/gnuplot", "--ffmpeg",
				 "/missing/ffmpeg"},
				tmp, tmp / "pages.log");
		expect(fs::last_write_time(batch / "plots/streaming_wave/l0/slice_er.png") == plotTime,
			   "HTML-only update needs no plotting tools and does not regenerate images");
		auto executions = [&] {
			for (int level : {0, 1})
				expect(readText(batch / "streaming_wave" / ("l" + std::to_string(level)) /
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
		auto saved = readJson(batch / "batch.json");
		saved["workflow"] = "run_live.py";
		saved.erase("options");
		writeJson(batch / "batch.json", saved);
		execute(resume, tmp, tmp / "legacy-resume.log");
		executions();
		auto radiationSource = source / "src/radiation/rad_grid.cpp";
		auto original = readText(radiationSource);
		atomicText(radiationSource, original + "// changed\n");
		expect(execute(resume, tmp, tmp / "changed-source.log", {}, false) != 0 &&
				   readText(tmp / "changed-source.log").find("Source changed") != std::string::npos,
			   "changed source rejected");
		atomicText(radiationSource, original);
		{
			std::ofstream out(solver, std::ios::app);
			out << "changed";
		}
		expect(execute(resume, tmp, tmp / "changed-exe.log", {}, false) != 0 &&
				   readText(tmp / "changed-exe.log").find("Executable changed") != std::string::npos,
			   "changed executable rejected");
		fs::copy_file(rr::absolute(argv[2]), solver, fs::copy_options::overwrite_existing);
		auto runFile = batch / "streaming_wave/l0/run.json";
		auto meta = readJson(runFile);
		meta["time"] = 5.;
		writeJson(runFile, meta);
		expect(execute(resume, tmp, tmp / "changed-meta.log", {}, false) != 0 &&
				   readText(tmp / "changed-meta.log").find("Saved run metadata differs: time") !=
					   std::string::npos,
			   "mismatched run rejected");
		executions();
		// Exercise the actual failing path: build all cases in an application project
		// with ONLY an octotiger target, then invoke the real bundled FFTW generator.
		auto allBatch = tmp / "all batch";
		execute({runner.string(), "live", "all", "0", "release", "--root", root.string(), "--output",
				 allBatch.string(), "--threads", "2", "--snapshots", "3", "--no-open", "--no-movies",
				 "--gnuplot", argv[4]},
				tmp, tmp / "all.log");
		expect(!readJson(allBatch / "live.json").at("active").get<bool>(), "all-case batch completed");
		for (const auto &name : cases)
			expect(readJson(allBatch / name / "l0/run.json").at("status") == "complete",
				   "all-case run completed: " + name);
		auto gaussian = allBatch / "gaussian_pulse/l0";
		auto display = readJson(allBatch / "plots/gaussian_pulse/l0/display.json");
		expect(display.at("energy_scale") == "log" && display.at("energy_floor").get<double>() > 0 &&
				   display.at("energy_floor").get<double>() < display.at("energy_maximum").get<double>(),
			   "Gaussian energy renders a finite logarithmic scale despite nonpositive excess samples");
		expect(fs::file_size(gaussian / "reference.bin") == 76 + 64 * 4 * 4 * 4,
			   "real generator wrote the full reference file");
		expect(readText(allBatch / "build.log").find("gen_radiation_reference") == std::string::npos,
			   "application build does not request reference-generator target");
		expect(readJson(allBatch / "batch.json").at("options").at("generator") ==
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
