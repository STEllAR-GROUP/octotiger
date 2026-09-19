#include "common.hpp"
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace rr;
int main(int argc, char **argv)
{
	fs::path tmp;
	try {
		require(argc == 5, "Usage: rr-terminal RUNNER FIXTURE_EXE GNUPLOT FFMPEG");
		tmp = fs::temp_directory_path() / ("rr-terminal-" + stamp());
		fs::create_directories(tmp);
		execute({argv[2], "--fixtures", tmp.string()}, tmp);
		auto folder = tmp / "streaming_wave/l0";
		auto image = tmp / "test.png";
		Options plot_options;
		plot_options.gnuplot = argv[3];
		gnuplot_script(tmp / "test.gnuplot",
					   "set terminal pngcairo size 256,256\nset output " + gp_quote(image.string()) +
						   "\nplot sin(x)\nunset output\n",
					   plot_options);
		setenv("RR_TEST_FRAME", image.c_str(), 1);
		auto fake_visit = tmp / "visit";
		atomic_text(fake_visit, "#!/bin/sh\nexec " + shell_quote(argv[2]) + " --fake-visit \"$@\"\n");
		fs::permissions(fake_visit, fs::perms::owner_exec, fs::perm_options::add);
		std::vector<std::string> command{argv[1],
										 "movies",
										 folder.string(),
										 "--visit",
										 fake_visit.string(),
										 "--session-dir",
										 (tmp / "visit_sessions").string(),
										 "--ffmpeg",
										 argv[4],
										 "--seconds",
										 "3",
										 "--fps",
										 "4",
										 "--hold",
										 ".25",
										 "--width",
										 "256",
										 "--height",
										 "256"};
		std::vector<char *> arguments;
		for (auto &s : command)
			arguments.push_back(s.data());
		arguments.push_back(nullptr);
		int master;
		std::cout << std::flush;
		pid_t pid = forkpty(&master, nullptr, nullptr, nullptr);
		require(pid >= 0, "forkpty failed");
		if (pid == 0) {
			try {
				// This command runs under a real controlling terminal, like the user's shell.
				execute({"/bin/sh", "-c", "test ! -t 0"}, tmp);
				execv(arguments[0], arguments.data());
			} catch (...) {
			}
			_exit(127);
		}
		std::string output;
		int status = 0;
		bool done = false;
		auto wait_until = [&](auto deadline) {
			while (!done && std::chrono::steady_clock::now() < deadline) {
				pollfd p{master, POLLIN | POLLHUP, 0};
				if (poll(&p, 1, 100) > 0) {
					char data[16384];
					auto n = read(master, data, sizeof(data));
					if (n > 0)
						output.append(data, n);
				}
				done = waitpid(pid, &status, WNOHANG) == pid;
			}
		};
		wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(20));
		bool timely = done;
		if (!done) {
			kill(pid, SIGINT);
			wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(4));
			if (!done) {
				kill(pid, SIGKILL);
				waitpid(pid, &status, 0);
			}
		}
		fcntl(master, F_SETFL, O_NONBLOCK);
		char remaining[16384];
		ssize_t count;
		while ((count = read(master, remaining, sizeof(remaining))) > 0)
			output.append(remaining, count);
		::close(master);
		atomic_text(tmp / "terminal.log", output);
		require(timely && WIFEXITED(status) && WEXITSTATUS(status) == 0,
				"Encoding stopped or failed under a controlling terminal; inspect terminal.log");
		require(output.find("Encoding 12/12 frames (100%)") != std::string::npos,
				"Missing terminal encoding progress");
		auto movie = folder / "movies/er-slice-z";
		require(fs::file_size(movie / "movie.mp4") > 500, "Missing MP4");
		require(read_text(movie / "encode.log").find("progress=end") != std::string::npos,
				"Missing FFmpeg completion record");

		setenv("RR_TEST_VISIT_MODE", "shutdown", 1);
		execute(command, tmp, tmp / "shutdown.log", {}, true, false);
		require(read_json(movie / "movie.json").at("visit_exit_code") == 250,
				"Validated VisIt shutdown exception must be recorded");
		auto preserved_movie = sha256(movie / "movie.mp4");
		for (auto mode : {"error", "partial", "corrupt"}) {
			setenv("RR_TEST_VISIT_MODE", mode, 1);
			require(execute(command, tmp, tmp / (std::string(mode) + ".log"), {}, false, false) != 0,
					"Failed VisIt render must be rejected");
			require(sha256(movie / "movie.mp4") == preserved_movie,
					"A failed render must preserve the previous movie");
		}
		unsetenv("RR_TEST_VISIT_MODE");

		// Reproduce gnuplot's near-constant CGS totals, including zero residuals.
		std::ostringstream history;
		history << "t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,"
				   "er_source,fx_source,fy_source,fz_source\n"
				<< std::setprecision(17);
		for (double t : {0., 1., 4.}) {
			double flux = t == 0 ? -5.19196e42 : std::nextafter(-5.19196e42, 0.);
			history << t << ",2.16e32,2.16e32,3.46131e42," << flux << ",0,0,0,0,0,0,0,0,0\n";
		}
		atomic_text(folder / "radiation-conservation.csv", history.str());
		execute({argv[1], "plot", folder.string(), "--gnuplot", argv[3]}, tmp, tmp / "plot.log");
		auto plot = folder / "plots/streaming_wave/l0/conservation.gnuplot";
		execute({argv[3], plot.string()}, tmp, tmp / "conservation-plot.log");
		require(read_text(tmp / "conservation-plot.log").empty(), "Conservation plot emitted warnings");
		fs::remove_all(tmp);
		std::cout << "Terminal encoding, progress, and conservation-axis regression passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << "\nTest files retained at " << tmp << '\n';
		return 1;
	}
}
