#include "common.hpp"
#include <thread>
namespace rr
{
Options arguments(int argc, char **argv)
{
	Options o;
	auto here = fs::canonical("/proc/self/exe").parent_path();
	o.root = here.filename() == ".build" ? here.parent_path().parent_path().parent_path().parent_path()
										 : fs::current_path();
	o.jobs = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
	int start = 1;
	if (argc > 1 && std::set<std::string>{"live", "run", "plot", "pages", "movies", "sessions", "install",
										  "validate-serial", "check"}
						.contains(argv[1])) {
		o.command = argv[1];
		start = 2;
	}
	std::vector<std::string> positional;
	auto integer = [](const std::string &s) {
		double v = numeric(s);
		require(v == std::floor(v) && v >= 0 && v <= 1000000, "Expected a nonnegative integer: " + s);
		return int(v);
	};
	for (int i = start; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--help" || arg == "-h") {
			std::cout
				<< R"(radiation-results [live|run] [all|wave|front|gaussian|sphere] [levels...] [build] [options]
radiation-results plot BATCH [--gnuplot PATH]
radiation-results pages BATCH_OR_RUN  (refresh HTML only; reuse existing plots and movies)
radiation-results movies BATCH_OR_RUN [options]
radiation-results sessions BATCH_OR_RUN [options]  (write VisIt sessions without rendering)
radiation-results check BATCH_OR_RUN
radiation-results install [--root PROJECT] [--check]
radiation-results validate-serial --root PROJECT --reference FILE --output DIR [--cells N] [--sanitize]

Run options: --root PATH --build PATH --exe PATH --generator PATH --output DIR
 --resume BATCH --threads N --jobs N --time SECONDS --snapshots N --odt SECONDS
 --hard-dt SECONDS --rad-reconstruction plm|ppm --no-build --no-open --dry-run
 --no-silo --no-movies
Movie options: --seconds 20 --fps 30 --hold 1 --field er|fx|fy|fz|fluxmag
 --view slice|3d --axis x|y|z --position 0 --width 1280 --height 960
 --color hot|viridis|gray|diverging --minimum X --maximum X --reuse-frames --allow-sparse
 --session-dir DIR (default: PROJECT/verification_results/movies)
Tools: --gnuplot PATH --visit PATH --ffmpeg PATH --cxx PATH
Defaults: live = all 2 3 4 debug, 12 threads, 4 s, 61 snapshots.
         run = all 2 3 release, same CGS units, plots only.
Resume uses saved simulation settings; specify only playback/thread options.
)";
			std::exit(0);
		}
		if (!arg.starts_with("--")) {
			positional.push_back(arg);
			continue;
		}
		auto eq = arg.find('=');
		std::string key = arg.substr(0, eq);
		o.explicit_options.insert(key);
		auto value = [&]() {
			if (eq != arg.npos)
				return arg.substr(eq + 1);
			require(++i < argc, "Missing value for " + key);
			return std::string(argv[i]);
		};
		auto flag = [&](bool &b) {
			require(eq == arg.npos, "Flag takes no value: " + key);
			b = true;
		};
		if (key == "--root")
			o.root = value();
		else if (key == "--build")
			o.build = value();
		else if (key == "--exe")
			o.exe = value();
		else if (key == "--generator")
			o.generator = value();
		else if (key == "--output")
			o.output = value();
		else if (key == "--resume")
			o.resume = value();
		else if (key == "--reference")
			o.reference = value();
		else if (key == "--threads")
			o.threads = integer(value());
		else if (key == "--jobs")
			o.jobs = integer(value());
		else if (key == "--snapshots")
			o.snapshots = integer(value());
		else if (key == "--fps")
			o.fps = integer(value());
		else if (key == "--width")
			o.width = integer(value());
		else if (key == "--height")
			o.height = integer(value());
		else if (key == "--cells")
			o.cells = integer(value());
		else if (key == "--time")
			o.time = numeric(value());
		else if (key == "--seconds")
			o.seconds = numeric(value());
		else if (key == "--hold")
			o.hold = numeric(value());
		else if (key == "--position")
			o.position = numeric(value());
		else if (key == "--odt")
			o.odt = numeric(value());
		else if (key == "--hard-dt")
			o.hard_dt = numeric(value());
		else if (key == "--minimum")
			o.minimum = numeric(value());
		else if (key == "--maximum")
			o.maximum = numeric(value());
		else if (key == "--gnuplot")
			o.gnuplot = value();
		else if (key == "--ffmpeg")
			o.ffmpeg = value();
		else if (key == "--cxx")
			o.cxx = value();
		else if (key == "--field")
			o.field = value();
		else if (key == "--view")
			o.view = value();
		else if (key == "--axis")
			o.axis = value();
		else if (key == "--color")
			o.color = value();
		else if (key == "--rad-reconstruction")
			o.reconstruction = lower(value());
		else if (key == "--no-build")
			flag(o.no_build);
		else if (key == "--no-open")
			flag(o.no_open);
		else if (key == "--dry-run")
			flag(o.dry_run);
		else if (key == "--no-silo")
			flag(o.no_silo);
		else if (key == "--no-movies")
			flag(o.no_movies);
		else if (key == "--reuse-frames")
			flag(o.reuse_frames);
		else if (key == "--allow-sparse")
			flag(o.allow_sparse);
		else if (key == "--check")
			flag(o.check);
		else if (key == "--sanitize")
			flag(o.sanitize);
		else if (key == "--visit")
			o.visit = value();
		else if (key == "--session-dir")
			o.session_dir = value();
		else
			throw std::runtime_error("Unknown option: " + key);
	}
	o.root = rr::absolute(o.root);
	require(o.time > 0 && o.threads > 0 && o.jobs > 0 && o.snapshots >= 3 && o.fps > 0 && o.width >= 128 &&
				o.height >= 128,
			"Invalid time, thread count, capture count or movie dimensions");
	require(o.odt >= 0 && o.hard_dt >= 0, "Negative timestep/interval");
	require(!o.explicit_options.contains("--odt") || o.odt > 0, "--odt must be positive");
	require(!o.explicit_options.contains("--hard-dt") || o.hard_dt > 0, "--hard-dt must be positive");
	frame_schedule({0, 1}, o.seconds, o.fps, o.hold);
	require(std::set<std::string>{"er", "fx", "fy", "fz", "fluxmag"}.contains(o.field),
			"Unknown movie field");
	require(o.view == "slice" || o.view == "3d", "Unknown movie view");
	require(o.axis == "x" || o.axis == "y" || o.axis == "z", "Unknown slice axis");
	require(o.reconstruction.empty() || o.reconstruction == "plm" || o.reconstruction == "ppm",
			"Unknown radiation reconstruction");
	if (o.command == "live" || o.command == "run") {
		if (!positional.empty())
			o.explicit_options.insert("positionals");
		std::string name = "all";
		if (!positional.empty()) {
			name = lower(positional.front());
			positional.erase(positional.begin());
		}
		std::map<std::string, std::string> aliases{
			{"wave", "streaming_wave"},		  {"front", "streaming_front"},
			{"streaming", "streaming_front"}, {"diffusion", "gaussian_pulse"},
			{"gaussian", "gaussian_pulse"},	  {"sphere", "equilibrium_sphere"}};
		if (aliases.contains(name))
			name = aliases[name];
		if (name == "all")
			o.selected_cases.assign(cases.begin(), cases.end());
		else {
			require(std::find(cases.begin(), cases.end(), name) != cases.end(), "Unknown case: " + name);
			o.selected_cases = {name};
		}
		if (!positional.empty() && !std::all_of(positional.back().begin(), positional.back().end(),
												[](unsigned char c) { return std::isdigit(c); })) {
			require(o.build.empty(), "Specify build once");
			o.build = positional.back();
			positional.pop_back();
		}
		if (o.build.empty())
			o.build = o.command == "live" ? "debug" : "release";
		auto build_name = lower(o.build.string());
		if (std::set<std::string>{"debug", "release", "relwithdebinfo"}.contains(build_name)) {
			std::string canonical = build_name == "debug"	  ? "Debug"
									: build_name == "release" ? "Release"
															  : "RelWithDebInfo";
			auto base = fs::path("build") / "octotiger";
			if (fs::is_directory(o.root / base / build_name))
				o.build = base / build_name;
			else if (fs::is_directory(o.root / base / canonical))
				o.build = base / canonical;
			else if (fs::is_directory(o.root / build_name))
				o.build = build_name;
			else if (fs::is_directory(o.root / canonical))
				o.build = canonical;
			else
				o.build = base / build_name;
		}
		for (auto &s : positional) {
			int level = integer(s);
			require(level <= 9, "Levels must be 0 through 9");
			o.levels.push_back(level);
		}
		if (o.levels.empty())
			o.levels = o.command == "live" ? std::vector<int>{2, 3, 4} : std::vector<int>{2, 3};
		std::sort(o.levels.begin(), o.levels.end());
		o.levels.erase(std::unique(o.levels.begin(), o.levels.end()), o.levels.end());
		if (o.command == "run")
			o.no_movies = true;
	} else if (o.command == "plot" || o.command == "pages" || o.command == "movies" ||
			   o.command == "sessions" || o.command == "check") {
		require(positional.size() == 1, "Provide one batch/run directory");
		o.input = rr::absolute(positional[0]);
	} else {
		require(positional.size() <= 1, "Too many positional arguments");
		if (!positional.empty())
			o.root = rr::absolute(positional[0]);
	}
	return o;
}
} // namespace rr
#ifndef RR_NO_MAIN
int main(int argc, char **argv)
{
	try {
		rr::install_signals();
		auto o = rr::arguments(argc, argv);
		if (o.command == "run" || o.command == "live")
			return rr::run(o);
		if (o.command == "plot") {
			std::cout << rr::render(o.input, o) << '\n';
			return 0;
		}
		if (o.command == "pages") {
			std::cout << rr::report_pages(o.input) << '\n';
			return 0;
		}
		if (o.command == "movies" || o.command == "sessions") {
			if (o.command == "movies")
				rr::check_dependencies(o, true);
			auto runs = rr::completed_runs(o.input);
			rr::require(!runs.empty(), "No completed runs");
			for (auto &[p, m] : runs) {
				if (o.command == "sessions")
					rr::make_session(p, o);
				else {
					rr::make_movie(p, o);
					rr::movie_index(o.input);
				}
			}
			return 0;
		}
		if (o.command == "check") {
			auto runs = rr::completed_runs(o.input);
			rr::require(!runs.empty(), "No completed runs");
			for (auto &[p, m] : runs) {
				rr::read_norms(p, m);
				rr::read_slice(p, m);
				rr::read_conservation(p, m);
				if (rr::cgs(m))
					rr::verify_cgs_log(p / "run.log");
				std::cout << "Verified " << p << '\n';
			}
			return 0;
		}
		return rr::maintenance(o);
	} catch (const std::exception &e) {
		std::cerr << "radiation-results: " << e.what() << '\n';
		return std::string(e.what()).starts_with("Interrupted") ? 130 : 1;
	}
}
#endif
