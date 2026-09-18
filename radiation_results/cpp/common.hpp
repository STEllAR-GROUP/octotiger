#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace rr
{
namespace fs = std::filesystem;
using Json = nlohmann::json;
inline constexpr double c_cgs = 2.99792458e10, half_width = 3e10;
inline const std::array<std::string, 4> fields{"er", "fx", "fy", "fz"};
inline const std::array<std::string, 3> norms{"L1", "L2", "Linf"};
inline const std::array<std::string, 4> cases{"streaming_wave", "streaming_front", "gaussian_pulse",
											  "equilibrium_sphere"};
inline const Json cgs_units{{"system", "CGS"},
							{"length", "cm"},
							{"time", "s"},
							{"mass", "g"},
							{"energy_density", "erg/cm^3"},
							{"flux", "erg/(cm^2 s)"},
							{"chi", "1/cm"},
							{"luminosity", "erg/s"},
							{"source", "erg/(cm^3 s)"}};
void require(bool, const std::string &);
std::string read_text(const fs::path &);
void atomic_text(const fs::path &, const std::string &);
Json read_json(const fs::path &);
void write_json(const fs::path &, const Json &);
std::string sha256(const fs::path &);
std::string sha256_text(const std::string &);
std::string trim(std::string);
std::string lower(std::string);
std::string number(double);
double numeric(const std::string &);
bool close(double, double, double = 1e-11, double = 0);
std::string stamp();
std::string shell_quote(const std::string &);
std::string gp_quote(const std::string &);
std::string html(const std::string &);
std::string url(const std::string &);
std::string page_start(const std::string &);
fs::path absolute(fs::path);
fs::path source_root(const fs::path &);
fs::path tools_root(const fs::path &);
fs::path executable(const std::string &);
void install_signals();
void check_interrupt();
int execute(const std::vector<std::string> &, const fs::path &, const fs::path & = {},
			const std::function<void(const std::string &)> & = {}, bool = true, bool = true);
void execute_detached(const std::vector<std::string> &, const fs::path &);
Json read_config(const fs::path &);
void write_config(const fs::path &, const Json &);
void records_csv(const fs::path &, const Json &, std::vector<std::string> = {});
struct Table {
	std::vector<std::string> columns;
	std::vector<std::vector<double>> rows;
};
Table read_csv(const fs::path &, const std::vector<std::string> &);
Json read_norms(const fs::path &, const Json &);
Table read_slice(const fs::path &, const Json &);
struct Budget {
	Table history;
	Json summary = Json::array();
	std::vector<std::array<double, 4>> residual, scale, normalized;
};
std::optional<Budget> read_conservation(const fs::path &, const Json &);
void verify_cgs_log(const fs::path &);
struct CGSCheck {
	bool pending = false, factors = false;
	int blocks = 0;
	void feed(const std::string &);
	void finish();
};
Json cadence(double, int, double, std::optional<double> = {});
std::vector<std::size_t> frame_schedule(const std::vector<double> &, double, int, double);
std::vector<fs::path> numerical_silos(const fs::path &);
struct Options {
	std::string command = "live", gnuplot = "gnuplot", ffmpeg = "ffmpeg", field = "er", view = "slice",
				axis = "z", color = "hot", cxx = "g++", visit = "visit";
	fs::path root, build, exe, generator, output, resume, input, reference, session_dir;
	std::vector<std::string> selected_cases;
	std::vector<int> levels;
	double time = 4, seconds = 20, hold = 1, position = 0, odt = 0, hard_dt = 0;
	int threads = 12, jobs = 8, snapshots = 61, fps = 30, width = 1280, height = 960, cells = 32;
	bool no_build = false, no_open = false, dry_run = false, no_silo = false, no_movies = false,
		 reuse_frames = false, allow_sparse = false, check = false, sanitize = false;
	std::optional<double> minimum, maximum;
	std::string reconstruction;
	std::set<std::string> explicit_options;
};
bool cgs(const Json &);
std::string field_label(const Json &, int);
std::vector<std::pair<fs::path, Json>> completed_runs(const fs::path &);
void gnuplot_script(const fs::path &, const std::string &, const Options &);
void plot_pair(const fs::path &, const std::string &, int, int, const Options &);
fs::path render(const fs::path &, const Options &);
void publish(const fs::path &, Json &);
fs::path report_pages(const fs::path &, bool = false);
void movie_index(const fs::path &);
Json make_movie(const fs::path &, const Options &);
Json make_session(const fs::path &, const Options &);
fs::path visit_executable(const Options &);
std::string visit_session_xml(const fs::path &, const Json &, const Options &);
void check_dependencies(const Options &, bool);
int run(Options);
int maintenance(const Options &);
Options arguments(int, char **);
} // namespace rr
