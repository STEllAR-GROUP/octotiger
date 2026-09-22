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
Json opacityMetadata(const Json &config);
inline constexpr double cCgs = 2.99792458e10, halfWidth = 3e10;
inline const std::array<std::string, 4> fields{"er", "fx", "fy", "fz"};
inline const std::array<std::string, 3> norms{"L1", "L2", "Linf"};
inline const std::array<std::string, 4> cases{"streaming_wave", "streaming_front", "gaussian_pulse",
											  "equilibrium_sphere"};
inline const Json cgsUnits{{"system", "CGS"},
							{"length", "cm"},
							{"time", "s"},
							{"mass", "g"},
							{"energy_density", "erg/cm^3"},
							{"flux", "erg/(cm^2 s)"},
							{"chi", "1/cm"},
							{"luminosity", "erg/s"},
							{"source", "erg/(cm^3 s)"}};
void require(bool, const std::string &);
std::string readText(const fs::path &);
void atomicText(const fs::path &, const std::string &);
Json readJson(const fs::path &);
void writeJson(const fs::path &, const Json &);
std::string sha256(const fs::path &);
std::string sha256Text(const std::string &);
std::string trim(std::string);
std::string lower(std::string);
std::string number(double);
double numeric(const std::string &);
bool close(double, double, double = 1e-11, double = 0);
std::string stamp();
std::string shellQuote(const std::string &);
std::string gpQuote(const std::string &);
std::string html(const std::string &);
std::string url(const std::string &);
std::string pageStart(const std::string &);
fs::path absolute(fs::path);
fs::path sourceRoot(const fs::path &);
fs::path toolsRoot(const fs::path &);
fs::path executable(const std::string &);
void installSignals();
void checkInterrupt();
int execute(const std::vector<std::string> &, const fs::path &, const fs::path & = {},
			const std::function<void(const std::string &)> & = {}, bool = true, bool = true);
void executeDetached(const std::vector<std::string> &, const fs::path &);
Json readConfig(const fs::path &);
void writeConfig(const fs::path &, const Json &);
void recordsCsv(const fs::path &, const Json &, std::vector<std::string> = {});
struct Table {
	std::vector<std::string> columns;
	std::vector<std::vector<double>> rows;
};
Table readCsv(const fs::path &, const std::vector<std::string> &);
Json readNorms(const fs::path &, const Json &);
Table readSlice(const fs::path &, const Json &);
struct Budget {
	Table history;
	Json summary = Json::array();
	std::vector<std::array<double, 4>> residual, scale, normalized;
};
std::optional<Budget> readConservation(const fs::path &, const Json &);
void verifyCgsSummary(const fs::path &);
Json cadence(double, int, double, std::optional<double> = {});
std::vector<std::size_t> frameSchedule(const std::vector<double> &, double, int, double);
std::vector<fs::path> numericalSilos(const fs::path &);
struct Options {
	std::string command = "live", gnuplot = "gnuplot", ffmpeg = "ffmpeg", field = "er", view = "slice",
				axis = "z", color = "hot", cxx = "g++", visit = "visit";
	fs::path root, build, exe, generator, output, resume, input, reference, sessionDir;
	std::vector<std::string> selectedCases;
	std::vector<int> levels;
	double time = 4, seconds = 20, hold = 1, position = 0, odt = 0, hard_dt = 0;
	int threads = 12, jobs = 8, snapshots = 61, fps = 30, width = 1280, height = 960, cells = 32;
	bool noBuild = false, noOpen = false, dryRun = false, noSilo = false, noMovies = false,
		 reuseFrames = false, allowSparse = false, check = false, sanitize = false;
	std::optional<double> minimum, maximum;
	std::string reconstruction;
	std::set<std::string> explicitOptions;
};
bool cgs(const Json &);
std::string fieldLabel(const Json &, int);
std::vector<std::pair<fs::path, Json>> completedRuns(const fs::path &);
void gnuplotScript(const fs::path &, const std::string &, const Options &);
void plotPair(const fs::path &, const std::string &, int, int, const Options &);
fs::path render(const fs::path &, const Options &);
void publish(const fs::path &, Json &);
fs::path reportPages(const fs::path &, bool = false);
void movieIndex(const fs::path &);
Json makeMovie(const fs::path &, const Options &);
Json makeSession(const fs::path &, const Options &);
fs::path visitExecutable(const Options &);
std::string visitSessionXml(const fs::path &, const Json &, const Options &);
void checkDependencies(const Options &, bool);
int run(Options);
int maintenance(const Options &);
Options arguments(int, char **);
} // namespace rr
