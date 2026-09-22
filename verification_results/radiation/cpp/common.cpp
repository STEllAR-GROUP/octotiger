#include "common.hpp"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <openssl/evp.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>
namespace rr
{
namespace
{
volatile sig_atomic_t interrupted = 0;
void signalHandler(int s)
{
	interrupted = s;
}
} // namespace
void require(bool ok, const std::string &s)
{
	if (!ok)
		throw std::runtime_error(s);
}
void installSignals()
{
	struct sigaction a {
	};
	a.sa_handler = signalHandler;
	sigemptyset(&a.sa_mask);
	sigaction(SIGINT, &a, nullptr);
	sigaction(SIGTERM, &a, nullptr);
}
void checkInterrupt()
{
	require(!interrupted, "Interrupted; completed results are retained");
}
std::string readText(const fs::path &p)
{
	std::ifstream f(p, std::ios::binary);
	require(bool(f), "Cannot read " + p.string());
	std::ostringstream s;
	s << f.rdbuf();
	require(!f.bad(), "Read failed: " + p.string());
	return s.str();
}
void atomicText(const fs::path &p, const std::string &s)
{
	fs::create_directories(p.parent_path());
	auto tmp = p;
	tmp += ".tmp-" + std::to_string(getpid());
	try {
		std::ofstream f(tmp, std::ios::binary);
		f.exceptions(std::ios::failbit | std::ios::badbit);
		f << s;
		f.close();
		fs::rename(tmp, p);
	} catch (...) {
		std::error_code e;
		fs::remove(tmp, e);
		throw;
	}
}
Json readJson(const fs::path &p)
{
	return Json::parse(readText(p));
}
void writeJson(const fs::path &p, const Json &j)
{
	atomicText(p, j.dump(2) + "\n");
}
std::string sha256(const fs::path &p)
{
	std::ifstream f(p, std::ios::binary);
	require(bool(f), "Cannot hash " + p.string());
	std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
	require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 initialization failed");
	std::array<char, 65536> b;
	while (f) {
		f.read(b.data(), b.size());
		require(EVP_DigestUpdate(ctx.get(), b.data(), f.gcount()) == 1, "SHA256 update failed");
	}
	require(f.eof(), "Read failed hashing " + p.string());
	unsigned char hash[EVP_MAX_MD_SIZE];
	unsigned int n;
	require(EVP_DigestFinal_ex(ctx.get(), hash, &n) == 1, "SHA256 finalization failed");
	std::ostringstream s;
	s << std::hex << std::setfill('0');
	for (unsigned i = 0; i < n; ++i)
		s << std::setw(2) << unsigned(hash[i]);
	return s.str();
}
std::string sha256Text(const std::string &text)
{
	unsigned char hash[EVP_MAX_MD_SIZE];
	unsigned int n = 0;
	require(EVP_Digest(text.data(), text.size(), hash, &n, EVP_sha256(), nullptr) == 1,
			"SHA256 text digest failed");
	std::ostringstream out;
	out << std::hex << std::setfill('0');
	for (unsigned i = 0; i < n; ++i)
		out << std::setw(2) << unsigned(hash[i]);
	return out.str();
}
std::string trim(std::string s)
{
	auto b = s.find_first_not_of(" \t\r\n");
	return b == s.npos ? "" : s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}
std::string lower(std::string s)
{
	for (auto &c : s)
		c = std::tolower(static_cast<unsigned char>(c));
	return s;
}
std::string number(double d)
{
	require(std::isfinite(d), "Nonfinite number");
	std::ostringstream s;
	s << std::setprecision(17) << d;
	return s.str();
}
double numeric(const std::string &s)
{
	std::size_t n;
	double d = std::stod(s, &n);
	require(n == s.size() && std::isfinite(d), "Invalid finite number: " + s);
	return d;
}
bool close(double a, double b, double r, double at)
{
	return std::isfinite(a) && std::isfinite(b) &&
		   std::abs(a - b) <= std::max(at, r * std::max(std::abs(a), std::abs(b)));
}
std::string stamp()
{
	auto t = std::chrono::system_clock::now();
	auto sec = std::chrono::system_clock::to_time_t(t);
	std::tm tm{};
	gmtime_r(&sec, &tm);
	std::ostringstream s;
	s << std::put_time(&tm, "%Y%m%d-%H%M%S") << '-'
	  << std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count() % 1000000;
	return s.str();
}
std::string shellQuote(const std::string &s)
{
	require(s.find('\0') == s.npos, "NUL in command argument");
	std::string o = "'";
	for (char c : s)
		o += c == '\'' ? "'\\''" : std::string(1, c);
	return o + "'";
}
std::string gpQuote(const std::string &s)
{
	std::string o = "\"";
	for (char c : s) {
		require(c != '\n' && c != '\r' && c != '\0', "Control character in gnuplot string");
		if (c == '\\' || c == '\"' || c == '`')
			o += '\\';
		o += c;
	}
	return o + '"';
}
std::string html(const std::string &s)
{
	std::string o;
	for (char c : s)
		switch (c) {
		case '&':
			o += "&amp;";
			break;
		case '<':
			o += "&lt;";
			break;
		case '>':
			o += "&gt;";
			break;
		case '"':
			o += "&quot;";
			break;
		case '\'':
			o += "&#39;";
			break;
		default:
			o += c;
		}
	return o;
}
std::string url(const std::string &s)
{
	std::ostringstream o;
	o << std::hex << std::uppercase;
	for (unsigned char c : s)
		if (std::isalnum(c) || c == '/' || c == '-' || c == '_' || c == '.' || c == '~')
			o << c;
		else
			o << '%' << std::setw(2) << std::setfill('0') << unsigned(c);
	return o.str();
}
std::string pageStart(const std::string &title)
{
	return "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
		   "content=\"width=device-width,initial-scale=1\"><title>" +
		   html(title) +
		   "</title><style>body{font:17px/1.5 system-ui;max-width:1500px;margin:32px auto;padding:0 "
		   "24px;color:#172335;background:#fafbfc}img,video{max-width:100%;height:auto}table{border-collapse:"
		   "collapse;width:100%}td,th{padding:8px;border-bottom:1px solid "
		   "#bbc;text-align:left}section,details{margin:24px "
		   "0;padding:12px;background:white}a{color:#0755a2}.scroll{overflow:auto}.error{color:#981818;white-"
		   "space:pre-wrap}</style></head><body><h1>" +
		   html(title) + "</h1>";
}
fs::path absolute(fs::path p)
{
	auto s = p.string();
	if (s == "~" || s.starts_with("~/")) {
		const char *h = getenv("HOME");
		require(h, "HOME is unset");
		p = fs::path(h) / s.substr(s.size() == 1 ? 1 : 2);
	}
	return fs::absolute(p).lexically_normal();
}
fs::path sourceRoot(const fs::path &project)
{
	auto root = rr::absolute(project);
	auto nested = root / "src/octotiger";
	if (fs::is_regular_file(nested / "CMakeLists.txt"))
		return nested;
	// Retain compatibility with batches made before the checkout was placed under src/octotiger.
	if (fs::is_regular_file(root / "CMakeLists.txt") && fs::is_directory(root / "src"))
		return root;
	return nested;
}
fs::path toolsRoot(const fs::path &project)
{
	return sourceRoot(project) / "verification_results/radiation";
}
fs::path executable(const std::string &s)
{
	if (s.find('/') != s.npos) {
		auto p = rr::absolute(s);
		require(fs::is_regular_file(p) && access(p.c_str(), X_OK) == 0,
				"Executable not found: " + p.string());
		return p;
	}
	std::istringstream dirs(getenv("PATH") ? getenv("PATH") : "");
	std::string dir;
	while (std::getline(dirs, dir, ':')) {
		auto p = rr::absolute(fs::path(dir) / s);
		if (fs::is_regular_file(p) && access(p.c_str(), X_OK) == 0)
			return p;
	}
	throw std::runtime_error("Executable not found: " + s);
}
int execute(const std::vector<std::string> &args, const fs::path &cwd, const fs::path &log,
			const std::function<void(const std::string &)> &feed, bool checked, bool echoOutput)
{
	checkInterrupt();
	require(!args.empty(), "Empty command");
	auto program = executable(args[0]);
	std::cout << '+';
	for (auto &a : args)
		std::cout << ' ' << shellQuote(a);
	std::cout << std::endl;
	std::ofstream out;
	if (!log.empty()) {
		out.open(log);
		require(bool(out), "Cannot write " + log.string());
	}
	std::vector<char *> argv;
	for (auto &a : args)
		argv.push_back(const_cast<char *>(a.c_str()));
	argv.push_back(nullptr);
	int fd[2];
	require(pipe(fd) == 0, "pipe failed");
	pid_t pid = fork();
	if (pid < 0) {
		::close(fd[0]);
		::close(fd[1]);
		throw std::runtime_error("fork failed");
	}
	if (pid == 0) {
		setpgid(0, 0);
		::close(fd[0]);
		// Children have their own process group so cancellation reaches descendants.
		// They must not read or configure the foreground terminal: that can stop
		// FFmpeg with SIGTTIN/SIGTTOU. These are noninteractive commands.
		int input = open("/dev/null", O_RDONLY);
		if (input < 0 || dup2(input, STDIN_FILENO) < 0)
			_exit(126);
		if (input != STDIN_FILENO)
			::close(input);
		if (chdir(cwd.c_str()) != 0)
			_exit(126);
		dup2(fd[1], 1);
		dup2(fd[1], 2);
		::close(fd[1]);
		execv(program.c_str(), argv.data());
		_exit(127);
	}
	setpgid(pid, pid);
	::close(fd[1]);
	fcntl(fd[0], F_SETFL, O_NONBLOCK);
	int status = 0;
	bool done = false, eof = false;
	std::string pending;
	auto emit = [&](const std::string &line) {
		if (echoOutput)
			std::cout << line << std::flush;
		if (out.is_open()) {
			out << line;
			out.flush();
			require(bool(out), "Log write failed");
		}
		if (feed)
			feed(line);
	};
	try {
		while (!done || !eof) {
			checkInterrupt();
			pollfd pfd{fd[0], POLLIN | POLLHUP, 0};
			poll(&pfd, 1, 100);
			char b[8192];
			ssize_t n;
			while ((n = read(fd[0], b, sizeof b)) > 0) {
				pending.append(b, n);
				std::size_t pos;
				while ((pos = pending.find('\n')) != pending.npos) {
					emit(pending.substr(0, pos + 1));
					pending.erase(0, pos + 1);
				}
			}
			if (n == 0)
				eof = true;
			if (!done) {
				auto w = waitpid(pid, &status, WNOHANG);
				require(w >= 0 || errno == EINTR, "waitpid failed");
				done = w == pid;
			}
		}
		if (!pending.empty())
			emit(pending);
		::close(fd[0]);
	} catch (...) {
		kill(-pid, SIGTERM);
		for (int i = 0; i < 20; ++i) {
			if (waitpid(pid, &status, WNOHANG) == pid) {
				done = true;
				break;
			}
			poll(nullptr, 0, 100);
		}
		kill(-pid, SIGKILL);
		if (!done)
			waitpid(pid, &status, 0);
		::close(fd[0]);
		throw;
	}
	int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
	require(!checked || code == 0, "Exit code " + std::to_string(code) + "; see " + log.string());
	return code;
}
void executeDetached(const std::vector<std::string> &args, const fs::path &cwd)
{
	require(!args.empty(), "Empty command");
	auto program = executable(args[0]);
	std::cout << '+';
	for (auto &a : args)
		std::cout << ' ' << shellQuote(a);
	std::cout << " &" << std::endl;
	std::vector<char *> argv;
	for (auto &a : args)
		argv.push_back(const_cast<char *>(a.c_str()));
	argv.push_back(nullptr);
	pid_t pid = fork();
	require(pid >= 0, "fork failed");
	if (pid == 0) {
		if (setsid() < 0)
			_exit(126);
		pid_t child = fork();
		if (child < 0)
			_exit(126);
		if (child > 0)
			_exit(0);
		int null = open("/dev/null", O_RDWR);
		if (null < 0 || dup2(null, STDIN_FILENO) < 0 || dup2(null, STDOUT_FILENO) < 0 ||
				dup2(null, STDERR_FILENO) < 0)
			_exit(126);
		if (null > STDERR_FILENO)
			::close(null);
		if (chdir(cwd.c_str()) != 0)
			_exit(126);
		execv(program.c_str(), argv.data());
		_exit(127);
	}
	int status = 0;
	pid_t waited;
	do {
		waited = waitpid(pid, &status, 0);
	} while (waited < 0 && errno == EINTR);
	require(waited == pid, "waitpid failed");
	require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "Detached launch failed");
}
bool cgs(const Json &m)
{
	return m.value("units", Json::object()).value("system", "") == "CGS";
}
std::string fieldLabel(const Json &m, int f)
{
	return std::array<std::string, 4>{"E", "Fx", "Fy", "Fz"}[f] +
		   (cgs(m) ? (f == 0 ? " (erg/cm^3)" : " (erg/(cm^2 s))") : " (code units)");
}
// Record configured material settings separately from prescribed-medium test chi.
// Values are strings, like run.ini/config; the source/executable hashes identify defaults.
Json opacityMetadata(const Json &config)
{
    Json result{{"schema_version", 1}, {"material_model", "legacy"},
        {"units", "cm2/g"}, {"absorption", "0"}, {"scattering", "0"},
        {"transport_absorption", "-1"}, {"legacy_constant", "-1"},
        {"legacy_constant_units", "code area/mass"}};
    for (auto key : {"units", "absorption", "scattering", "transport_absorption"})
        if (config.contains(std::string("radiation.opacity.") + key))
            result[key] = config.at(std::string("radiation.opacity.") + key);
    if (config.contains("radiation.opacity.model")) result["material_model"] = config.at("radiation.opacity.model");
    for (auto key : {"rad_opacity", "radiation.opacity.constant"})
        if (config.contains(key)) result["legacy_constant"] = config.at(key);
    for (auto key : {"rad_test_chi", "radiation.test.extinction"})
        if (config.contains(key)) result["prescribed_test_chi_code_inverse_length"] = config.at(key);
    result["scope"] = "Material settings; prescribed regression sources retain their independent test chi";
    return result;
}
} // namespace rr
