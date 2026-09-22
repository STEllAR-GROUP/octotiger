#include "common.hpp"
#include <limits>
namespace rr
{
Json readConfig(const fs::path &p)
{
	Json j = Json::object();
	std::istringstream in(readText(p));
	std::string s;
	while (std::getline(in, s)) {
		s = trim(s.substr(0, s.find('#')));
		if (s.empty())
			continue;
		auto eq = s.find('=');
		require(eq != s.npos, "Invalid INI line: " + s);
		auto k = trim(s.substr(0, eq));
		require(!k.empty() && !j.contains(k), "Duplicate/empty INI key: " + k);
		j[k] = trim(s.substr(eq + 1));
	}
	return j;
}
void writeConfig(const fs::path &p, const Json &j)
{
	std::string s;
	for (auto it = j.begin(); it != j.end(); ++it)
		s += it.key() + "=" + it.value().get<std::string>() + "\n";
	atomicText(p, s);
}
void recordsCsv(const fs::path &p, const Json &rows, std::vector<std::string> keys)
{
	if (keys.empty() && !rows.empty())
		for (auto it = rows[0].begin(); it != rows[0].end(); ++it)
			keys.push_back(it.key());
	auto quote = [](std::string s) {
		std::string o = "\"";
		for (char c : s)
			o += c == '"' ? "\"\"" : std::string(1, c);
		return o + '"';
	};
	std::ostringstream out;
	for (std::size_t i = 0; i < keys.size(); ++i)
		out << (i ? "," : "") << quote(keys[i]);
	out << '\n';
	for (auto &row : rows) {
		for (std::size_t i = 0; i < keys.size(); ++i) {
			if (i)
				out << ',';
			auto v = row.value(keys[i], Json());
			if (v.is_string())
				out << quote(v.get<std::string>());
			else if (v.is_number_float())
				out << std::scientific << std::setprecision(17) << v.get<double>();
			else if (!v.is_null())
				out << v.dump();
		}
		out << '\n';
	}
	atomicText(p, out.str());
}
Table readCsv(const fs::path &p, const std::vector<std::string> &columns)
{
	std::ifstream in(p);
	require(bool(in), "Cannot read " + p.string());
	std::string line;
	require(bool(std::getline(in, line)), "Empty CSV: " + p.string());
	if (!line.empty() && line.back() == '\r')
		line.pop_back();
	std::string expected;
	for (auto &c : columns) {
		if (!expected.empty())
			expected += ',';
		expected += c;
	}
	require(line == expected, "Invalid CSV columns: " + p.string());
	Table t{columns, {}};
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		std::vector<double> r;
		std::istringstream s(line);
		std::string value;
		while (std::getline(s, value, ','))
			r.push_back(numeric(trim(value)));
		require(r.size() == columns.size() && !line.ends_with(','), "Invalid CSV row: " + p.string());
		t.rows.push_back(std::move(r));
	}
	require(!in.bad() && !t.rows.empty(), "Empty or unreadable CSV: " + p.string());
	return t;
}
Json readNorms(const fs::path &folder, const Json &m)
{
	Json rows = Json::object();
	auto summary = readJson(folder / "runSummary.json");
	require(summary.value("schemaVersion", 0) == 1 && summary.value("status", "") == "completed",
		"Missing or incomplete structured run summary in " + folder.string());
	require(close(summary.at("finalTime"), m.at("time")), "Wrong final comparison time");
	const auto &analytic = summary.at("analytic");
	for (const auto &field : fields) {
		require(analytic.contains(field), "Missing analytic field: " + field);
		for (const auto &norm : norms) {
			double value = analytic.at(field).at(lower(norm));
			require(std::isfinite(value) && value >= 0, "Invalid analytic norm: " + field + " " + norm);
			rows[field][norm] = value;
		}
	}
	std::string line;
	for (auto &n : norms) {
		std::istringstream in(readText(folder / (n + ".dat")));
		std::vector<double> vals;
		int count = 0;
		while (std::getline(in, line)) {
			if (trim(line).empty())
				continue;
			++count;
			std::istringstream s(line);
			std::string v;
			while (s >> v)
				vals.push_back(numeric(v));
		}
		require(count == 1 && vals.size() >= 6, "Expected one comparison in " + n + ".dat");
		require(close(vals[0], m.at("dx"), 2e-6) && vals[1] == m.at("level").get<int>(),
				"Norm resolution mismatch");
		for (int f = 0; f < 4; ++f)
			require(close(vals[vals.size() - 4 + f], rows[fields[f]][n], 2e-6),
				"Structured summary and norm file disagree");
	}
	return rows;
}
Table readSlice(const fs::path &folder, const Json &m)
{
	std::vector<std::string> cols{"t",	"dx", "x",		"y",	  "z",		"er",	 "fx",
								  "fy", "fz", "er_ref", "fx_ref", "fy_ref", "fz_ref"};
	Table t{cols, {}};
	auto dir = folder / "radiation-slices";
	require(fs::is_directory(dir), "Missing slices: " + dir.string());
	int n = m.at("cells");
	double dx = m.at("dx"), len = m.at("length");
	require(n > 0 && dx > 0 && len > 0, "Invalid slice metadata");
	t.rows.resize(std::size_t(n) * n);
	std::size_t count = 0;
	std::regex pattern("slice-.*\\.csv");
	for (auto &e : fs::directory_iterator(dir))
		if (std::regex_match(e.path().filename().string(), pattern))
			for (auto &r : readCsv(e.path(), cols).rows) {
				require(close(r[0], m.at("time")) && close(r[1], dx) && close(r[4], dx / 2),
						"Slice time/spacing/z mismatch");
				double a = (r[2] + len / 2) / dx - .5, b = (r[3] + len / 2) / dx - .5;
				require(a >= -.5 && a < n - .5 && b >= -.5 && b < n - .5, "Slice outside domain");
				int i = std::lround(a), j = std::lround(b);
				require(close(r[2], -len / 2 + (i + .5) * dx, 0, 1e-11 * dx) &&
							close(r[3], -len / 2 + (j + .5) * dx, 0, 1e-11 * dx),
						"Misaligned slice coordinates");
				auto k = std::size_t(i) * n + j;
				require(t.rows[k].empty(), "Duplicate slice cell");
				t.rows[k] = r;
				++count;
			}
	require(count == std::size_t(n) * n, "Missing slice cells");
	return t;
}
std::optional<Budget> readConservation(const fs::path &folder, const Json &m)
{
	auto p = folder / "radiation-conservation.csv";
	if (!fs::exists(p))
		return {};
	std::vector<std::string> cols{"t", "volume"};
	for (auto &f : fields)
		cols.push_back(f);
	for (auto &f : fields)
		cols.push_back(f + "_boundary");
	for (auto &f : fields)
		cols.push_back(f + "_source");
	Budget b;
	b.history = readCsv(p, cols);
	auto &rows = b.history.rows;
	double len = m.at("length"), c = m.at("c"), vol = len * len * len;
	require(len > 0 && c > 0 && std::isfinite(vol) && std::isfinite(c), "Invalid conservation metadata");
	require(rows[0][0] == 0 && close(rows.back()[0], m.at("time")),
			"Conservation must span t=0 through final time");
	for (int j = 6; j < 14; ++j)
		require(rows[0][j] == 0, "Nonzero initial boundary/source ledger");
	b.residual.resize(rows.size());
	b.scale.resize(rows.size());
	b.normalized.resize(rows.size());
	// Fix the normalization at t=0: growth of a source or boundary budget must
	// not reduce the apparent fractional conservation error later in the run.
	const double initialEnergy = std::abs(rows[0][2]);
	const double initialFluxScale = c * initialEnergy;
	constexpr double zeroTolerance = 64 * std::numeric_limits<double>::epsilon();
	require(initialEnergy > 0 && std::isfinite(initialFluxScale) && initialFluxScale > 0,
			"Conservation fractions require a nonzero initial energy and finite c times that energy");
	std::array<double, 4> initialScale;
	std::array<std::string, 4> basis;
	initialScale[0] = initialEnergy;
	basis[0] = "abs(initial_energy)";
	for (int f = 1; f < 4; ++f) {
		const double initial = std::abs(rows[0][2 + f]);
		const bool zeroNetFlux = initial / initialFluxScale <= zeroTolerance;
		initialScale[f] = zeroNetFlux ? initialFluxScale : initial;
		basis[f] = zeroNetFlux ? "c * abs(initial_energy)" : "abs(initial_component)";
	}
	for (std::size_t i = 0; i < rows.size(); ++i) {
		auto &r = rows[i];
		require(close(r[1], vol), "Conservation domain-volume mismatch");
		require(!i || r[0] > rows[i - 1][0], "Conservation times must increase");
		for (int f = 0; f < 4; ++f) {
			double residual = r[2 + f] - rows[0][2 + f] + r[6 + f] - r[10 + f];
			double scale = initialScale[f];
			require(std::isfinite(residual) && std::isfinite(residual / scale),
					"Invalid derived conservation budget");
			b.residual[i][f] = residual;
			b.scale[i][f] = scale;
			b.normalized[i][f] = residual / scale;
		}
	}
	for (int f = 0; f < 4; ++f) {
		double maxr = 0, maxn = 0;
		for (std::size_t i = 0; i < rows.size(); ++i) {
			maxr = std::max(maxr, std::abs(b.residual[i][f]));
			maxn = std::max(maxn, std::abs(b.normalized[i][f]));
		}
		b.summary.push_back({{"field", fields[f]},
							 {"initial", rows[0][2 + f]},
							 {"final", rows.back()[2 + f]},
							 {"raw_change", rows.back()[2 + f] - rows[0][2 + f]},
							 {"boundary", rows.back()[6 + f]},
							 {"source", rows.back()[10 + f]},
							 {"residual", b.residual.back()[f]},
							 {"normalization_scale", b.scale.back()[f]},
							 {"normalization_basis", basis[f]},
							 {"zero_initial_flux_relative_tolerance", zeroTolerance},
							 {"initial_fraction", rows[0][2 + f] / initialScale[f]},
							 {"final_fraction", rows.back()[2 + f] / initialScale[f]},
							 {"boundary_fraction", rows.back()[6 + f] / initialScale[f]},
							 {"source_fraction", rows.back()[10 + f] / initialScale[f]},
							 {"normalized_residual", b.normalized.back()[f]},
							 {"normalized_error", std::abs(b.normalized.back()[f])},
							 {"max_abs_residual", maxr},
							 {"max_normalized_error", maxn},
							 {"integral_units", cgs(m) ? (f == 0 ? "erg" : "erg cm/s") : "code units"}});
	}
	return b;
}
void verifyCgsSummary(const fs::path &p)
{
	const auto summary = readJson(p);
	const auto &units = summary.at("units");
	for (const auto &field : {"grams", "centimeters", "seconds"})
		require(close(units.at(field), 1, 1e-12),
			"Solver is not using CGS unit factors; rebuild");
	require(close(units.at("lightSpeed"), cCgs, 5e-7),
		"Wrong solver light speed; rebuild for CGS");
}
Json cadence(double t, int n, double cfl, std::optional<double> odt)
{
	require(n >= 3 && t > 0 && cfl > 0 && std::isfinite(t) && std::isfinite(cfl), "Invalid capture cadence");
	double interval = odt.value_or(t / (n - 1));
	require(interval > 0 && interval < t && std::isfinite(interval), "Invalid snapshot interval");
	double steps = std::max(1., std::floor(2 / cfl + .5));
	require(steps < 1e9, "CFL too small");
	return {{"odt", interval},
			{"hard_dt", interval / steps},
			{"steps_per_output_check", int(steps)},
			{"requested_snapshots", int(std::ceil(t / interval - 1e-10)) + 1}};
}
std::vector<std::size_t> frameSchedule(const std::vector<double> &t, double seconds, int fps, double hold)
{
	require(t.size() >= 2 && std::isfinite(seconds) && seconds > 0 && fps > 0 && std::isfinite(hold) &&
				hold >= 0,
			"Invalid playback parameters");
	for (std::size_t i = 0; i < t.size(); ++i)
		require(std::isfinite(t[i]) && (!i || t[i] > t[i - 1]), "Invalid snapshot times");
	require(seconds * fps < 1e7 && hold * fps < 1e7, "Too many video frames");
	auto total = std::llround(seconds * fps), pause = std::llround(hold * fps), active = total - 2 * pause;
	require(active >= 2, "Movie duration must exceed endpoint holds");
	std::vector<std::size_t> seq(pause, 0);
	for (long long i = 0; i < active; ++i) {
		double value = t.front() + (t.back() - t.front()) * i / (active - 1);
		auto it = std::upper_bound(t.begin(), t.end(), value);
		seq.push_back(it == t.begin() ? 0 : std::size_t(it - t.begin() - 1));
	}
	seq.back() = t.size() - 1;
	seq.insert(seq.end(), pause, t.size() - 1);
	return seq;
}
std::vector<fs::path> numericalSilos(const fs::path &dir)
{
	std::map<unsigned long long, fs::path> sorted;
	std::regex re(R"(X\.(\d+)\.silo)");
	std::smatch m;
	for (auto &e : fs::directory_iterator(dir)) {
		auto name = e.path().filename().string();
		if (std::regex_match(name, m, re)) {
			auto i = std::stoull(m[1]);
			require(!sorted.contains(i), "Duplicate snapshot number");
			sorted[i] = rr::absolute(e.path());
		}
	}
	require(!sorted.empty() && fs::is_regular_file(dir / "final.silo"),
			"Need X.*.silo and final.silo snapshots");
	std::vector<fs::path> v;
	for (auto &[i, p] : sorted)
		v.push_back(p);
	v.push_back(rr::absolute(dir / "final.silo"));
	for (auto &p : v)
		require(fs::file_size(p) > 0, "Empty Silo file");
	return v;
}
std::vector<std::pair<fs::path, Json>> completedRuns(const fs::path &dir)
{
	std::vector<std::pair<fs::path, Json>> out;
	auto add = [&](const fs::path &p) {
		if (!fs::is_regular_file(p / "run.json"))
			return;
		auto m = readJson(p / "run.json");
		if (m.value("status", "") != "complete")
			return;
		auto name = m.at("case").get<std::string>();
		require(std::find(cases.begin(), cases.end(), name) != cases.end(), "Unknown case: " + name);
		out.emplace_back(p, m);
	};
	if (fs::is_regular_file(dir / "run.json"))
		add(dir);
	else
		for (auto &c : cases)
			if (fs::is_directory(dir / c))
				for (auto &e : fs::directory_iterator(dir / c))
					if (e.is_directory() && e.path().filename().string().starts_with('l'))
						add(e.path());
	std::sort(out.begin(), out.end(), [](auto &a, auto &b) {
		return std::make_pair(a.second.at("cells").template get<int>(),
							  a.second.at("case").template get<std::string>()) <
			   std::make_pair(b.second.at("cells").template get<int>(),
							  b.second.at("case").template get<std::string>());
	});
	return out;
}
} // namespace rr
