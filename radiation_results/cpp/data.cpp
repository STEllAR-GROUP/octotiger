#include <stddef.h>
#include "common.hpp"
#include <limits>
namespace rr
{
Json read_config(const fs::path &p)
{
	Json j = Json::object();
	std::istringstream in(read_text(p));
	std::string s;
	while (std::getline(in, s)) {
		s = trim(s.substr(0, s.find('#')));
		if (s.empty()) {
      continue;
    }
    auto eq = s.find('=');
		require(eq != s.npos, "Invalid INI line: " + s);
		auto k = trim(s.substr(0, eq));
		require(!k.empty() && !j.contains(k), "Duplicate/empty INI key: " + k);
		j[k] = trim(s.substr(eq + 1));
	}
	return j;
}
void write_config(const fs::path &p, const Json &j)
{
	std::string s;
	for (auto it = j.begin(); it != j.end(); ++it) {
    s += it.key() + "=" + it.value().get<std::string>() + "\n";
  }
  atomic_text(p, s);
}
void records_csv(const fs::path &p, const Json &rows, std::vector<std::string> keys)
{
	if (keys.empty() && !rows.empty()) {
    for (auto it = rows[0].begin(); it != rows[0].end(); ++it) {
      keys.push_back(it.key());
    }
  }
  auto quote = [](std::string s) {
		std::string o = "\"";
		for (char c : s) {
      o += c == '"' ? "\"\"" : std::string(1, c);
    }
    return o + '"';
	};
	std::ostringstream out;
	for (size_t i = 0; i < keys.size(); ++i) {
    out << (i ? "," : "") << quote(keys[i]);
  }
  out << '\n';
	for (auto &row : rows) {
		for (size_t i = 0; i < keys.size(); ++i) {
			if (i) {
        out << ',';
      }
      auto v = row.value(keys[i], Json());
			if (v.is_string()) {
        out << quote(v.get<std::string>());
      } else if (v.is_number_float()) {
        out << std::scientific << std::setprecision(17) << v.get<double>();
      } else if (!v.is_null()) {
        out << v.dump();
      }
    }
		out << '\n';
	}
	atomic_text(p, out.str());
}
Table read_csv(const fs::path &p, const std::vector<std::string> &columns)
{
	std::ifstream in(p);
	require(bool(in), "Cannot read " + p.string());
	std::string line;
	require(bool(std::getline(in, line)), "Empty CSV: " + p.string());
	if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  std::string expected;
	for (auto &c : columns) {
		if (!expected.empty()) {
      expected += ',';
    }
    expected += c;
	}
	require(line == expected, "Invalid CSV columns: " + p.string());
	Table t{columns, {}};
	while (std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    std::vector<double> r;
		std::istringstream s(line);
		std::string value;
		while (std::getline(s, value, ',')) {
      r.push_back(numeric(trim(value)));
    }
    require(r.size() == columns.size() && !line.ends_with(','), "Invalid CSV row: " + p.string());
		t.rows.push_back(std::move(r));
	}
	require(!in.bad() && !t.rows.empty(), "Empty or unreadable CSV: " + p.string());
	return t;
}
Json read_norms(const fs::path &folder, const Json &m)
{
	require(!fld_model(m), "FLD has no M1 analytic error norms");
	Json rows = Json::object();
	std::istringstream log(read_text(folder / "run.log"));
	std::string line;
	int finished = 0;
	std::regex marker(R"(^RADIATION_TEST_FINISHED\s+(\w+)\s+t=(\S+)\s*$)"),
		norm(R"(^\s*(er|fx|fy|fz|egas)\s+(\S+)\s+(\S+)\s+(\S+)\s*$)");
	std::smatch match;
	while (std::getline(log, line)) {
		if (std::regex_match(line, match, marker)) {
			++finished;
			require(lower(match[1]) == (m.at("case") == "marshak" ? "marshak" : "radiation_" + m.at("case").get<std::string>()) &&
						close(numeric(match[2]), m.at("time")),
					"Wrong comparison marker/time");
		}
		if (std::regex_match(line, match, norm)) {
			auto f = match[1].str();
            if (f == "egas") {
                if (m.at("case") != "marshak") { continue; }
                f = "material_egas";
            }
			require(!rows.contains(f), "Duplicate norm: " + f);
			for (int i = 0; i < 3; ++i) {
				double x = numeric(match[i + 2]);
				require(x >= 0, "Negative norm");
				rows[f][norms[i]] = x;
			}
		}
	}
	require(finished == 1 && rows.size() == (m.at("case") == "marshak" ? 5 : 4),
			"Missing/duplicate final comparison or norms in " + folder.string());
	for (auto &n : norms) {
		std::istringstream in(read_text(folder / (n + ".dat")));
		std::vector<double> vals;
		int count = 0;
		while (std::getline(in, line)) {
			if (trim(line).empty()) {
        continue;
      }
      ++count;
			std::istringstream s(line);
			std::string v;
			while (s >> v) {
        vals.push_back(numeric(v));
      }
    }
		require(count == 1 && vals.size() >= 6, "Expected one comparison in " + n + ".dat");
		require(close(vals[0], m.at("dx"), 2e-6) && vals[1] == m.at("level").get<int>(),
				"Norm resolution mismatch");
		for (int f = 0; f < 4; ++f) {
      require(close(vals[vals.size() - 4 + f], rows[fields[f]][n], 2e-6), "Log and norm file disagree");
    }
    if (m.at("case") == "marshak") {
        // egas is conservative hydro field1, after dx and level in the norm file.
        require(vals.size() >= 8 && close(vals[3], rows["material_egas"][n], 2e-6),
            "Marshak material-energy norm file and log disagree");
    }
  }
	return rows;
}
Json read_fld_diagnostics(const fs::path &folder, const Json &m)
{
    require(fld_model(m) && fld_case(m.at("case")), "Unsupported FLD result model/problem");
    require(m.value("radiation_flux_semantics", "") == "derived_fld_flux" &&
        m.value("reference_kind", "") == "linear-diffusion-limit" &&
        m.contains("reference_exact_for_solver") && m.at("reference_exact_for_solver") == false &&
        m.value("reference_model", "") == (m.at("case") == "marshak" ?
            "su-olson-linear-halfspace" : "periodic-constant-diffusion"), "Invalid FLD reference semantics");
    const auto cfg = m.value("config", Json::object());
    const double tolerance = numeric(cfg.value("fld_nonlinear_tolerance", "1e-10"));
    require(tolerance > 0, "Invalid FLD nonlinear tolerance");
    std::istringstream input(read_text(folder / "run.log"));
    std::string line;
    std::smatch match;
    const std::regex marker(R"(^RADIATION_FLD_FINISHED\s+(\w+)\s+t=(\S+)\s*$)");
    const std::regex accepted(R"(^FLD accepted t=(\S+) dt=(\S+) substeps=(\d+) retries=(\d+) nonlinear_iterations=(\d+) linear_iterations=(\d+) vcycles=(\d+) nonlinear_residual=(\S+) linear_residual=(\S+) boundary_energy=(\S+) conservation_error=(\S+)\s*$)");
    const double start = m.contains("checkpoint_resume") ?
        m.at("checkpoint_resume").at("checkpoint_time").get<double>() : 0.;
    double end = start, maximum_residual = 0, maximum_conservation_error = 0;
    size_t finished = 0, substeps = 0, retries = 0;
    Json steps = Json::array();
    while (std::getline(input, line)) {
        if (std::regex_match(line, match, marker)) {
            ++finished;
            const std::string expected = m.at("case") == "marshak" ? "marshak" :
                "radiation_" + m.at("case").get<std::string>();
            require(lower(match[1]) == expected && close(numeric(match[2]), m.at("time")),
                "Wrong FLD completion marker/time");
        } else if (std::regex_match(line, match, accepted)) {
            require(finished == 0, "FLD step after completion marker");
            const double time = numeric(match[1]), dt = numeric(match[2]);
            const double nonlinear = numeric(match[8]), linear = numeric(match[9]);
            const double boundary = numeric(match[10]), conservation = numeric(match[11]);
            const size_t count = std::stoull(match[3]), retry = std::stoull(match[4]);
            require(dt > 0 && close(time, end, 1.e-10) && count > 0 && retry <= 16 &&
                nonlinear >= 0 && nonlinear <= tolerance * (1 + 1.e-6) && linear >= 0,
                "Invalid or nonconverged FLD accepted step");
            end = time + dt;
            require(std::isfinite(end) && end > time, "Invalid FLD step endpoint");
            substeps += count;
            retries += retry;
            maximum_residual = std::max(maximum_residual, nonlinear);
            maximum_conservation_error = std::max(maximum_conservation_error, std::abs(conservation));
            steps.push_back({{"t", time}, {"dt", dt}, {"substeps", count}, {"retries", retry},
                {"nonlinear_iterations", std::stoull(match[5])}, {"linear_iterations", std::stoull(match[6])},
                {"vcycles", std::stoull(match[7])}, {"nonlinear_residual", nonlinear},
                {"linear_residual", linear}, {"boundary_energy", boundary}, {"conservation_error", conservation}});
        } else {
            require(!line.starts_with("FLD accepted") && !line.starts_with("RADIATION_FLD_FINISHED"),
                "Malformed FLD diagnostic record");
        }
    }
    require(finished == 1 && !steps.empty() && close(end, m.at("time")),
        "Missing, duplicate, or incomplete FLD final diagnostics");
    const auto budget = read_conservation(folder, m);
    require(bool(budget), "Missing FLD radiation energy conservation history");
    const auto slice = read_slice(folder, m);
    const auto material = m.at("case") == "marshak" ?
        std::optional<Table>(read_marshak_material(folder, m)) : std::nullopt;
    if (material) {
        for (size_t i = 0; i < slice.rows.size(); ++i) {
            for (size_t k = 0; k < 5; ++k) {
                require(close(slice.rows[i][k], material->rows[i][k]),
                    "FLD material/radiation reference slice geometry differs");
            }
        }
    }
    Json differences;
    for (int f = 0; f < (material ? 5 : 4); ++f) {
        double l1 = 0, l2 = 0, maximum = 0;
        for (size_t i = 0; i < slice.rows.size(); ++i) {
            const double difference = f == 4 ? material->rows[i][5] - material->rows[i][6] :
                slice.rows[i][5 + f] - slice.rows[i][9 + f];
            // Coarse leaves are repeated over their finest display footprint.
            // Equal pixel-area weights therefore recover the native leaf areas.
            const double weight = 1. / slice.rows.size();
            l1 += weight * std::abs(difference);
            l2 = std::hypot(l2, std::sqrt(weight) * difference);
            maximum = std::max(maximum, std::abs(difference));
        }
        require(std::isfinite(l1) && std::isfinite(l2) && std::isfinite(maximum),
            "Nonfinite FLD reference difference");
        differences[f == 4 ? "material_egas" : fields[f]] = {{"L1", l1}, {"L2", l2}, {"Linf", maximum}};
    }
    return {{"accepted_steps", steps.size()}, {"substeps", substeps}, {"retries", retries},
        {"history_start_time", start}, {"history_end_time", end},
        {"maximum_nonlinear_residual", maximum_residual},
        {"maximum_step_coupled_conservation_error", maximum_conservation_error},
        {"radiation_energy_budget", budget->summary.at(0)}, {"steps", steps},
        {"reference_kind", "linear-diffusion-limit"}, {"reference_exact_for_solver", false},
        {"reference_norm_scope", "projected-positive-side-slice"}, {"slice_reference_norms", differences}};
}
namespace {
Table adaptive_slice(const fs::path &directory, const Json &m, const std::vector<std::string> &columns)
{
    const int n = m.at("cells"), max_level = m.at("level"), min_level = m.at("min_level");
    const double dx = m.at("dx"), length = m.at("length");
    require(n > 0 && dx > 0 && close(n * dx, length) && min_level >= 0 && min_level <= max_level,
        "Invalid adaptive slice mesh metadata");
    require(fs::is_directory(directory), "Missing adaptive slices: " + directory.string());
    Table data{columns, std::vector<std::vector<double>>(size_t(n) * n)};
    size_t covered = 0;
    for (const auto &entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() != ".csv") {
            continue;
        }
        for (const auto &row : read_csv(entry.path(), columns).rows) {
            const double ratio = row[1] / dx;
            require(ratio >= 1 && ratio <= std::ldexp(1., max_level - min_level), "Adaptive leaf spacing outside configured levels");
            const int width = std::lround(ratio);
            require(close(ratio, width) && (width & (width - 1)) == 0, "Adaptive spacing is not dyadic");
            require(close(row[0], m.at("time")) && close(row[4], row[1] / 2), "Adaptive slice time or positive-side layer mismatch");
            const double a = (row[2] - row[1] / 2 + length / 2) / dx;
            const double b = (row[3] - row[1] / 2 + length / 2) / dx;
            const int i = std::lround(a), j = std::lround(b);
            require(close(a, i, 0, 1.e-8) && close(b, j, 0, 1.e-8) && i % width == 0 && j % width == 0 &&
                i >= 0 && j >= 0 && i + width <= n && j + width <= n, "Adaptive cell footprint is misaligned or outside the domain");
            for (int ii = i; ii < i + width; ++ii) {
                for (int jj = j; jj < j + width; ++jj) {
                    auto &pixel = data.rows[size_t(ii) * n + jj];
                    require(pixel.empty(), "Overlapping adaptive slice cells");
                    pixel = row;
                    // Constant projection of actual cell averages for display only.
                    // dx and z retain the source leaf values; no new analytic samples.
                    pixel[2] = -length / 2 + (ii + .5) * dx;
                    pixel[3] = -length / 2 + (jj + .5) * dx;
                    ++covered;
                }
            }
        }
    }
    require(covered == size_t(n) * n, "Missing adaptive slice coverage");
    return data;
}
}
Table read_slice(const fs::path &folder, const Json &m)
{
	std::vector<std::string> cols{"t",	"dx", "x",		"y",	  "z",		"er",	 "fx",
								  "fy", "fz", "er_ref", "fx_ref", "fy_ref", "fz_ref"};
    auto validate = [&](const Table &data) {
        if (fld_model(m)) {
            for (const auto &row : data.rows) {
                require(row[5] >= 0 && std::hypot(row[6], row[7], row[8]) <=
                    m.at("c").get<double>() * row[5] * (1 + 1.e-10),
                    "Nonphysical FLD energy or derived flux");
            }
        }
        return data;
    };
    if (m.value("adaptive", false)) {
        return validate(adaptive_slice(folder / "radiation-slices", m, cols));
    }
	Table t{cols, {}};
	auto dir = folder / "radiation-slices";
	require(fs::is_directory(dir), "Missing slices: " + dir.string());
	int n = m.at("cells");
	double dx = m.at("dx"), len = m.at("length");
	require(n > 0 && dx > 0 && len > 0, "Invalid slice metadata");
	t.rows.resize(size_t(n) * n);
	size_t count = 0;
	std::regex pattern("slice-.*\\.csv");
	for (auto &e : fs::directory_iterator(dir)) {
    if (std::regex_match(e.path().filename().string(), pattern)) {
      for (auto &r : read_csv(e.path(), cols).rows) {
				require(close(r[0], m.at("time")) && close(r[1], dx) && close(r[4], dx / 2),
						"Slice time/spacing/z mismatch");
				double a = (r[2] + len / 2) / dx - .5, b = (r[3] + len / 2) / dx - .5;
				require(a >= -.5 && a < n - .5 && b >= -.5 && b < n - .5, "Slice outside domain");
				int i = std::lround(a), j = std::lround(b);
				require(close(r[2], -len / 2 + (i + .5) * dx, 0, 1e-11 * dx) &&
							close(r[3], -len / 2 + (j + .5) * dx, 0, 1e-11 * dx),
						"Misaligned slice coordinates");
				auto k = size_t(i) * n + j;
				require(t.rows[k].empty(), "Duplicate slice cell");
				t.rows[k] = r;
				++count;
			}
    }
  }
  require(count == size_t(n) * n, "Missing slice cells");
	return validate(t);
}
Table read_marshak_material(const fs::path &folder, const Json &m)
{
    auto mesh = m;
    if (!mesh.contains("min_level")) { mesh["min_level"] = mesh.at("level"); }
    std::vector<std::string> columns{"t", "dx", "x", "y", "z", "egas"};
    columns.push_back("egas_ref");
    auto data = adaptive_slice(folder / "marshak-material-slices", mesh, columns);
    for (const auto &row : data.rows) {
        require(row[5] >= 0, "Negative Marshak material energy");
    }
    return data;
}
Table read_hydro_profile(const fs::path &folder, const Json &m)
{
    const std::vector<std::string> columns{"t", "dx", "x", "y", "z", "rho", "vx", "vy", "vz",
        "gas_temperature", "radiation_temperature", "er", "fx", "fy", "fz"};
    if (m.value("adaptive", false)) {
        auto result = adaptive_slice(folder / "radiation-hydro-slices", m, columns);
        for (const auto &r : result.rows) {
            require(r[5] > 0 && r[9] > 0 && r[10] >= 0 && r[11] >= 0 &&
                std::hypot(r[12], r[13], r[14]) <= m.at("c").get<double>() * r[11] * (1 + 1.e-10),
                "Nonphysical adaptive hydro slice state");
        }
        return result;
    }
    const int n = m.at("cells");
    const double dx = m.at("dx"), length = m.at("length");
    require(n > 0 && dx > 0 && length > 0, "Invalid hydro profile dimensions");
    Table data{columns, std::vector<std::vector<double>>(size_t(n) * n)};
    const auto dir = folder / "radiation-hydro-slices";
    require(fs::is_directory(dir), "Missing hydro slices: " + dir.string());
    size_t count = 0;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".csv") {
            continue;
        }
        for (const auto &r : read_csv(entry.path(), columns).rows) {
            require(close(r[0], m.at("time")) && close(r[1], dx) && close(r[4], dx / 2),
                "Hydro slice time, spacing or plane mismatch");
            const double a = (r[2] + length / 2) / dx - .5;
            const double b = (r[3] + length / 2) / dx - .5;
            require(a >= -.5 && a < n - .5 && b >= -.5 && b < n - .5, "Hydro slice outside domain");
            const int i = std::lround(a), j = std::lround(b);
            require(close(r[2], -length / 2 + (i + .5) * dx, 0, 1e-10 * dx) &&
                close(r[3], -length / 2 + (j + .5) * dx, 0, 1e-10 * dx),
                "Hydro slice position mismatch");
            require(r[5] > 0 && r[9] > 0 && r[10] >= 0 && r[11] >= 0,
                "Nonphysical hydro profile density, temperature or radiation energy");
            require(std::hypot(r[12], r[13], r[14]) <= m.at("c").get<double>() * r[11] * (1 + 1e-10),
                "Hydro profile radiation flux exceeds c E");
            auto &cell = data.rows[size_t(i) * n + j];
            require(cell.empty(), "Duplicate hydro slice cell");
            cell = r;
            ++count;
        }
    }
    require(count == size_t(n) * n, "Missing hydro slice cells");
    return data;
}
namespace {
Json checkpoint_provenance(const fs::path &folder, const Json &m)
{
    const auto path = folder / "restart-provenance.json";
    if (!fs::exists(path)) {
        require(!m.contains("checkpoint_resume"), "Missing declared checkpoint provenance");
        return {};
    }
    require(fs::is_regular_file(path) && !fs::is_symlink(path), "Invalid checkpoint provenance file");
    const auto p = read_json(path);
    require(p.at("schema") == 1, "Unsupported checkpoint provenance schema");
    const auto &spec = p.at("checkpoint_resume");
    const double start = spec.at("checkpoint_time"), end = m.at("time");
    require(std::isfinite(start) && start > 0 && start < end,
        "Checkpoint time must be inside the completed physical interval");
    const std::regex digest("[0-9a-f]{64}"), main_name(R"(X\.([0-9]+)\.silo)");
    std::smatch match;
    const auto checkpoint = spec.at("checkpoint").get<std::string>();
    require(std::regex_match(checkpoint, match, main_name), "Invalid checkpoint snapshot name");
    const size_t last = std::stoull(match[1]);
    require(!spec.at("source_attempt_started_utc").get<std::string>().empty() &&
        !spec.at("source_runtime_profile").get<std::string>().empty() &&
        !spec.at("source_data_directory").get<std::string>().empty() &&
        !p.at("runtime_profile").get<std::string>().empty() &&
        std::regex_match(p.at("solver_library_sha256").get<std::string>(), digest),
        "Missing checkpoint runtime/attempt identity");
    const auto &hashes = spec.at("snapshot_sha256");
    require(hashes.is_object() && !hashes.empty() && last < hashes.size(), "Missing retained checkpoint hashes");
    for (size_t i = 0; i <= last; ++i) {
        const std::string name = "X." + std::to_string(i) + ".silo";
        require(hashes.contains(name), "Incomplete retained checkpoint sequence");
        const auto shards = folder / (name + ".data");
        require(fs::is_directory(shards) && !fs::is_symlink(shards), "Missing retained checkpoint shards");
        size_t count = 0;
        for (const auto &entry : fs::directory_iterator(shards)) {
            require(entry.is_regular_file() && !entry.is_symlink() &&
                hashes.contains(name + ".data/" + entry.path().filename().string()),
                "Unrecorded retained checkpoint shard");
            ++count;
        }
        require(count > 0, "Empty retained checkpoint shards");
    }
    for (auto it = hashes.begin(); it != hashes.end(); ++it) {
        require(std::regex_match(it.key(), match, std::regex(R"(X\.([0-9]+)\.silo(?:\.data/[^/]+\.silo)?)")) &&
            std::stoull(match[1]) <= last && std::regex_match(it.value().get<std::string>(), digest),
            "Invalid retained checkpoint file/hash");
        const auto file = folder / it.key();
        require(fs::is_regular_file(file) && !fs::is_symlink(file) && fs::file_size(file) > 0 &&
            sha256(file) == it.value().get<std::string>(), "Retained checkpoint hash differs");
    }
    const auto &segments = p.at("conservation_segments");
    require(segments.is_array() && segments.size() == 2 &&
        segments[0].at("file") == "conservation-segments/original-attempt.csv" &&
        segments[1].at("file") == "radiation-conservation.csv" &&
        segments[0].at("used_until_checkpoint_time") == start &&
        segments[1].at("expected_start_time") == start &&
        segments[1].at("diagnostic_counters_reset") == true,
        "Invalid checkpoint conservation segment declaration");
    const auto old = folder / "conservation-segments/original-attempt.csv";
    require(!fs::is_symlink(old.parent_path()) && fs::is_regular_file(old) && !fs::is_symlink(old) &&
        std::regex_match(spec.at("conservation_sha256").get<std::string>(), digest) &&
        sha256(old) == spec.at("conservation_sha256").get<std::string>(), "Original conservation segment hash differs");
    if (m.contains("checkpoint_resume")) {
        require(m.at("checkpoint_resume") == spec &&
            m.at("restart_provenance_sha256") == sha256(path), "Recorded checkpoint provenance differs");
    }
    return p;
}
} // namespace

void adopt_checkpoint_provenance(const fs::path &folder, Json &m)
{
    const auto p = checkpoint_provenance(folder, m);
    if (!p.is_null()) {
        m["checkpoint_resume"] = p.at("checkpoint_resume");
        m["restart_provenance_sha256"] = sha256(folder / "restart-provenance.json");
        m["performance_eligible"] = false;
        m["queue_calibration_eligible"] = false;
    }
}

std::string conservation_scope_note(const Budget &b, const Json &m)
{
    if (b.history.rows.front()[0] == 0) { return ""; }
    return "<p class=\"note\"><strong>Checkpoint continuation conservation segment:</strong> t = " +
        html(number(b.history.rows.front()[0])) + " to " + html(number(b.history.rows.back()[0])) +
        (cgs(m) ? " s. " : " in normalized time. ") +
        "The reference time t_ref and fixed normalization are the checkpoint state. Boundary/source "
        "counters restart at zero; residuals describe this segment only. The original ledger is retained "
        "separately; its rows after the checkpoint belong to the abandoned attempt and are not joined "
        "to this history. Whole-run conservation is not inferred from this segment.</p>";
}

std::optional<Budget> read_conservation(const fs::path &folder, const Json &m)
{
	auto p = folder / "radiation-conservation.csv";
	if (!fs::exists(p)) {
    return {};
  }
  std::vector<std::string> cols{"t", "volume"};
	for (auto &f : fields) {
    cols.push_back(f);
  }
  for (auto &f : fields) {
    cols.push_back(f + "_boundary");
  }
  for (auto &f : fields) {
    cols.push_back(f + "_source");
  }
  Budget b;
	b.history = read_csv(p, cols);
	auto &rows = b.history.rows;
	double len = m.at("length"), c = m.at("c"), vol = len * len * len;
	require(len > 0 && c > 0 && std::isfinite(vol) && std::isfinite(c), "Invalid conservation metadata");
    const auto restart = checkpoint_provenance(folder, m);
    const bool resumed = !restart.is_null();
    require(!resumed || m.contains("checkpoint_resume"), "Checkpoint provenance must be adopted before reporting");
    const double start = resumed ? restart.at("checkpoint_resume").at("checkpoint_time").get<double>() : 0;
    if (resumed) {
        const auto old = read_csv(folder / "conservation-segments/original-attempt.csv", cols);
        require(old.rows.front()[0] == 0 && old.rows.back()[0] >= start,
            "Original conservation segment does not cover checkpoint time");
        for (size_t i = 1; i < old.rows.size(); ++i) {
            require(old.rows[i][0] > old.rows[i - 1][0], "Original conservation times must increase");
        }
    }
    require((resumed ? close(rows[0][0], start, 1e-12) : rows[0][0] == 0) &&
        close(rows.back()[0], m.at("time")), "Conservation must span its declared initial time through final time");
	for (int j = 6; j < 14; ++j) {
    require(rows[0][j] == 0, "Nonzero initial boundary/source ledger");
  }
  b.residual.resize(rows.size());
	b.scale.resize(rows.size());
	b.normalized.resize(rows.size());
	// Fix normalization at the declared initial time: growing source/boundary budgets must
	// not reduce the apparent fractional conservation error later in the run.
	const bool marshak = m.value("case", "") == "marshak";
    // The cold Marshak domain starts with zero radiation. Its prescribed bath
    // provides a fixed, physically meaningful scale independent of the run.
	const double initial_energy = marshak ? 4 * vol / c : std::abs(rows[0][2]);
	const double initial_flux_scale = c * initial_energy;
	constexpr double zero_tolerance = 64 * std::numeric_limits<double>::epsilon();
	require(initial_energy > 0 && std::isfinite(initial_flux_scale) && initial_flux_scale > 0,
			"Conservation fractions require a nonzero initial energy and finite c times that energy");
	std::array<double, 4> initial_scale;
	std::array<std::string, 4> basis;
	initial_scale[0] = initial_energy;
	basis[0] = marshak ? "bath_energy * volume" : resumed ? "abs(segment_initial_energy)" : "abs(initial_energy)";
	for (int f = 1; f < 4; ++f) {
		const double initial = std::abs(rows[0][2 + f]);
		const bool zero_net_flux = marshak || initial / initial_flux_scale <= zero_tolerance;
		initial_scale[f] = zero_net_flux ? initial_flux_scale : initial;
		basis[f] = marshak ? "c * bath_energy * volume" : zero_net_flux ?
            (resumed ? "c * abs(segment_initial_energy)" : "c * abs(initial_energy)") :
            (resumed ? "abs(segment_initial_component)" : "abs(initial_component)");
	}
	for (size_t i = 0; i < rows.size(); ++i) {
		auto &r = rows[i];
		require(close(r[1], vol), "Conservation domain-volume mismatch");
		require(!i || r[0] > rows[i - 1][0], "Conservation times must increase");
		for (int f = 0; f < (fld_model(m) ? 1 : 4); ++f) {
			double residual = r[2 + f] - rows[0][2 + f] + r[6 + f] - r[10 + f];
			double scale = initial_scale[f];
			require(std::isfinite(residual) && std::isfinite(residual / scale),
					"Invalid derived conservation budget");
			b.residual[i][f] = residual;
			b.scale[i][f] = scale;
			b.normalized[i][f] = residual / scale;
		}
	}
	for (int f = 0; f < (fld_model(m) ? 1 : 4); ++f) {
		double maxr = 0, maxn = 0;
		for (size_t i = 0; i < rows.size(); ++i) {
			maxr = std::max(maxr, std::abs(b.residual[i][f]));
			maxn = std::max(maxn, std::abs(b.normalized[i][f]));
		}
		b.summary.push_back({{"field", fields[f]},
                             {"history_scope", resumed ? "checkpoint_continuation_segment" : "full_run"},
                             {"history_start_time", rows.front()[0]}, {"history_end_time", rows.back()[0]},
                             {"normalization_time", start},
							 {"initial", rows[0][2 + f]},
							 {"final", rows.back()[2 + f]},
							 {"raw_change", rows.back()[2 + f] - rows[0][2 + f]},
							 {"boundary", rows.back()[6 + f]},
							 {"source", rows.back()[10 + f]},
							 {"residual", b.residual.back()[f]},
							 {"normalization_scale", b.scale.back()[f]},
							 {"normalization_basis", basis[f]},
							 {"zero_initial_flux_relative_tolerance", zero_tolerance},
							 {"initial_fraction", rows[0][2 + f] / initial_scale[f]},
							 {"final_fraction", rows.back()[2 + f] / initial_scale[f]},
							 {"boundary_fraction", rows.back()[6 + f] / initial_scale[f]},
							 {"source_fraction", rows.back()[10 + f] / initial_scale[f]},
							 {"normalized_residual", b.normalized.back()[f]},
							 {"normalized_error", std::abs(b.normalized.back()[f])},
							 {"max_abs_residual", maxr},
							 {"max_normalized_error", maxn},
							 {"integral_units", cgs(m) ? (f == 0 ? "erg" : "erg cm/s") : "code units"}});
	}
	return b;
}
void CGSCheck::feed(const std::string &line)
{
	std::smatch m;
	if (std::regex_search(line, std::regex(R"(^\s*normalized\s+constants\b)", std::regex::icase))) {
		require(!pending, "Incomplete normalization block");
		pending = true;
		factors = false;
		return;
	}
	if (pending && !factors) {
		std::istringstream s(line);
		std::vector<double> v;
		double d;
		while (s >> d) {
      v.push_back(d);
    }
    if (v.size() == 4 && s.eof()) {
			for (double x : v) {
        require(close(x, 1, 1e-12), "Solver is not using CGS unit factors; rebuild");
      }
      factors = true;
		}
	}
	if (std::regex_search(line, m, std::regex(R"(\|\s*c\s*=\s*(\S+)\s*\|)"))) {
		require(close(numeric(m[1]), expected_c, 5e-7), "Wrong solver light speed for recorded units; rebuild");
		if (pending && factors) {
			++blocks;
			pending = false;
		}
	}
}
void CGSCheck::finish()
{
	require(blocks && !pending, "Could not verify the running solver's CGS units and light speed;");
}
void verify_cgs_log(const fs::path &p, double expected_c)
{
	CGSCheck c;
    c.expected_c = expected_c;
	std::istringstream s(read_text(p));
	std::string line;
	while (std::getline(s, line)) {
    c.feed(line);
  }
  c.finish();
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
std::vector<size_t> frame_schedule(const std::vector<double> &t, double seconds, int fps, double hold)
{
	require(t.size() >= 2 && std::isfinite(seconds) && seconds > 0 && fps > 0 && std::isfinite(hold) &&
				hold >= 0,
			"Invalid playback parameters");
	for (size_t i = 0; i < t.size(); ++i) {
    require(std::isfinite(t[i]) && (!i || t[i] > t[i - 1]), "Invalid snapshot times");
  }
  require(seconds * fps < 1e7 && hold * fps < 1e7, "Too many video frames");
	auto total = std::llround(seconds * fps), pause = std::llround(hold * fps), active = total - 2 * pause;
	require(active >= 2, "Movie duration must exceed endpoint holds");
	std::vector<size_t> seq(pause, 0);
	for (long long i = 0; i < active; ++i) {
		double value = t.front() + (t.back() - t.front()) * i / (active - 1);
		auto it = std::upper_bound(t.begin(), t.end(), value);
		seq.push_back(it == t.begin() ? 0 : size_t(it - t.begin() - 1));
	}
	seq.back() = t.size() - 1;
	seq.insert(seq.end(), pause, t.size() - 1);
	return seq;
}
std::vector<fs::path> numerical_silos(const fs::path &dir)
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
	for (auto &[i, p] : sorted) {
    v.push_back(p);
  }
  v.push_back(rr::absolute(dir / "final.silo"));
	for (auto &p : v) {
    require(fs::file_size(p) > 0, "Empty Silo file");
  }
  return v;
}
std::vector<std::pair<fs::path, Json>> completed_runs(const fs::path &dir)
{
	std::vector<std::pair<fs::path, Json>> out;
	auto add = [&](const fs::path &p) {
		if (!fs::is_regular_file(p / "run.json")) {
      return;
    }
    auto m = read_json(p / "run.json");
		if (m.value("status", "") != "complete") {
      return;
    }
    auto name = m.at("case").get<std::string>();
		require(std::find(cases.begin(), cases.end(), name) != cases.end(), "Unknown case: " + name);
		out.emplace_back(p, m);
	};
	if (fs::is_regular_file(dir / "run.json")) {
    add(dir);
  } else {
    for (auto &c : cases) {
      if (fs::is_directory(dir / c)) {
        for (auto &e : fs::directory_iterator(dir / c)) {
          if (e.is_directory() && e.path().filename().string().starts_with('l')) {
            add(e.path());
          }
        }
      }
    }
  }
  std::sort(out.begin(), out.end(), [](auto &a, auto &b) {
		return std::make_pair(a.second.at("cells").template get<int>(),
							  a.second.at("case").template get<std::string>()) <
			   std::make_pair(b.second.at("cells").template get<int>(),
							  b.second.at("case").template get<std::string>());
	});
	return out;
}
Json slice_mesh_summary(const fs::path &folder, const Json &m)
{
    const bool hydro = coupled_case(m.at("case"));
    const std::vector<std::string> columns = hydro ?
        std::vector<std::string>{"t", "dx", "x", "y", "z", "rho", "vx", "vy", "vz", "gas_temperature", "radiation_temperature", "er", "fx", "fy", "fz"} :
        std::vector<std::string>{"t", "dx", "x", "y", "z", "er", "fx", "fy", "fz", "er_ref", "fx_ref", "fy_ref", "fz_ref"};
    Json counts = Json::object();
    size_t total = 0;
    for (const auto &entry : fs::directory_iterator(folder / (hydro ? "radiation-hydro-slices" : "radiation-slices"))) {
        if (entry.path().extension() != ".csv") {
            continue;
        }
        for (const auto &row : read_csv(entry.path(), columns).rows) {
            const int level = m.at("level").get<int>() - int(std::llround(std::log2(row[1] / m.at("dx").get<double>())));
            const auto key = std::to_string(level);
            counts[key] = counts.value(key, size_t(0)) + 1;
            ++total;
        }
    }
    Json result{{"native_slice_cells", total}, {"slice_cells_by_level", counts},
        {"projection", "Piecewise-constant leaf cell averages onto finest display pixels; raw CSV retains native cells"},
        {"geometry", "Positive-side leaf cells touching geometric z=0; cell centers have z=leaf_dx/2"}};
    if (fs::is_regular_file(folder / "gradient-amr.csv")) {
        const auto history = read_csv(folder / "gradient-amr.csv", {"t", "step", "level", "leaf_blocks", "leaf_cells"});
        const auto last = history.rows.back();
        Json leaves = Json::object();
        size_t blocks = 0, cells = 0;
        double covered_volume_fraction = 0;
        const int inx = m.at("inx");
        require(last[0] >= 0 && last[0] <= m.at("time").get<double>() * (1 + 1.e-12) && last[1] >= 0,
            "Adaptive mesh census time/step outside the run");
        for (const auto &row : history.rows) {
            if (row[0] == last[0] && row[1] == last[1]) {
                require(row[2] >= m.at("min_level").get<int>() && row[2] <= m.at("level").get<int>() &&
                    row[3] >= 1 && row[4] >= 1 && row[2] == std::floor(row[2]) &&
                    row[3] == std::floor(row[3]) && row[4] == std::floor(row[4]), "Invalid adaptive mesh census");
                const auto key = std::to_string(int(row[2]));
                require(!leaves.contains(key), "Duplicate level in adaptive mesh census");
                require(row[4] == row[3] * inx * inx * inx, "Adaptive mesh census cells disagree with block count");
                const double side_cells = std::ldexp(double(inx), int(row[2]));
                covered_volume_fraction += row[4] / (side_cells * side_cells * side_cells);
                leaves[key] = {{"leaf_blocks", size_t(row[3])}, {"leaf_cells", size_t(row[4])}};
                blocks += size_t(row[3]);
                cells += size_t(row[4]);
            }
        }
        require(close(covered_volume_fraction, 1.), "Adaptive mesh census does not cover the full domain");
        result["full_mesh"] = {{"last_regrid_time", last[0]}, {"last_regrid_step", last[1]},
            {"leaf_blocks", blocks}, {"leaf_cells", cells}, {"levels", leaves}};
    }
    return result;
}
std::string mesh_label(const Json &m)
{
    if (!m.value("adaptive", false)) {
        return std::to_string(m.at("cells").get<int>()) + "³ cells";
    }
    std::string label = "AMR levels " + std::to_string(m.at("min_level").get<int>()) + "–" +
        std::to_string(m.at("level").get<int>());
    if (m.contains("mesh") && m["mesh"].contains("full_mesh")) {
        label += "; " + std::to_string(m["mesh"]["full_mesh"]["leaf_cells"].get<size_t>()) + " active cells";
    }
    return label + "; finest equivalent " + std::to_string(m.at("cells").get<int>()) + "³";
}
std::string slice_geometry_note(const Json &m)
{
    return m.value("adaptive", false) ?
        "AMR: positive-side cells touching z=0, projected as constant leaf averages. Centers have z=leaf_dx/2; the ray similarly touches y=0. This is not a common cell-center plane. Raw CSV preserves native cells; display CSV repeats coarse values on finest pixels, retaining source dx/z. Full-volume norms use actual leaf volumes; uniform-grid convergence orders are not reported." :
        "Central cell layer at z=dx/2; ray at y=z=dx/2.";
}
} // namespace rr
