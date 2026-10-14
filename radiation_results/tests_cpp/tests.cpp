#include <stddef.h>
#include "silo_reader.hpp"
#include <silo.h>
using namespace rr;
namespace
{
int checks = 0;
void expect(bool ok, const std::string &what)
{
	require(ok, "TEST FAILED: " + what);
	++checks;
}
template <class F> void rejects(F f, const std::string &what)
{
	bool failed = false;
	try {
		f();
	} catch (const std::exception &) {
		failed = true;
	}
	expect(failed, what);
}
Json metadata(std::string name, int level = 0)
{
	int n = 4 * (1 << level);
	double length = name == "marshak" ? 2. : 6e10;
	return {{"case", name},
			{"level", level},
			{"inx", 4},
			{"cells", n},
			{"length", length},
			{"dx", length / n},
			{"time", name == "marshak" ? 1. : 4.},
			{"c", name == "marshak" ? 1. : c_cgs},
			{"units", name == "marshak" ? marshak_units : cgs_units},
			{"background", 1.},
			{"status", "complete"},
			{"origin", "synthetic validation fixture"},
			{"executable_sha256", "fixture"},
			{"comparison_signature", {{"fixture", "1"}}},
			{"movie_capture", {{"requested_snapshots", 3}}}};
}
void results(const fs::path &p, Json m)
{
	fs::create_directories(p / "radiation-slices");
	write_json(p / "run.json", m);
	int n = m.at("cells");
	double dx = m.at("dx"), len = m.at("length"), time = m.at("time");
	std::string name = m.at("case");
	std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
	std::ostringstream log;
	log << std::setprecision(17) << "normalized constants\n1 1 1 1\n| c = " << m.at("c").get<double>()
		<< " |\nRADIATION_TEST_FINISHED " << (m.at("case") == "marshak" ? "" : "RADIATION_") << name << " t=" << time << '\n';
    if (coupled_case(m.at("case"))) {
        atomic_text(p / "run.log", log.str());
        std::ostringstream csv;
        csv << "t,dx,x,y,z,rho,vx,vy,vz,gas_temperature,radiation_temperature,er,fx,fy,fz\n"
            << std::scientific << std::setprecision(17);
        for (int i = n - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                const double x = -len / 2 + (i + .5) * dx, y = -len / 2 + (j + .5) * dx;
                const double temperature = 100 + 10 * std::cos(6.283185307179586 * x / len);
                csv << time << ',' << dx << ',' << x << ',' << y << ',' << dx / 2 <<
                    ",7.78e-10,-600000,0,0," << temperature << ',' << temperature / 2 <<
                    ",1,0,0,0\n";
            }
        }
        atomic_text(p / "radiation-hydro-slices/slice-000.csv", csv.str());
        return;
    }
    if (fld_model(m)) {
        log.str("");
        log.clear();
        log << std::setprecision(17) << "normalized constants\n1 1 1 1\n| c = " << m.at("c").get<double>() << " |\n";
        const auto cfg = m.value("config", Json::object());
        const double cap = numeric(cfg.value("hard_dt", number(time)));
        for (double t = 0.; t < time;) {
            const double dt = std::min(cap, time - t);
            log << "FLD accepted t=" << t << " dt=" << dt << " substeps=1 retries=0 nonlinear_iterations=3 "
                "linear_iterations=9 vcycles=9 nonlinear_residual=1e-12 linear_residual=1e-13 "
                "boundary_energy=0 conservation_error=0\n";
            t += dt;
        }
        log << "RADIATION_FLD_FINISHED " << (m.at("case") == "marshak" ? "" : "RADIATION_") << name << " t=" << time << " requested=" << time << '\n';
        atomic_text(p / "run.log", log.str());
        std::ostringstream csv, gas;
        csv << "t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref\n" << std::scientific << std::setprecision(17);
        gas << "t,dx,x,y,z,egas,egas_ref\n" << std::scientific << std::setprecision(17);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double x = -len / 2 + (i + .5) * dx, y = -len / 2 + (j + .5) * dx;
                const double energy = 1 + .5 * std::cos(2 * 3.141592653589793 * x / len);
                csv << time << ',' << dx << ',' << x << ',' << y << ',' << dx / 2 << ',' << energy + 1.e-3 / n << ",0,0,0," << energy << ",0,0,0\n";
                gas << time << ',' << dx << ',' << x << ',' << y << ',' << dx / 2 << ',' << energy / 2 + 2.e-3 / n << ',' << energy / 2 << '\n';
            }
        }
        atomic_text(p / "radiation-slices/slice-000.csv", csv.str());
        if (m.at("case") == "marshak") { atomic_text(p / "marshak-material-slices/slice-000.csv", gas.str()); }
        const double volume = len * len * len;
        std::ostringstream budget;
        budget << "t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,er_source,fx_source,fy_source,fz_source\n"
            << std::scientific << std::setprecision(17);
        for (double t : {0., time / 4, time}) {
            // Flux diagnostic integrals deliberately change without a moment ledger.
            // FLD must assess only its evolved radiation energy equation.
            budget << t << ',' << volume << ',' << volume * (1 + t) << ',' << t << ",0,0,0,0,0,0," << volume * t << ",0,0,0\n";
        }
        atomic_text(p / "radiation-conservation.csv", budget.str());
        return;
    }
	double error = 1e-3 / n;
	for (int f = 0; f < 4; ++f) {
    log << fields[f] << ' ' << (f == 0 ? error : 0) << ' ' << (f == 0 ? 2 * error : 0) << ' '
			<< (f == 0 ? 4 * error : 0) << '\n';
  }
  if (m.at("case") == "marshak") {
        log << "egas " << 2 * error << ' ' << 4 * error << ' ' << 8 * error << '\n';
    }
  atomic_text(p / "run.log", log.str());
	for (int k = 0; k < 3; ++k) {
    atomic_text(p / (norms[k] + ".dat"), number(dx) + " " + std::to_string(m.at("level").get<int>()) +
												 (m.at("case") == "marshak" ? " 0 " + number(2 * error * (1 << k)) : "") + " " + number(error * (1 << k)) + " 0 0 0\n");
  }
  std::ostringstream csv;
	csv << "t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref\n" << std::scientific << std::setprecision(17);
	for (int i = n - 1; i >= 0; --i) {
    for (int j = n - 1; j >= 0; --j) {
			double x = -len / 2 + (i + .5) * dx, y = -len / 2 + (j + .5) * dx,
				   ref = 1 + .5 * std::cos(2 * 3.141592653589793 * x / len);
			csv << time << ',' << dx << ',' << x << ',' << y << ',' << dx / 2 << ',' << ref + error
				<< ",0,0,0," << ref << ",0,0,0\n";
		}
  }
  atomic_text(p / "radiation-slices/slice-000.csv", csv.str());
    if (m.at("case") == "marshak") {
        std::ostringstream gas;
        gas << "t,dx,x,y,z,egas,egas_ref\n" << std::scientific << std::setprecision(17);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const double x=-len/2+(i+.5)*dx,y=-len/2+(j+.5)*dx;
                const double value=.5+.25*std::cos(2*3.141592653589793*x/len);
                gas << time << ',' << dx << ',' << x << ',' << y << ',' << dx/2 << ',' << value+2*error << ',' << value << '\n';
            }
        }
        atomic_text(p / "marshak-material-slices/slice-000.csv", gas.str());
    }
	double volume = len * len * len;
	std::ostringstream budget;
	budget << "t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,er_source,fx_source,fy_"
			  "source,fz_source\n"
		   << std::scientific << std::setprecision(17);
	for (double t : {0., time / 4, time}) {
    budget << t << ',' << volume << ',' << volume * (1 + t) << ",0,0,0,0,0,0,0," << volume * t
			   << ",0,0,0\n";
  }
  atomic_text(p / "radiation-conservation.csv", budget.str());
}
void checkpoint_results(const fs::path &folder, const Json &m)
{
    const double end = m.at("time"), start = end / 2, volume = std::pow(m.at("length").get<double>(), 3);
    const std::string header = "t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,er_source,fx_source,fy_source,fz_source\n";
    auto history = [&](double initial, const std::vector<double> &times) {
        std::ostringstream text;
        text << header << std::scientific << std::setprecision(17);
        for (double t : times) {
            text << t << ',' << volume << ',' << volume * (1 + t - initial)
                << ",0,0,0,0,0,0,0," << volume * (t - initial) << ",0,0,0\n";
        }
        return text.str();
    };
    atomic_text(folder / "conservation-segments/original-attempt.csv", history(0, {0, start, end}));
    atomic_text(folder / "radiation-conservation.csv", history(start, {start, (start + end) / 2, end}));
    Json hashes = Json::object();
    for (const auto &name : {"X.0.silo", "X.0.silo.data/0.silo", "X.1.silo", "X.1.silo.data/0.silo"}) {
        atomic_text(folder / name, std::string("synthetic checkpoint fixture ") + name);
        hashes[name] = sha256(folder / name);
    }
    write_json(folder / "restart-provenance.json", {{"schema", 1}, {"runtime_profile", "fixture-recovery"},
        {"solver_library_sha256", std::string(64, 'a')}, {"checkpoint_resume", {
            {"source_attempt_started_utc", "2026-10-12T00:00:00Z"}, {"source_runtime_profile", "baseline"},
            {"source_data_directory", "attempts/fixture"}, {"checkpoint", "X.1.silo"},
            {"checkpoint_time", start}, {"snapshot_sha256", hashes},
            {"conservation_sha256", sha256(folder / "conservation-segments/original-attempt.csv")}}},
        {"conservation_segments", Json::array({{{"file", "conservation-segments/original-attempt.csv"},
            {"used_until_checkpoint_time", start}}, {{"file", "radiation-conservation.csv"},
            {"expected_start_time", start}, {"diagnostic_counters_reset", true}}})}});
}
void adaptive_results(const fs::path &folder, const Json &m)
{
    const bool hydro = coupled_case(m.at("case"));
    auto uniform = m;
    uniform["adaptive"] = false;
    const auto data = hydro ? read_hydro_profile(folder, uniform) : read_slice(folder, uniform);
    const int n = m.at("cells");
    const double h = m.at("dx"), length = m.at("length");
    const auto directory = folder / (hydro ? "radiation-hydro-slices" : "radiation-slices");
    fs::remove(directory / "slice-000.csv");
    Json rows = Json::array();
    for (int i = 0; i < n;) {
        const int width = i < n / 2 ? 2 : 1;
        for (int j = 0; j < n; j += width) {
            auto row = data.rows[size_t(i) * n + j];
            row[1] = width * h;
            row[2] = -length / 2 + (i + width / 2.) * h;
            row[3] = -length / 2 + (j + width / 2.) * h;
            row[4] = row[1] / 2;
            Json record;
            for (size_t k = 0; k < row.size(); ++k) {
                record[data.columns[k]] = row[k];
            }
            rows.push_back(record);
        }
        i += width;
    }
    records_csv(directory / "slice-mixed.csv", rows, data.columns);
    std::string header;
    for (const auto &column : data.columns) {
        header += (header.empty() ? "" : ",") + column;
    }
    const auto csv = read_text(directory / "slice-mixed.csv");
    atomic_text(directory / "slice-mixed.csv", header + csv.substr(csv.find('\n')));
    const int base = 1 << (3 * m.at("min_level").get<int>());
    const int inx = m.at("inx");
    atomic_text(folder / "gradient-amr.csv", "t,step,level,leaf_blocks,leaf_cells\n0,0," +
        std::to_string(m.at("min_level").get<int>()) + "," + std::to_string(base / 2) + "," +
        std::to_string(base / 2 * inx * inx * inx) + "\n0,0," + std::to_string(m.at("level").get<int>()) +
        "," + std::to_string(4 * base) + "," + std::to_string(4 * base * inx * inx * inx) + "\n");
}
void silos(const fs::path &folder, const Json &m)
{
	int n = m.at("cells");
	double length = m.at("length"), half = length / 2;
	int vdims[3]{n / 2, n, n}, mdims[3]{n / 2 + 1, n + 1, n + 1};
	for (auto entry : std::vector<std::pair<std::string, double>>{
			 {"X.0.silo", 0}, {"X.1.silo", 1}, {"X.2.silo", 4}, {"final.silo", 4}}) {
		auto file = folder / entry.first, dir = fs::path(file.string() + ".data");
		fs::create_directories(dir);
		auto data = dir / "0.silo";
		DBfile *db = DBCreate(data.c_str(), DB_CLOBBER, DB_LOCAL, "fixture", DB_HDF5);
		require(db, "fixture Silo create failed");
		std::array<std::vector<double>, 3> x;
		for (int d = 0; d < 3; ++d) {
      x[d].resize(mdims[d]);
    }
    double time = entry.second;
		auto opt = DBMakeOptlist(4);
		DBAddOption(opt, DBOPT_DTIME, &time);
		int order = DB_ROWMAJOR;
		DBAddOption(opt, DBOPT_MAJORORDER, &order);
		for (int block = 0; block < 2; ++block) {
			auto name = "block" + std::to_string(block);
			DBMkDir(db, name.c_str());
			DBSetDir(db, name.c_str());
			for (int d = 0; d < 3; ++d) {
        for (int i = 0; i < mdims[d]; ++i) {
          x[d][i] = -half + (d == 0 ? block * half : 0) + i * length / n;
        }
      }
      double *coords[]{x[0].data(), x[1].data(), x[2].data()};
			const char *labels[]{"x", "y", "z"};
			require(DBPutQuadmesh(db, "mesh", labels, coords, mdims, 3, DB_DOUBLE, DB_COLLINEAR, opt) >= 0,
					"fixture mesh failed");
			for (int f = 0; f < 4; ++f) {
				std::vector<double> v(size_t(n) * n * n / 2);
				for (int k = 0; k < n; ++k) {
          for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n / 2; ++i) {
							double px = (x[0][i] + x[0][i + 1]) / 2 / length,
								   py = (x[1][j] + x[1][j + 1]) / 2 / length,
								   pz = (x[2][k] + x[2][k + 1]) / 2 / length;
							v[i + (n / 2) * (j + n * k)] = (f + 1) * (2 + px + 2 * py + 3 * pz + time / 10);
						}
          }
        }
        require(DBPutQuadvar1(db, fields[f].c_str(), "mesh", v.data(), vdims, 3, nullptr, 0,
									  DB_DOUBLE, DB_ZONECENT, opt) >= 0,
						"fixture quadvar failed");
			}
			DBSetDir(db, "/");
		}
		DBFreeOptlist(opt);
		DBClose(db);
		db = DBCreate(file.c_str(), DB_CLOBBER, DB_LOCAL, "fixture root", DB_HDF5);
		require(db, "fixture root failed");
		opt = DBMakeOptlist(2);
		DBAddOption(opt, DBOPT_DTIME, &time);
		std::string mesh0 = entry.first + ".data/0.silo:/block0/mesh",
					mesh1 = entry.first + ".data/0.silo:/block1/mesh";
		const char *mesh_names[] = {mesh0.c_str(), mesh1.c_str()};
		int mesh_types[] = {DB_QUAD_RECT, DB_QUAD_RECT};
		require(DBPutMultimesh(db, "mesh", 2, mesh_names, mesh_types, opt) >= 0, "fixture multimesh failed");
		char mesh_name[] = "mesh";
		DBAddOption(opt, DBOPT_MMESH_NAME, mesh_name);
		for (auto &f : fields) {
			std::string a = entry.first + ".data/0.silo:/block0/" + f,
						b = entry.first + ".data/0.silo:/block1/" + f;
			const char *names[]{a.c_str(), b.c_str()};
			int types[]{DB_QUADVAR, DB_QUADVAR};
			require(DBPutMultivar(db, f.c_str(), 2, names, types, opt) >= 0, "fixture multivar failed");
		}
		DBFreeOptlist(opt);
		DBClose(db);
	}
}
} // namespace
int main(int argc, char **argv)
{
	try {
		if (argc > 1 && std::string(argv[1]) == "--fake-visit") {
			// Test double for the terminal/encoder regression, not a VisIt compatibility test.
			std::map<std::string, std::string> opts;
			for (int i = 2; i + 1 < argc; ++i) {
				std::string arg = argv[i];
				if (arg == "-sessionfile" || arg == "-output" || arg == "-end") {
          opts[arg] = argv[++i];
        }
      }
			require(read_text(opts.at("-sessionfile")).find("Pseudocolor_1.0") != std::string::npos,
					"Missing session argument");
			const char *mode_env = std::getenv("RR_TEST_VISIT_MODE");
			std::string mode = mode_env ? mode_env : "";
			int last = std::stoi(opts.at("-end"));
			for (int i = 0; i <= last; ++i) {
				if (mode == "partial" && i == last) {
          continue;
        }
        std::ostringstream out;
				out << opts.at("-output") << std::setfill('0') << std::setw(4) << i << ".png";
				if (mode == "corrupt") {
          atomic_text(out.str(), "not a PNG");
        } else {
          fs::copy_file(std::getenv("RR_TEST_FRAME"), out.str());
        }
      }
			if (mode == "error") {
        std::cout << "VisIt: Error - test rendering error\n";
      }
      std::cout << "VisIt completed generating frames.\n";
			return (mode == "shutdown" || mode == "partial" || mode == "corrupt") ? 250 : 0;
		}

		if (argc > 1 && std::string(argv[1]).starts_with("--config_file=")) {
			const auto p = fs::current_path();
			auto m = read_json(p / "run.json");
			results(p, m);
            if (std::getenv("RR_TEST_CHECKPOINT")) {
                checkpoint_results(p, m);
            }
            if (m.value("adaptive", false)) {
                adaptive_results(p, m);
            }
			std::ofstream count(p / "solver-executions", std::ios::app);
			count << "run\n";
			count.close();
			std::cout << read_text(p / "run.log");
			return 0;
		}
		if (argc == 3 && std::string(argv[1]) == "--fixtures") {
			auto dir = rr::absolute(argv[2]);
			for (auto &name : cases) {
        for (int level = 0; level < 2; ++level) {
					auto p = dir / name / ("l" + std::to_string(level));
					auto m = metadata(name, level);
					results(p, m);
					if (name == cases[0] && level == 0) {
            silos(p, m);
          }
        }
      }
      return 0;
		}
		auto tmp = fs::temp_directory_path() / ("rr-tests-" + stamp());
		fs::create_directory(tmp);
		auto m = metadata(cases[0]);
		auto p = tmp / "wave";
		results(p, m);
		auto ns = read_norms(p, m);
		expect(close(ns["er"]["L1"], 2.5e-4), "norm values");
		auto slice = read_slice(p, m);
		expect(slice.rows.size() == 16 && slice.rows.front()[2] < slice.rows.back()[2],
			   "sort and validate distributed slice cells");
		auto b = read_conservation(p, m);
		expect(b && b->summary[0]["max_normalized_error"].get<double>() < 1e-15,
			   "physical source budget cancellation");
		expect(b->summary[1]["normalized_error"] == 0, "zero net flux normalization");
        {
            auto resumed_meta = m;
            const auto resumed = tmp / "checkpoint";
            results(resumed, resumed_meta);
            checkpoint_results(resumed, resumed_meta);
            const auto original = read_text(resumed / "conservation-segments/original-attempt.csv");
            rejects([&] { read_conservation(resumed, resumed_meta); }, "undeclared nonzero history start rejected");
            adopt_checkpoint_provenance(resumed, resumed_meta);
            const auto segment = *read_conservation(resumed, resumed_meta);
            expect(segment.history.rows.front()[0] == 2 && segment.history.rows.back()[0] == 4 &&
                segment.summary[0]["normalization_time"] == 2 &&
                segment.summary[0]["normalization_basis"] == "abs(segment_initial_energy)",
                "resumed segment has physical checkpoint times and fixed checkpoint normalization");
            expect(segment.scale.front() == segment.scale.back() &&
                segment.summary[0]["max_normalized_error"].get<double>() < 1e-15,
                "resumed segment budget closes without rebasing timestamps or joining old tail");
            expect(conservation_scope_note(segment, resumed_meta).find("abandoned attempt") != std::string::npos &&
                conservation_scope_note(segment, resumed_meta).find("t = 2 to 4 s") != std::string::npos,
                "visible continuation note states time, normalization and abandoned-tail scope");
            expect(read_text(resumed / "conservation-segments/original-attempt.csv") == original,
                "original full raw ledger remains byte-identical");
            auto wrong = resumed_meta;
            wrong["checkpoint_resume"]["checkpoint_time"] = 1.;
            rejects([&] { read_conservation(resumed, wrong); }, "changed metadata checkpoint rejected");
            const auto ledger = read_text(resumed / "radiation-conservation.csv");
            atomic_text(resumed / "radiation-conservation.csv", ledger.substr(0, ledger.find('\n') + 1) +
                "1,1,1,0,0,0,0,0,0,0,0,0,0,0\n4,1,1,0,0,0,0,0,0,0,0,0,0,0\n");
            rejects([&] { read_conservation(resumed, resumed_meta); }, "wrong actual checkpoint ledger start rejected");
            atomic_text(resumed / "radiation-conservation.csv", ledger);
            atomic_text(resumed / "X.1.silo.data/0.silo", "changed checkpoint shard");
            rejects([&] { read_conservation(resumed, resumed_meta); }, "changed retained snapshot rejected");
        }
		{
			auto original = read_text(p / "radiation-conservation.csv");
			auto small = m;
			small["length"] = 2.;
			small["c"] = 4.;
			atomic_text(p / "radiation-conservation.csv", original.substr(0, original.find('\n') + 1) +
															  "0,8,4,-2,0,1.6e-15,0,0,0,0,0,0,0,0\n"
															  "4,8,45,-1,4,8,0,0.5,0,0,40,0.25,2,0\n");
			auto fixed = *read_conservation(p, small);
			expect(close(fixed.normalized.back()[0], .25),
				   "source growth does not dilute initial-energy fractional error");
			expect(fixed.scale.front() == fixed.scale.back(),
				   "all conservation denominators remain fixed at t=0");
			expect(fixed.summary[1]["initial_fraction"] == -1 && fixed.summary[1]["final_fraction"] == -.5,
				   "negative initial component uses its magnitude and preserves signs");
			expect(close(fixed.normalized.back()[1], .625),
				   "boundary and source corrected fractional residual");
			expect(fixed.scale[0][2] == 16 && close(fixed.normalized.back()[2], .125),
				   "zero initial flux uses fixed c times initial energy");
			expect(fixed.scale[0][3] == 16, "roundoff-level initial flux uses the labeled fallback");
			expect(fixed.summary[2]["normalization_basis"] == "c * abs(initial_energy)",
				   "fallback denominator is recorded");
			atomic_text(p / "radiation-conservation.csv", original);
		}
		verify_cgs_log(p / "run.log");
        {
            const auto marshak = metadata("marshak");
            const auto path = tmp / "marshak";
            results(path, marshak);
            verify_cgs_log(path / "run.log", 1.);
            const auto values = read_norms(path, marshak);
            expect(values.size() == 5 && close(values.at("material_egas").at("L1"), 5e-4),
                "Marshak material and radiation full-volume norms use normalized units");
            expect(read_marshak_material(path, marshak).rows.size() == 16,
                "Marshak material slice has complete cell coverage");
            const auto original = read_text(path / "radiation-conservation.csv");
            atomic_text(path / "radiation-conservation.csv", original.substr(0, original.find('\n') + 1) +
                "0,8,0,0,0,0,0,0,0,0,0,0,0,0\n"
                "1,8,2,1,0,0,-3,-2,0,0,-1,-1,0,0\n");
            const auto budget = *read_conservation(path, marshak);
            expect(budget.summary[0]["normalization_basis"] == "bath_energy * volume" &&
                budget.summary[1]["normalization_basis"] == "c * bath_energy * volume" &&
                budget.scale.front() == budget.scale.back() && budget.scale.front()[0] == 32 &&
                budget.normalized.back()[0] == 0 && budget.normalized.back()[1] == 0,
                "cold Marshak conservation uses fixed prescribed bath scales");
            atomic_text(path / "radiation-conservation.csv", original);
        }
        for (const auto &name : {"gaussian_pulse", "marshak"}) {
            auto fld = metadata(name);
            fld.update({{"radiation_model", "fld"}, {"radiation_algorithm_version", 1},
                {"radiation_integrator", "backward_euler"}, {"radiation_flux_semantics", "derived_fld_flux"},
                {"reference_kind", "linear-diffusion-limit"}, {"reference_exact_for_solver", false},
                {"reference_model", std::string(name) == "marshak" ? "su-olson-linear-halfspace" : "periodic-constant-diffusion"}});
            const auto folder = tmp / (std::string("fld-") + name);
            results(folder, fld);
            expect(read_slice(folder, fld).columns.size() == 13, "FLD slices include analytic diffusion reference columns");
            if (std::string(name) == "marshak") {
                expect(read_marshak_material(folder, fld).columns.size() == 7, "FLD material slice includes its diffusion reference");
            }
            const auto diagnostics = read_fld_diagnostics(folder, fld);
            expect(diagnostics.at("accepted_steps") == 1 && diagnostics.at("maximum_nonlinear_residual") == 1.e-12,
                "FLD accepted solver diagnostics parsed");
            expect(diagnostics.at("reference_norm_scope") == "projected-positive-side-slice" &&
                close(diagnostics.at("slice_reference_norms").at("er").at("L1"), 2.5e-4) &&
                diagnostics.at("reference_exact_for_solver") == false,
                "FLD reference discrepancy norms have explicit slice scope and qualified exactness");
            if (std::string(name) == "marshak") {
                expect(close(diagnostics.at("slice_reference_norms").at("material_egas").at("L1"), 5.e-4),
                    "FLD material energy receives its own analytic reference discrepancy");
            }
            auto false_exactness = fld;
            false_exactness["reference_exact_for_solver"] = true;
            rejects([&] { read_fld_diagnostics(folder, false_exactness); },
                "linear diffusion reference cannot be marked exact for nonlinear FLD");
            const auto budget = *read_conservation(folder, fld);
            expect(budget.summary.size() == 1 && budget.summary.at(0).at("field") == "er" &&
                budget.normalized.back()[0] == 0, "FLD only assesses the radiation energy ledger");
            rejects([&] { read_norms(folder, fld); }, "FLD cannot reuse M1 analytic norms");
            const auto original = read_text(folder / "run.log");
            atomic_text(folder / "run.log", original + "RADIATION_FLD_FINISHED " +
                (std::string(name) == "marshak" ? "MARSHAK" : "RADIATION_GAUSSIAN_PULSE") +
                " t=" + number(fld.at("time")) + "\n");
            rejects([&] { read_fld_diagnostics(folder, fld); }, "duplicate FLD final marker rejected");
            auto nonconverged = original;
            nonconverged.replace(nonconverged.find("nonlinear_residual=1e-12"), std::string("nonlinear_residual=1e-12").size(), "nonlinear_residual=1e-2");
            atomic_text(folder / "run.log", nonconverged);
            rejects([&] { read_fld_diagnostics(folder, fld); }, "nonconverged FLD accepted step rejected");
            atomic_text(folder / "run.log", original);
            auto bad_request = original;
            bad_request.replace(bad_request.find(" requested="), std::string::npos,
                " requested=" + number(fld.at("time").get<double>() * 2) + "\n");
            atomic_text(folder / "run.log", bad_request);
            rejects([&] { read_fld_diagnostics(folder, fld); }, "mismatched FLD requested completion time rejected");
            auto short_marker = original;
            short_marker.erase(short_marker.find(" requested="));
            atomic_text(folder / "run.log", short_marker + "\n");
            expect(read_fld_diagnostics(folder, fld).at("accepted_steps") == 1,
                "short-form FLD completion marker remains readable");
            atomic_text(folder / "run.log", original);
            auto wrong = fld;
            wrong["time"] = fld.at("time").get<double>() * 2;
            rejects([&] { read_fld_diagnostics(folder, wrong); }, "incomplete FLD interval rejected");
            const auto text = read_text(folder / "radiation-slices/slice-000.csv");
            atomic_text(folder / "radiation-slices/slice-000.csv", text + text.substr(text.find('\n') + 1));
            rejects([&] { read_slice(folder, fld); }, "duplicate FLD slice cells rejected");
            atomic_text(folder / "radiation-slices/slice-000.csv", text);
            report_pages(folder);
            const auto report = read_text(folder / (std::string(name) + ".html"));
            expect(report.find("Analytic diffusion-limit reference") != std::string::npos &&
                report.find("Full-volume norms and orders") == std::string::npos &&
                report.find("Derived FLD flux") != std::string::npos,
                "FLD page labels qualified analytic reference and derived flux without M1 full-volume norms");
        }
		auto cap = cadence(4, 61, .4);
		expect(cap["steps_per_output_check"] == 5 && close(cap["hard_dt"], 4. / 300),
			   "capture output-check timestep cap");
		auto seq = frame_schedule({0, .25, 1}, 4, 2, .5);
		expect(seq == std::vector<size_t>({0, 0, 0, 1, 1, 1, 2, 2}),
			   "nonuniform physical-time schedule");
		rejects([] { frame_schedule({0, 0, 1}, 4, 2, .5); }, "duplicate time rejected");
		auto csv = read_text(p / "radiation-slices/slice-000.csv");
		atomic_text(p / "radiation-slices/slice-001.csv", csv);
		rejects([&] { read_slice(p, m); }, "duplicate cells rejected");
		fs::remove(p / "radiation-slices/slice-001.csv");
		auto log = read_text(p / "run.log");
		atomic_text(p / "run.log", log + "RADIATION_TEST_FINISHED RADIATION_STREAMING_WAVE t=4\n");
		rejects([&] { read_norms(p, m); }, "duplicate finish marker rejected");
		atomic_text(p / "run.log", log);
		auto wrong = m;
		wrong["time"] = 5.;
		rejects([&] { read_norms(p, wrong); }, "stale final time rejected");
		rejects([&] { read_conservation(p, wrong); }, "stale conservation history rejected");
		auto data = read_text(p / "radiation-conservation.csv");
		atomic_text(p / "radiation-conservation.csv", data + "nan,1\n");
		rejects([&] { read_conservation(p, m); }, "malformed budget rejected");
		atomic_text(p / "radiation-conservation.csv", data);
		silos(p, m);
		for (auto axis : {"x", "y", "z"}) {
			Options o;
			o.axis = axis;
			auto s = read_silo(p / "X.1.silo", o, m);
			expect(s.cells.size() == 16 && s.time == 1, "multiblock Silo slice/timestamp");
			for (auto &c : s.cells) {
				double value = 2.1 + (c.lo[0] + c.hi[0]) / 2 / 6e10 + 2 * (c.lo[1] + c.hi[1]) / 2 / 6e10 +
							   3 * (c.lo[2] + c.hi[2]) / 2 / 6e10;
				expect(close(c.value, value), "Silo orientation and strides");
			}
		}
		Options o;
		o.field = "fluxmag";
		auto s = read_silo(p / "X.0.silo", o, m);
		for (auto &c : s.cells) {
			double value = 2 + (c.lo[0] + c.hi[0]) / 2 / 6e10 + 2 * (c.lo[1] + c.hi[1]) / 2 / 6e10 +
						   3 * (c.lo[2] + c.hi[2]) / 2 / 6e10;
			expect(close(c.value, std::sqrt(29.) * value), "flux magnitude");
		}
		expect(numerical_silos(p).size() == 4, "numerical snapshot listing");
		Options sessions;
		sessions.session_dir = tmp / "flat sessions";
		auto first_session = make_session(p, sessions);
		auto xml = read_text(first_session.at("session_file").get<std::string>());
		expect(first_session.at("frames").size() == 3 &&
				   fs::path(first_session["frames"].back()["silo"].get<std::string>()).filename() ==
					   "final.silo",
			   "session deduplicates final state and prefers numerical final.silo");
		expect(first_session["frames"][0]["time"] == 0 && first_session["frames"][1]["time"] == 1 &&
				   first_session["frames"][2]["time"] == 4,
			   "session retains irregular recorded times");
		auto again = make_session(p, sessions);
		expect(again["session_file"] == first_session["session_file"], "stable session filename on repeat");
		sessions.position = 1e9;
		auto shifted = make_session(p, sessions);
		expect(shifted["session_file"] != first_session["session_file"],
			   "distinct slice positions cannot collide");
		expect(fs::path(shifted["session_file"].get<std::string>()).parent_path() == sessions.session_dir,
			   "all session files live in one flat directory");
		expect(xml.find("Pseudocolor_1.0") != std::string::npos && xml.find("Slice_1.0") != std::string::npos,
			   "session contains native VisIt plot and slice plugins");
        auto coupled_meta = metadata("subcritical_shock");
        const auto coupled_folder = tmp / "coupled";
        results(coupled_folder, coupled_meta);
        auto coupled_data = read_hydro_profile(coupled_folder, coupled_meta);
        expect(coupled_data.rows.size() == 16 && coupled_data.rows[0][2] < coupled_data.rows[15][2],
            "coupled slices sorted into complete plane without analytic data");
        auto wrong_time = coupled_meta;
        wrong_time["time"] = 5.;
        rejects([&] { read_hydro_profile(coupled_folder, wrong_time); }, "wrong hydro final time rejected");
        const auto hydro_csv = coupled_folder / "radiation-hydro-slices/slice-000.csv";
        auto hydro_text = read_text(hydro_csv);
        auto invalid_hydro = hydro_text;
        auto density_pos = invalid_hydro.find(",7.78e-10");
        invalid_hydro.replace(density_pos, 9, ",-7.78e-10");
        atomic_text(hydro_csv, invalid_hydro);
        rejects([&] { read_hydro_profile(coupled_folder, coupled_meta); }, "negative gas density rejected");
        atomic_text(hydro_csv, hydro_text);
        atomic_text(coupled_folder / "radiation-hydro-slices/duplicate.csv", hydro_text);
        rejects([&] { read_hydro_profile(coupled_folder, coupled_meta); }, "duplicate hydro cells rejected");
        fs::remove(coupled_folder / "radiation-hydro-slices/duplicate.csv");
        for (const auto &name : {"streaming_front", "subcritical_shock"}) {
            auto adaptive_meta = metadata(name, 2);
            adaptive_meta["adaptive"] = true;
            adaptive_meta["min_level"] = 1;
            auto adaptive_folder = tmp / (std::string(name) + "-adaptive");
            results(adaptive_folder, adaptive_meta);
            adaptive_results(adaptive_folder, adaptive_meta);
            const bool hydro = coupled_case(name);
            auto read = [&] { return hydro ? read_hydro_profile(adaptive_folder, adaptive_meta) : read_slice(adaptive_folder, adaptive_meta); };
            const auto projected = read();
            expect(projected.rows.size() == 256 && projected.rows[0][1] == 2 * adaptive_meta.at("dx").get<double>() &&
                projected.rows[0][5] == projected.rows[16][5] && projected.rows[0][2] != projected.rows[16][2],
                "mixed leaves project constant values while retaining native width");
            const auto summary = slice_mesh_summary(adaptive_folder, adaptive_meta);
            expect(summary.at("native_slice_cells") == 160 && summary["full_mesh"]["leaf_cells"] == 2304,
                "adaptive metadata distinguishes actual slice and full-volume cell counts");
            const auto directory = adaptive_folder / (hydro ? "radiation-hydro-slices" : "radiation-slices");
            const auto text = read_text(directory / "slice-mixed.csv");
            atomic_text(directory / "duplicate.csv", text);
            rejects(read, "overlapping AMR leaves rejected");
            fs::remove(directory / "duplicate.csv");
            auto nondyadic = text;
            const auto first_comma = nondyadic.find(',', nondyadic.find('\n') + 1);
            const auto second_comma = nondyadic.find(',', first_comma + 1);
            nondyadic.replace(first_comma + 1, second_comma - first_comma - 1, number(1.5 * adaptive_meta.at("dx").get<double>()));
            atomic_text(directory / "slice-mixed.csv", nondyadic);
            rejects(read, "non-dyadic AMR leaf spacing rejected");
            atomic_text(directory / "slice-mixed.csv", text.substr(0, text.find('\n', text.find('\n') + 1) + 1));
            rejects(read, "missing AMR footprint rejected");
            atomic_text(directory / "slice-mixed.csv", text);
            auto wrong = adaptive_meta;
            wrong["adaptive"] = false;
            rejects([&] { hydro ? read_hydro_profile(adaptive_folder, wrong) : read_slice(adaptive_folder, wrong); },
                "uniform reader remains strict about mixed leaf spacing");
        }
        silos(coupled_folder, coupled_meta);
        const auto coupled_session = make_session(coupled_folder, sessions);
        expect(coupled_session.at("frames").size() == 3,
            "coupled movie session accepts numerical profiles without analytic norms");
		auto gaussian_meta = m;
		gaussian_meta["case"] = "gaussian_pulse";
		results(p, gaussian_meta);
		sessions.position = 0;
		auto gaussian_session = make_session(p, sessions);
		auto gaussian_xml = read_text(gaussian_session["session_file"].get<std::string>());
		expect(gaussian_xml.find("max(er-(1),") != std::string::npos &&
				   gaussian_xml.find(">\"Log\"</Field>") != std::string::npos,
			   "Gaussian session clips energy excess and uses logarithmic scaling");
		expect(gaussian_session["color_limits"][0].get<double>() > 0, "Gaussian log floor is positive");
		sessions.view = "3d";
		sessions.field = "fluxmag";
		auto surface = make_session(p, sessions);
		auto surface_xml = read_text(surface["session_file"].get<std::string>());
		expect(surface_xml.find("sqrt(fx*fx+fy*fy+fz*fz)") != std::string::npos &&
				   surface_xml.find("Slice_1.0") == std::string::npos,
			   "3D magnitude session retains physical flux expression without slicing");
		write_json(p / "run.json", m);

		atomic_text(tmp / "hash.txt", "abc");
		expect(sha256(tmp / "hash.txt") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
			   "SHA256 compatible with Python");
		fs::remove_all(tmp);
		std::cout << checks << " checks passed\n";
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
