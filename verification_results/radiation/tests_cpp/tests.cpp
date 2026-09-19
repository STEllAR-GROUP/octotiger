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
	double length = 6e10;
	return {{"case", name},
			{"level", level},
			{"inx", 4},
			{"cells", n},
			{"length", length},
			{"dx", length / n},
			{"time", 4.},
			{"c", c_cgs},
			{"units", cgs_units},
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
		<< " |\nRADIATION_TEST_FINISHED RADIATION_" << name << " t=" << time << '\n';
	double error = 1e-3 / n;
	for (int f = 0; f < 4; ++f)
		log << fields[f] << ' ' << (f == 0 ? error : 0) << ' ' << (f == 0 ? 2 * error : 0) << ' '
			<< (f == 0 ? 4 * error : 0) << '\n';
	atomic_text(p / "run.log", log.str());
	for (int k = 0; k < 3; ++k)
		atomic_text(p / (norms[k] + ".dat"), number(dx) + " " + std::to_string(m.at("level").get<int>()) +
												 " " + number(error * (1 << k)) + " 0 0 0\n");
	std::ostringstream csv;
	csv << "t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref\n" << std::scientific << std::setprecision(17);
	for (int i = n - 1; i >= 0; --i)
		for (int j = n - 1; j >= 0; --j) {
			double x = -len / 2 + (i + .5) * dx, y = -len / 2 + (j + .5) * dx,
				   ref = 1 + .5 * std::cos(2 * 3.141592653589793 * x / len);
			csv << time << ',' << dx << ',' << x << ',' << y << ',' << dx / 2 << ',' << ref + error
				<< ",0,0,0," << ref << ",0,0,0\n";
		}
	atomic_text(p / "radiation-slices/slice-000.csv", csv.str());
	double volume = len * len * len;
	std::ostringstream budget;
	budget << "t,volume,er,fx,fy,fz,er_boundary,fx_boundary,fy_boundary,fz_boundary,er_source,fx_source,fy_"
			  "source,fz_source\n"
		   << std::scientific << std::setprecision(17);
	for (double t : {0., time / 4, time})
		budget << t << ',' << volume << ',' << volume * (1 + t) << ",0,0,0,0,0,0,0," << volume * t
			   << ",0,0,0\n";
	atomic_text(p / "radiation-conservation.csv", budget.str());
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
		for (int d = 0; d < 3; ++d)
			x[d].resize(mdims[d]);
		double time = entry.second;
		auto opt = DBMakeOptlist(4);
		DBAddOption(opt, DBOPT_DTIME, &time);
		int order = DB_ROWMAJOR;
		DBAddOption(opt, DBOPT_MAJORORDER, &order);
		for (int block = 0; block < 2; ++block) {
			auto name = "block" + std::to_string(block);
			DBMkDir(db, name.c_str());
			DBSetDir(db, name.c_str());
			for (int d = 0; d < 3; ++d)
				for (int i = 0; i < mdims[d]; ++i)
					x[d][i] = -half + (d == 0 ? block * half : 0) + i * length / n;
			double *coords[]{x[0].data(), x[1].data(), x[2].data()};
			const char *labels[]{"x", "y", "z"};
			require(DBPutQuadmesh(db, "mesh", labels, coords, mdims, 3, DB_DOUBLE, DB_COLLINEAR, opt) >= 0,
					"fixture mesh failed");
			for (int f = 0; f < 4; ++f) {
				std::vector<double> v(std::size_t(n) * n * n / 2);
				for (int k = 0; k < n; ++k)
					for (int j = 0; j < n; ++j)
						for (int i = 0; i < n / 2; ++i) {
							double px = (x[0][i] + x[0][i + 1]) / 2 / length,
								   py = (x[1][j] + x[1][j + 1]) / 2 / length,
								   pz = (x[2][k] + x[2][k + 1]) / 2 / length;
							v[i + (n / 2) * (j + n * k)] = (f + 1) * (2 + px + 2 * py + 3 * pz + time / 10);
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
        auto op = opacity_metadata({{"radiation.opacity.model", "grey"},
            {"radiation.opacity.absorption", "2"}, {"radiation.opacity.scattering", "3"},
            {"radiation.test.extinction", "5"}});
        expect(op.at("material_model")=="grey" && op.at("absorption")=="2" &&
            op.at("scattering")=="3" && op.at("transport_absorption")=="-1" &&
            op.at("prescribed_test_chi_code_inverse_length")=="5", "opacity metadata roles");
        expect(opacity_metadata({{"rad_opacity", "0.7"}}).at("legacy_constant")=="0.7",
            "legacy opacity metadata");

		if (argc > 1 && std::string(argv[1]) == "--fake-visit") {
			// Test double for the terminal/encoder regression, not a VisIt compatibility test.
			std::map<std::string, std::string> opts;
			for (int i = 2; i + 1 < argc; ++i) {
				std::string arg = argv[i];
				if (arg == "-sessionfile" || arg == "-output" || arg == "-end")
					opts[arg] = argv[++i];
			}
			require(read_text(opts.at("-sessionfile")).find("Pseudocolor_1.0") != std::string::npos,
					"Missing session argument");
			const char *mode_env = std::getenv("RR_TEST_VISIT_MODE");
			std::string mode = mode_env ? mode_env : "";
			int last = std::stoi(opts.at("-end"));
			for (int i = 0; i <= last; ++i) {
				if (mode == "partial" && i == last)
					continue;
				std::ostringstream out;
				out << opts.at("-output") << std::setfill('0') << std::setw(4) << i << ".png";
				if (mode == "corrupt")
					atomic_text(out.str(), "not a PNG");
				else
					fs::copy_file(std::getenv("RR_TEST_FRAME"), out.str());
			}
			if (mode == "error")
				std::cout << "VisIt: Error - test rendering error\n";
			std::cout << "VisIt completed generating frames.\n";
			return (mode == "shutdown" || mode == "partial" || mode == "corrupt") ? 250 : 0;
		}

		if (argc > 1 && std::string(argv[1]).starts_with("--config_file=")) {
			const auto p = fs::current_path();
			auto m = read_json(p / "run.json");
			results(p, m);
			std::ofstream count(p / "solver-executions", std::ios::app);
			count << "run\n";
			count.close();
			std::cout << read_text(p / "run.log");
			return 0;
		}
		if (argc == 3 && std::string(argv[1]) == "--fixtures") {
			auto dir = rr::absolute(argv[2]);
			for (auto &name : cases)
				for (int level = 0; level < 2; ++level) {
					auto p = dir / name / ("l" + std::to_string(level));
					auto m = metadata(name, level);
					results(p, m);
					if (name == cases[0] && level == 0)
						silos(p, m);
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
		auto cap = cadence(4, 61, .4);
		expect(cap["steps_per_output_check"] == 5 && close(cap["hard_dt"], 4. / 300),
			   "capture output-check timestep cap");
		auto seq = frame_schedule({0, .25, 1}, 4, 2, .5);
		expect(seq == std::vector<std::size_t>({0, 0, 0, 1, 1, 1, 2, 2}),
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
