#include "common.hpp"
#include <sys/wait.h>
namespace rr
{
void gnuplotScript(const fs::path &script, const std::string &body, const Options &o)
{
	atomicText(script, body);
	auto gp = executable(o.gnuplot);
	// All paths are quoted for the shell; plot text lives in a separate file.
	std::string cmd = shellQuote(gp.string()) + " " + shellQuote(rr::absolute(script).string());
	checkInterrupt();
	int status = std::system(cmd.c_str());
	checkInterrupt();
	require(status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0,
			"gnuplot failed; inspect " + script.string());
}
void plotPair(const fs::path &stem, const std::string &body, int w, int h, const Options &o)
{
	std::string script = "set encoding utf8\nset datafile separator whitespace\nset border lc rgb "
						 "'#738092'\nset grid lc rgb "
						 "'#e5e9ef'\nset xtics autofreq\nset ytics autofreq\nset format x "
						 "'%.1e'\nset format y '%.1e'\nset "
						 "format cb '%.2e'\n";
	for (auto ext : {"png", "pdf"}) {
		auto target = stem;
		target += '.' + std::string(ext);
		auto tmp = target;
		tmp += ".tmp";
		script += (std::string(ext) == "png" ? "set terminal pngcairo size " + std::to_string(w) + "," +
												   std::to_string(h) + " font 'Sans,11' noenhanced\n"
											 : "set terminal pdfcairo size " + number(w / 110.) + "," +
												   number(h / 110.) + " font 'Sans,11' noenhanced\n");
		script += "set output " + gpQuote(tmp.string()) + "\n" + body + "\nunset output\n";
	}
	auto p = stem;
	p += ".gnuplot";
	gnuplotScript(p, script, o);
	for (auto ext : {"png", "pdf"}) {
		auto target = stem;
		target += '.' + std::string(ext);
		auto tmp = target;
		tmp += ".tmp";
		require(fs::is_regular_file(tmp) && fs::file_size(tmp) > 0, "gnuplot produced no " + target.string());
		fs::rename(tmp, target);
	}
}
namespace
{
std::string col(int n)
{
	return "$" + std::to_string(n);
}
std::string sci(double x)
{
	std::ostringstream o;
	o << std::scientific << std::setprecision(6) << x;
	return o.str();
}
std::string imageTag(std::string s, std::string alt)
{
	return "<img loading=\"lazy\" src=\"" + url(s) + "\" alt=\"" + html(alt) + "\">";
}
std::string dataPath(const fs::path &p)
{
	return gpQuote(p.string());
}
void renderRun(const fs::path &folder, const Json &m, const fs::path &target,
				const std::optional<Budget> &budget, const Options &o)
{
	auto data = readSlice(folder, m);
	int n = m.at("cells");
	double dx = m.at("dx"), length = m.at("length");
	fs::create_directories(target);
	bool excess = m.at("case") == "gaussian_pulse" || m.at("case") == "equilibrium_sphere";
	std::string title = m.at("case").get<std::string>() + " | " + std::to_string(n) +
						"^3 cells | t=" + number(m.at("time")) + (cgs(m) ? " s" : "");
	std::ostringstream grid, row;
	grid << std::scientific << std::setprecision(17);
	row << std::scientific << std::setprecision(17);
	Json records = Json::array();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			auto &r = data.rows[std::size_t(i) * n + j];
			Json rec;
			for (std::size_t k = 0; k < r.size(); ++k)
				rec[data.columns[k]] = r[k];
			for (int f = 0; f < 4; ++f)
				rec[fields[f] + "_error"] = r[5 + f] - r[9 + f];
			records.push_back(rec);
			auto emit = [&](std::ostream &s) {
				for (double x : r)
					s << x << ' ';
				for (int f = 0; f < 4; ++f)
					s << r[5 + f] - r[9 + f] << ' ';
				s << '\n';
			};
			emit(grid);
			if (j == n / 2)
				emit(row);
		}
		grid << '\n';
	}
	auto cols = data.columns;
	for (auto &f : fields)
		cols.push_back(f + "_error");
	recordsCsv(target / "slice.csv", records, cols);
	atomicText(target / "slice.dat", grid.str());
	atomicText(target / "profiles.dat", row.str());
	Json units{{"t", cgs(m) ? "s" : "code time"}};
	for (auto s : {"dx", "x", "y", "z"})
		units[s] = cgs(m) ? "cm" : "code length";
	for (int f = 0; f < 4; ++f)
		for (auto suffix : {"", "_ref", "_error"})
			units[fields[f] + suffix] = cgs(m) ? (f == 0 ? "erg/cm^3" : "erg/(cm^2 s)") : "code units";
	writeJson(target / "slice.units.json", units);
	std::string profiles = "set multiplot layout 4,2 rowsfirst title " +
						   gpQuote(title + " | y=z=dx/2=" + sci(dx / 2)) +
						   "\nset key top right\nset xlabel " + gpQuote(cgs(m) ? "x (cm)" : "x") +
						   "\nset autoscale\nset xrange [" + number(-length / 2) + ":" + number(length / 2) +
						   "]\nset xtics " + number(length / 4) + "\n";
	for (int f = 0; f < 4; ++f) {
		double bg = excess && f == 0 ? m.at("background").get<double>() : 0;
		bool logarithmic = m.at("case") == "gaussian_pulse" && f == 0;
		double peak = 0;
		for (auto &r : data.rows)
			peak = std::max({peak, r[5 + f] - bg, r[9 + f] - bg});
		double logHigh = peak > 0 ? peak : std::max(std::abs(bg), 1.) * 1e-12;
		double floor = logHigh * 1e-6;
		auto value = [&](int column) {
			std::string v = "(" + col(column) + "-" + number(bg) + ")";
			return logarithmic ? "(" + v + ">" + number(floor) + "?" + v + ":" + number(floor) + ")" : v;
		};
		auto label = fieldLabel(m, f) + (bg ? " minus background" : "") +
					 (logarithmic ? " | log; floor=" + sci(floor) : "");
		if (logarithmic)
			writeJson(target / "display.json",
					   {{"energy_scale", "log"},
						{"energy_background", bg},
						{"energy_floor", floor},
						{"energy_maximum", logHigh},
						{"below_floor", "Values at or below the floor, including nonpositive excesses, use "
										"the lowest color; raw data unchanged"}});
		double profileError = 0;
		for (int i = 0; i < n; ++i) {
			const auto &r = data.rows[std::size_t(i) * n + n / 2];
			profileError = std::max(profileError, std::abs(r[5 + f] - r[9 + f]));
		}
		double errorLimit = profileError > 0 ? 1.1 * profileError : 1e-30;
		profiles +=
			"set autoscale y\n" +
			(logarithmic
				 ? "set logscale y\nset yrange [" + number(floor) + ":" + number(logHigh * 1.1) + "]\n"
				 : "") +
			"set ylabel " +
			gpQuote(logarithmic ? "E - Ebg (" + std::string(cgs(m) ? "erg/cm^3" : "code units") + "), log"
								 : label) +
			"\n" +
			(logarithmic ? "set label 1 " + gpQuote("Display floor: " + sci(floor)) +
							   " at graph 0.03,0.07 left font ',9'\n"
						 : "") +
			"plot " + dataPath(target / "profiles.dat") + " using 3:" + value(10 + f) +
			" with lines dt 2 lc rgb '#111111' title 'Reference', '' using 3:" + value(6 + f) +
			" with linespoints pt 7 ps 0.4 lc rgb '#2873b9' title 'Numerical'\nunset label 1\nunset logscale "
			"y\nset "
			"ylabel " +
			gpQuote(fieldLabel(m, f) + " numerical - reference") + "\nset yrange [" + number(-errorLimit) +
			":" + number(errorLimit) + "]\nplot " + dataPath(target / "profiles.dat") +
			" using 3:" + std::to_string(14 + f) + " with lines lc rgb '#b4422e' notitle\n";
		double lo = INFINITY, hi = -INFINITY, err = 0;
		for (auto &r : data.rows) {
			lo = std::min({lo, r[5 + f] - bg, r[9 + f] - bg});
			hi = std::max({hi, r[5 + f] - bg, r[9 + f] - bg});
			err = std::max(err, std::abs(r[5 + f] - r[9 + f]));
		}
		if (hi == lo) {
			double pad = std::max(std::abs(lo) * 1e-8, 1e-30);
			lo -= pad;
			hi += pad;
		}
		if (logarithmic) {
			lo = floor;
			hi = logHigh;
		}
		err = std::max(err, 1e-30);
		std::string maps =
			"set multiplot layout 1,3 title " + gpQuote(title + " | " + label + " | z=dx/2=" + sci(dx / 2)) +
			"\nunset key\nunset grid\nset lmargin 12\nset rmargin 13\nset bmargin "
			"5\nset tmargin 4\nset size ratio -1\nset xrange [" +
			number(-length / 2) + ":" + number(length / 2) + "]\nset yrange [" + number(-length / 2) + ":" +
			number(length / 2) + "]\nset xlabel " + gpQuote(cgs(m) ? "x (cm)" : "x") + "\nset ylabel " +
			gpQuote(cgs(m) ? "y (cm)" : "y") + "\nset xtics " + number(length / 2) + "\nset ytics " +
			number(length / 2) + "\n";
		for (int k = 0; k < 3; ++k) {
			maps += logarithmic && k != 2 ? "set logscale cb\n" : "unset logscale cb\n";
			maps += k == 2 ? "set palette defined (0 '#2166ac', 0.5 '#f7f7f7', 1 '#b2182b')\n"
						   : "set palette defined (0 '#440154', 0.25 '#3b528b', 0.5 '#21918c', 0.75 "
							 "'#5ec962', 1 '#fde725')\n";
			maps +=
				"set cbrange [" + number(k == 2 ? -err : lo) + ":" + number(k == 2 ? err : hi) +
				"]\nset title " +
				gpQuote(std::array<std::string, 3>{"Numerical", "Reference", "Numerical - reference"}[k]) +
				"\nplot " + dataPath(target / "slice.dat") +
				" using 3:4:" + (k == 2 ? "(" + col(14 + f) + ")" : value(k == 0 ? 6 + f : 10 + f)) +
				" with image\n";
		}
		maps += "unset multiplot\nset lmargin\nset rmargin\nset bmargin\nset tmargin\nset size "
				"noratio\nset autoscale\nset grid\n";
		plotPair(target / ("slice_" + fields[f]), maps, 1800, 650, o);
	}
	profiles += "unset multiplot\n";
	plotPair(target / "profiles", profiles, 1400, 1500, o);
	if (!budget) {
		for (auto name : {"conservation.png", "conservation.pdf", "conservation.csv",
						  "conservation-summary.csv", "conservation-summary.json", "conservation.units.json",
						  "conservation-fractions.csv", "conservation-fractions.units.json"})
			fs::remove(target / name);
		return;
	}
	auto &b = *budget;
	Json fractions = Json::array();
	std::vector<std::string> fractionColumns{"t"};
	for (auto suffix :
		 {"_fraction", "_expected_fraction", "_residual_fraction", "_boundary_fraction", "_source_fraction"})
		for (auto &f : fields)
			fractionColumns.push_back(f + suffix);
	std::ostringstream history;
	history << std::scientific << std::setprecision(17);
	for (std::size_t i = 0; i < b.history.rows.size(); ++i) {
		const auto &r = b.history.rows[i];
		Json record{{"t", r[0]}};
		for (int f = 0; f < 4; ++f) {
			const double scale = b.scale[i][f];
			record[fields[f] + "_fraction"] = r[2 + f] / scale;
			record[fields[f] + "_expected_fraction"] =
				(b.history.rows[0][2 + f] - r[6 + f] + r[10 + f]) / scale;
			record[fields[f] + "_residual_fraction"] = b.normalized[i][f];
			record[fields[f] + "_boundary_fraction"] = r[6 + f] / scale;
			record[fields[f] + "_source_fraction"] = r[10 + f] / scale;
		}
		for (auto &column : fractionColumns) {
			double value = record.at(column);
			require(std::isfinite(value), "Nonfinite conservation fraction: " + column);
			history << value << ' ';
		}
		history << '\n';
		fractions.push_back(std::move(record));
	}
	atomicText(target / "conservation.dat", history.str());
	recordsCsv(target / "conservation-fractions.csv", fractions, fractionColumns);
	Json fractionUnits{{"t", cgs(m) ? "s" : "code time"}};
	for (std::size_t i = 1; i < fractionColumns.size(); ++i)
		fractionUnits[fractionColumns[i]] = "dimensionless";
	writeJson(target / "conservation-fractions.units.json", fractionUnits);
	std::string plots = "set multiplot layout 4,2 rowsfirst title " +
						gpQuote(title + " | fractions of fixed initial totals") +
						"\nset autoscale\nset grid\nset key top right\nset xlabel " +
						gpQuote(cgs(m) ? "Time (s)" : "Time") + "\n";
	for (int f = 0; f < 4; ++f) {
		auto q = dataPath(target / "conservation.dat");
		const bool fallback = b.summary[f].at("normalization_basis") == "c * abs(initial_energy)";
		auto denominator = fallback ? "(c |er(0)|)" : "|" + fields[f] + "(0)|";
		double lo = INFINITY, hi = -INFINITY, residualMax = 0;
		for (std::size_t i = 0; i < b.history.rows.size(); ++i) {
			double measured = fractions[i].at(fields[f] + "_fraction");
			double expected = fractions[i].at(fields[f] + "_expected_fraction");
			lo = std::min({lo, measured, expected});
			hi = std::max({hi, measured, expected});
			residualMax = std::max(residualMax, std::abs(b.normalized[i][f]));
		}
		// Roundoff-level drift about a large conserved total defeats gnuplot autoscaling.
		// Show totals with finite headroom; retain small errors in the residual panel.
		double magnitude = std::max(std::abs(lo), std::abs(hi));
		double pad = magnitude > 0 ? std::max(.05 * (hi - lo), .05 * magnitude) : 1;
		double residualLimit = residualMax > 0 ? 1.1 * residualMax : 1e-16;
		plots +=
			"set title " +
			gpQuote(fallback ? "Zero/negligible initial flux: fixed c E0 scale"
							  : "Fixed scale = " + denominator) +
			"\nset yrange [" + number(lo - pad) + ":" + number(hi + pad) + "]\nset ylabel " +
			gpQuote(fields[f] + " / " + denominator) + "\nplot " + q + " using 1:" + std::to_string(2 + f) +
			" with lines title 'Measured / scale', '' using 1:" + std::to_string(6 + f) +
			" with lines dt 2 title '(Initial - boundary + source) / scale'\nset yrange [" +
			number(-residualLimit) + ":" + number(residualLimit) + "]\nset ylabel " +
			gpQuote("R_" + fields[f] + " / " + denominator) + "\n" +
			(residualMax == 0 ? "set label 1 'All recorded residuals are zero' at graph 0.5,0.8 center\n"
							   : "") +
			"plot " + q + " using 1:" + std::to_string(10 + f) +
			" with lines lc rgb '#b4422e' notitle\nunset label 1\n";
	}
	plots += "unset multiplot\nunset title\n";
	plotPair(target / "conservation", plots, 1400, 1500, o);
	atomicText(target / "conservation.csv", readText(folder / "radiation-conservation.csv"));
	recordsCsv(target / "conservation-summary.csv", b.summary);
	writeJson(target / "conservation-summary.json", b.summary);
	Json u{{"t", cgs(m) ? "s" : "code time"}, {"volume", cgs(m) ? "cm^3" : "code volume"}};
	for (int f = 0; f < 4; ++f)
		for (auto s : {"", "_boundary", "_source"})
			u[fields[f] + s] = cgs(m) ? (f == 0 ? "erg" : "erg cm/s") : "code units";
	writeJson(target / "conservation.units.json", u);
}
} // namespace
fs::path render(const fs::path &batch, const Options &o)
{
	auto runs = completedRuns(batch);
	require(!runs.empty(), "No completed runs in " + batch.string());
	auto output = batch / "plots";
	fs::create_directories(output);
	std::map<std::string, std::vector<std::pair<fs::path, Json>>> groups;
	for (auto &[p, m] : runs) {
		m["norms"] = readNorms(p, m);
		groups[m.at("case")].emplace_back(p, m);
	}
	Json errors = Json::array(), budgets = Json::array();
	std::string page = pageStart("Octo-TIGER radiation results") +
					   "<p><a href=\"errors.csv\">All norms/orders CSV</a> · <a "
					   "href=\"conservation.csv\">Conservation "
					   "CSV</a> · <a href=\"../movies.html\">Movies</a></p><p>Norms use the full "
					   "3D domain: L1 = Σ|e|ΔV/V; "
					   "L2 = √(Σe²ΔV/V); L∞ = max|e|. Orders compare matching physical times and "
					   "configurations. Zero "
					   "errors have no reported order.</p>";
	for (auto &[name, series] : groups) {
		std::sort(series.begin(), series.end(), [](auto &a, auto &b) {
			return a.second.at("dx").template get<double>() > b.second.at("dx").template get<double>();
		});
		auto &first = series.front().second;
		std::set<double> seen;
		auto caseDir = output / name;
		fs::create_directories(caseDir);
		page +=
			"<section><h2>" + html(name) + "</h2><p>" +
			(name.starts_with("streaming")
				 ? std::string("Exact nonlinear M1 streaming reference.")
				 : (name == "gaussian_pulse"
						? std::string("Linearized telegraph-equation reference; finite-amplitude nonlinear "
									  "M1 can have a model error floor. Energy excess uses a logarithmic "
									  "display with a labeled floor; signed errors and fluxes remain linear.")
						: std::string("Steady diffusion-limit reference; finite-flux nonlinear M1 can have a "
									  "model error floor."))) +
			"</p>";
		for (auto &[p, m] : series) {
			for (auto key : {"time", "length", "c", "background", "comparison_signature", "executable_sha256",
							 "origin", "units"})
				require(m.value(key, Json()) == first.value(key, Json()),
						"Cannot mix different " + std::string(key) + " in convergence series");
			require(seen.insert(m.at("dx").get<double>()).second,
					"Duplicate resolution in convergence series");
		}
		if (series.size() < 2) {
			for (auto ext : {"png", "pdf"})
				fs::remove(caseDir / ("convergence." + std::string(ext)));
			page += "<p>Convergence becomes available after two resolutions are complete.</p>";
		} else {
			std::ostringstream dat;
			dat << std::scientific << std::setprecision(17);
			for (auto &[p, m] : series) {
				dat << m.at("dx").get<double>();
				for (auto &f : fields)
					for (auto &n : norms)
						dat << ' ' << m["norms"][f][n].get<double>();
				dat << '\n';
			}
			atomicText(caseDir / "convergence.dat", dat.str());
			std::string s = "set multiplot layout 2,2 title " +
							gpQuote(name + " | full-volume norms | t=" + number(first.at("time")) +
									 (cgs(first) ? " s" : "")) +
							"\nset logscale xy\nset xrange [" + number(first.at("dx")) + ":" +
							number(series.back().second.at("dx")) + "]\nset xlabel " +
							gpQuote(cgs(first) ? "dx (cm; finer ->)" : "dx (finer ->)") +
							"\nset key top left\n";
			for (int f = 0; f < 4; ++f) {
				bool any = false;
				for (auto &[p, m] : series)
					for (auto &n : norms)
						any |= m["norms"][fields[f]][n].get<double>() > 0;
				s += "set title " + gpQuote(fieldLabel(first, f)) +
					 "\nset ylabel 'Volume-normalized error'\n";
				if (!any) {
					s += "unset logscale y\nset yrange [-1:1]\nset label 1 'All norms are zero' at "
						 "graph "
						 "0.5,0.5 center\nplot 0 notitle lc rgb '#ffffff'\nunset label 1\nset "
						 "logscale "
						 "y\nset autoscale y\n";
					continue;
				}
				s += "set autoscale y\nplot ";
				for (int k = 0; k < 3; ++k) {
					if (k)
						s += ", ";
					int c = 2 + 3 * f + k;
					s += dataPath(caseDir / "convergence.dat") + " using 1:(" + col(c) + ">0?" + col(c) +
						 ":1/0) with linespoints pt " + std::to_string(5 + k) + " title " +
						 gpQuote(norms[k]);
				}
				double base = first["norms"][fields[f]]["L1"];
				if (base > 0)
					for (int power = 1; power <= 2; ++power)
						s += ", " + number(base) + "*(x/" + number(first.at("dx")) + ")**" +
							 std::to_string(power) + " with lines dt " + std::to_string(power + 1) +
							 " lc rgb '#999999' title 'dx^" + std::to_string(power) + " guide'";
				s += '\n';
			}
			s += "unset multiplot\nunset logscale\nset autoscale\n";
			plotPair(caseDir / "convergence", s, 1320, 1000, o);
			page += imageTag(name + "/convergence.png", "Convergence");
		}
		Json previous;
		page += "<div class=\"scroll\"><table><tr><th>N</th><th>E L1</th><th>E L2</th><th>E "
				"L∞</th><th>p(L1)</th><th>p(L2)</th><th>p(L∞)</th></tr>";
		std::string details;
		for (auto &[folder, m] : series) {
			auto target = caseDir / ("l" + std::to_string(m.at("level").get<int>()));
			auto b = readConservation(folder, m);
			Json signature{{"meta", m},
						   {"renderer_sha256", sha256("/proc/self/exe")},
						   {"gnuplot", executable(o.gnuplot).string()},
						   {"conservation", b ? Json(sha256(folder / "radiation-conservation.csv")) : Json()},
						   {"slices", Json::object()}};
			for (auto &e : fs::directory_iterator(folder / "radiation-slices"))
				if (e.path().extension() == ".csv")
					signature["slices"][e.path().filename().string()] = sha256(e.path());
			bool reuse = false;
			try {
				reuse = readJson(target / "plot-inputs.json") == signature;
			} catch (const std::exception &) {
			}
			for (auto stem : {"profiles", "slice_er", "slice_fx", "slice_fy", "slice_fz"})
				for (auto ext : {".png", ".pdf"})
					reuse &= fs::is_regular_file(target / (std::string(stem) + ext));
			reuse &= fs::is_regular_file(target / "slice.csv");
			if (b)
				for (auto ext : {".png", ".pdf", ".csv", "-summary.csv", "-summary.json", ".units.json",
								 "-fractions.csv", "-fractions.units.json"})
					reuse &= fs::is_regular_file(target / ("conservation" + std::string(ext)));
			if (!reuse) {
				renderRun(folder, m, target, b, o);
				writeJson(target / "plot-inputs.json", signature);
			} else if (!b)
				for (auto ext : {".png", ".pdf", ".csv", "-summary.csv", "-summary.json", ".units.json",
								 "-fractions.csv", "-fractions.units.json"})
					fs::remove(target / ("conservation" + std::string(ext)));
			std::array<Json, 3> rates;
			for (int f = 0; f < 4; ++f)
				for (int k = 0; k < 3; ++k) {
					auto n = norms[k];
					double error = m["norms"][fields[f]][n];
					Json order;
					if (!previous.is_null() && error > 0 && previous["norms"][fields[f]][n].get<double>() > 0)
						order = std::log(previous["norms"][fields[f]][n].get<double>() / error) /
								std::log(previous.at("dx").get<double>() / m.at("dx").get<double>());
					if (f == 0)
						rates[k] = order;
					errors.push_back(
						{{"case", name},
						 {"level", m.at("level")},
						 {"cells_per_side", m.at("cells")},
						 {"dx", m.at("dx")},
						 {"time", m.at("time")},
						 {"field", fields[f]},
						 {"norm", n},
						 {"error", error},
						 {"order", order},
						 {"origin", m.at("origin")},
						 {"dx_units", cgs(m) ? "cm" : "code length"},
						 {"time_units", cgs(m) ? "s" : "code time"},
						 {"error_units", cgs(m) ? (f == 0 ? "erg/cm^3" : "erg/(cm^2 s)") : "code units"}});
				}
			page += "<tr><td>" + std::to_string(m.at("cells").get<int>()) + "</td>";
			for (auto &n : norms)
				page += "<td>" + sci(m["norms"]["er"][n]) + "</td>";
			for (auto &v : rates)
				page += "<td>" + (v.is_null() ? "—" : sci(v)) + "</td>";
			page += "</tr>";
			std::string rel = name + "/l" + std::to_string(m.at("level").get<int>());
			details += "<details open><summary>" + std::to_string(m.at("cells").get<int>()) +
					   "³ cells</summary><p>Central cell layer at z=dx/2. <a href=\"" + rel +
					   "/slice.csv\">Paired values CSV</a> · <a href=\"" + rel +
					   "/profiles.pdf\">Profiles PDF</a></p>" +
					   imageTag(rel + "/slice_er.png", "Energy slice") +
					   imageTag(rel + "/profiles.png", "Profiles and errors");
			details += "<details><summary>Flux maps</summary>";
			for (int f = 1; f < 4; ++f)
				details += imageTag(rel + "/slice_" + fields[f] + ".png", fields[f]);
			details += "</details><h3>Radiation conservation</h3>";
			if (b) {
				details +=
					"<p>R = Q(t) − Q(0) + B(t) − S(t). B is cumulative outward transport; S is the "
					"signed source change. Flux integrals divided by c² give radiation momentum. "
					"All plotted totals and residuals are fractions of the fixed initial magnitude |Q(0)|. "
					"Zero/negligible initial flux (|Q(0)| ≤ 64ε c|E₀|) uses the labeled fallback c|E₀|. "
					"Neither denominator changes during the run.</p><div "
					"class=\"scroll\"><table><tr><th>Field</th><th>Scale</th><th>Final "
					"R/scale</th><th>Final |R|/scale</th><th>Maximum |R|/scale</th></tr>";
				for (auto record : b->summary) {
					details += "<tr><td>" + record["field"].get<std::string>() + "</td><td>" +
							   html(record.at("normalization_basis")) + "</td><td>" +
							   sci(record["normalized_residual"]) + "</td><td>" +
							   sci(record["normalized_error"]) + "</td><td>" +
							   sci(record["max_normalized_error"]) + "</td></tr>";
					record.update({{"case", name},
								   {"level", m.at("level")},
								   {"cells_per_side", m.at("cells")},
								   {"time", m.at("time")},
								   {"origin", m.at("origin")}});
					budgets.push_back(record);
				}
				details += "</table></div><p><a href=\"" + rel +
						   "/conservation-summary.csv\">Budget summary CSV</a> · <a href=\"" + rel +
						   "/conservation.csv\">Raw history CSV</a> · <a href=\"" + rel +
						   "/conservation-fractions.csv\">Fraction history CSV</a></p>" +
						   imageTag(rel + "/conservation.png", "Conservation budgets");
			} else
				details += "<p>Conservation diagnostics unavailable for this run.</p>";
			details += "</details>";
			previous = m;
		}
		page += "</table></div>" + details + "</section>";
	}
	recordsCsv(output / "errors.csv", errors);
	writeJson(output / "errors.json", errors);
	recordsCsv(output / "conservation.csv", budgets,
				{"case",
				 "level",
				 "cells_per_side",
				 "time",
				 "origin",
				 "field",
				 "initial",
				 "final",
				 "raw_change",
				 "boundary",
				 "source",
				 "residual",
				 "normalization_scale",
				 "normalized_error",
				 "normalization_basis",
				 "initial_fraction",
				 "final_fraction",
				 "boundary_fraction",
				 "source_fraction",
				 "normalized_residual",
				 "max_abs_residual",
				 "max_normalized_error",
				 "integral_units"});
	writeJson(output / "conservation.json", budgets);
	atomicText(output / "index.html", page + "</body></html>\n");
	reportPages(batch);
	return output / "index.html";
}
void publish(const fs::path &output, Json &state)
{
	state["updated_utc"] = stamp();
	writeJson(output / "live.json", state);
	reportPages(output, true);
}
} // namespace rr
