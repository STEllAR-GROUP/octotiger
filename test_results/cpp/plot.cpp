#include "common.hpp"
#include <sys/wait.h>
namespace rr
{
void gnuplot_script(const fs::path &script, const std::string &body, const Options &o)
{
	atomic_text(script, body);
	auto gp = executable(o.gnuplot);
	// All paths are quoted for the shell; plot text lives in a separate file.
	std::string cmd = shell_quote(gp.string()) + " " + shell_quote(rr::absolute(script).string());
	check_interrupt();
	int status = std::system(cmd.c_str());
	check_interrupt();
	require(status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0,
			"gnuplot failed; inspect " + script.string());
}
void plot_pair(const fs::path &stem, const std::string &body, int w, int h, const Options &o)
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
		script += "set output " + gp_quote(tmp.string()) + "\n" + body + "\nunset output\n";
	}
	auto p = stem;
	p += ".gnuplot";
	gnuplot_script(p, script, o);
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
std::string image_tag(std::string s, std::string alt)
{
	return "<img loading=\"lazy\" src=\"" + url(s) + "\" alt=\"" + html(alt) + "\">";
}
std::string data_path(const fs::path &p)
{
	return gp_quote(p.string());
}
void render_run(const fs::path &folder, const Json &m, const fs::path &target,
				const std::optional<Budget> &budget, const Options &o)
{
	auto data = read_slice(folder, m);
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
	records_csv(target / "slice.csv", records, cols);
	atomic_text(target / "slice.dat", grid.str());
	atomic_text(target / "profiles.dat", row.str());
	Json units{{"t", cgs(m) ? "s" : "code time"}};
	for (auto s : {"dx", "x", "y", "z"})
		units[s] = cgs(m) ? "cm" : "code length";
	for (int f = 0; f < 4; ++f)
		for (auto suffix : {"", "_ref", "_error"})
			units[fields[f] + suffix] = cgs(m) ? (f == 0 ? "erg/cm^3" : "erg/(cm^2 s)") : "code units";
	write_json(target / "slice.units.json", units);
	std::string profiles = "set multiplot layout 4,2 rowsfirst title " +
						   gp_quote(title + " | y=z=dx/2=" + sci(dx / 2)) +
						   "\nset key top right\nset xlabel " + gp_quote(cgs(m) ? "x (cm)" : "x") +
						   "\nset autoscale\nset xrange [" + number(-length / 2) + ":" + number(length / 2) +
						   "]\nset xtics " + number(length / 4) + "\n";
	for (int f = 0; f < 4; ++f) {
		double bg = excess && f == 0 ? m.at("background").get<double>() : 0;
		bool logarithmic = m.at("case") == "gaussian_pulse" && f == 0;
		double peak = 0;
		for (auto &r : data.rows)
			peak = std::max({peak, r[5 + f] - bg, r[9 + f] - bg});
		double log_high = peak > 0 ? peak : std::max(std::abs(bg), 1.) * 1e-12;
		double floor = log_high * 1e-6;
		auto value = [&](int column) {
			std::string v = "(" + col(column) + "-" + number(bg) + ")";
			return logarithmic ? "(" + v + ">" + number(floor) + "?" + v + ":" + number(floor) + ")" : v;
		};
		auto label = field_label(m, f) + (bg ? " minus background" : "") +
					 (logarithmic ? " | log; floor=" + sci(floor) : "");
		if (logarithmic)
			write_json(target / "display.json",
					   {{"energy_scale", "log"},
						{"energy_background", bg},
						{"energy_floor", floor},
						{"energy_maximum", log_high},
						{"below_floor", "Values at or below the floor, including nonpositive excesses, use "
										"the lowest color; raw data unchanged"}});
		double profile_error = 0;
		for (int i = 0; i < n; ++i) {
			const auto &r = data.rows[std::size_t(i) * n + n / 2];
			profile_error = std::max(profile_error, std::abs(r[5 + f] - r[9 + f]));
		}
		double error_limit = profile_error > 0 ? 1.1 * profile_error : 1e-30;
		profiles +=
			"set autoscale y\n" +
			(logarithmic
				 ? "set logscale y\nset yrange [" + number(floor) + ":" + number(log_high * 1.1) + "]\n"
				 : "") +
			"set ylabel " +
			gp_quote(logarithmic ? "E - Ebg (" + std::string(cgs(m) ? "erg/cm^3" : "code units") + "), log"
								 : label) +
			"\n" +
			(logarithmic ? "set label 1 " + gp_quote("Display floor: " + sci(floor)) +
							   " at graph 0.03,0.07 left font ',9'\n"
						 : "") +
			"plot " + data_path(target / "profiles.dat") + " using 3:" + value(10 + f) +
			" with lines dt 2 lc rgb '#111111' title 'Reference', '' using 3:" + value(6 + f) +
			" with linespoints pt 7 ps 0.4 lc rgb '#2873b9' title 'Numerical'\nunset label 1\nunset logscale "
			"y\nset "
			"ylabel " +
			gp_quote(field_label(m, f) + " numerical - reference") + "\nset yrange [" + number(-error_limit) +
			":" + number(error_limit) + "]\nplot " + data_path(target / "profiles.dat") +
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
			hi = log_high;
		}
		err = std::max(err, 1e-30);
		std::string maps =
			"set multiplot layout 1,3 title " + gp_quote(title + " | " + label + " | z=dx/2=" + sci(dx / 2)) +
			"\nunset key\nunset grid\nset lmargin 12\nset rmargin 13\nset bmargin "
			"5\nset tmargin 4\nset size ratio -1\nset xrange [" +
			number(-length / 2) + ":" + number(length / 2) + "]\nset yrange [" + number(-length / 2) + ":" +
			number(length / 2) + "]\nset xlabel " + gp_quote(cgs(m) ? "x (cm)" : "x") + "\nset ylabel " +
			gp_quote(cgs(m) ? "y (cm)" : "y") + "\nset xtics " + number(length / 2) + "\nset ytics " +
			number(length / 2) + "\n";
		for (int k = 0; k < 3; ++k) {
			maps += logarithmic && k != 2 ? "set logscale cb\n" : "unset logscale cb\n";
			maps += k == 2 ? "set palette defined (0 '#2166ac', 0.5 '#f7f7f7', 1 '#b2182b')\n"
						   : "set palette defined (0 '#440154', 0.25 '#3b528b', 0.5 '#21918c', 0.75 "
							 "'#5ec962', 1 '#fde725')\n";
			maps +=
				"set cbrange [" + number(k == 2 ? -err : lo) + ":" + number(k == 2 ? err : hi) +
				"]\nset title " +
				gp_quote(std::array<std::string, 3>{"Numerical", "Reference", "Numerical - reference"}[k]) +
				"\nplot " + data_path(target / "slice.dat") +
				" using 3:4:" + (k == 2 ? "(" + col(14 + f) + ")" : value(k == 0 ? 6 + f : 10 + f)) +
				" with image\n";
		}
		maps += "unset multiplot\nset lmargin\nset rmargin\nset bmargin\nset tmargin\nset size "
				"noratio\nset autoscale\nset grid\n";
		plot_pair(target / ("slice_" + fields[f]), maps, 1800, 650, o);
	}
	profiles += "unset multiplot\n";
	plot_pair(target / "profiles", profiles, 1400, 1500, o);
	if (!budget) {
		for (auto name : {"conservation.png", "conservation.pdf", "conservation.csv",
						  "conservation-summary.csv", "conservation-summary.json", "conservation.units.json"})
			fs::remove(target / name);
		return;
	}
	auto &b = *budget;
	std::ostringstream history;
	history << std::scientific << std::setprecision(17);
	for (std::size_t i = 0; i < b.history.rows.size(); ++i) {
		for (double v : b.history.rows[i])
			history << v << ' ';
		for (double v : b.residual[i])
			history << v << ' ';
		history << '\n';
	}
	atomic_text(target / "conservation.dat", history.str());
	std::string plots = "set multiplot layout 4,2 rowsfirst title " +
						gp_quote(title + " | full-domain radiation budgets") +
						"\nset autoscale\nset grid\nset key top right\nset xlabel " +
						gp_quote(cgs(m) ? "Time (s)" : "Time") + "\n";
	for (int f = 0; f < 4; ++f) {
		auto q = data_path(target / "conservation.dat");
		auto unit = cgs(m) ? (f == 0 ? "erg" : "erg cm/s") : "code units";
		double lo = INFINITY, hi = -INFINITY, residual_max = 0;
		for (std::size_t i = 0; i < b.history.rows.size(); ++i) {
			const auto &r = b.history.rows[i];
			double expected = b.history.rows[0][2 + f] - r[6 + f] + r[10 + f];
			lo = std::min({lo, r[2 + f], expected});
			hi = std::max({hi, r[2 + f], expected});
			residual_max = std::max(residual_max, std::abs(b.residual[i][f]));
		}
		// Roundoff-level drift about a large conserved total defeats gnuplot autoscaling.
		// Show totals with finite headroom; retain small errors in the residual panel.
		double magnitude = std::max(std::abs(lo), std::abs(hi));
		double pad = magnitude > 0 ? std::max(.05 * (hi - lo), .05 * magnitude) : 1;
		double residual_limit = residual_max > 0 ? 1.1 * residual_max : 1;
		plots += "set yrange [" + number(lo - pad) + ":" + number(hi + pad) + "]\nset ylabel " +
				 gp_quote(fields[f] + " integral (" + unit + ")") + "\nplot " + q +
				 " using 1:" + std::to_string(3 + f) + " with lines title 'Measured total', '' using 1:(" +
				 number(b.history.rows[0][2 + f]) + "-" + col(7 + f) + "+" + col(11 + f) +
				 ") with lines dt 2 title 'Initial - boundary + source'\nset yrange [" +
				 number(-residual_limit) + ":" + number(residual_limit) + "]\nset ylabel " +
				 gp_quote(fields[f] + " residual (" + unit + ")") + "\nplot " + q +
				 " using 1:" + std::to_string(15 + f) + " with lines lc rgb '#b4422e' notitle\n";
	}
	plots += "unset multiplot\n";
	plot_pair(target / "conservation", plots, 1400, 1500, o);
	atomic_text(target / "conservation.csv", read_text(folder / "radiation-conservation.csv"));
	records_csv(target / "conservation-summary.csv", b.summary);
	write_json(target / "conservation-summary.json", b.summary);
	Json u{{"t", cgs(m) ? "s" : "code time"}, {"volume", cgs(m) ? "cm^3" : "code volume"}};
	for (int f = 0; f < 4; ++f)
		for (auto s : {"", "_boundary", "_source"})
			u[fields[f] + s] = cgs(m) ? (f == 0 ? "erg" : "erg cm/s") : "code units";
	write_json(target / "conservation.units.json", u);
}
} // namespace
fs::path render(const fs::path &batch, const Options &o)
{
	auto runs = completed_runs(batch);
	require(!runs.empty(), "No completed runs in " + batch.string());
	auto output = batch / "plots";
	fs::create_directories(output);
	std::map<std::string, std::vector<std::pair<fs::path, Json>>> groups;
	for (auto &[p, m] : runs) {
		m["norms"] = read_norms(p, m);
		groups[m.at("case")].emplace_back(p, m);
	}
	Json errors = Json::array(), budgets = Json::array();
	std::string page = page_start("Octo-TIGER radiation results") +
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
		auto case_dir = output / name;
		fs::create_directories(case_dir);
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
				fs::remove(case_dir / ("convergence." + std::string(ext)));
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
			atomic_text(case_dir / "convergence.dat", dat.str());
			std::string s = "set multiplot layout 2,2 title " +
							gp_quote(name + " | full-volume norms | t=" + number(first.at("time")) +
									 (cgs(first) ? " s" : "")) +
							"\nset logscale xy\nset xrange [" + number(first.at("dx")) + ":" +
							number(series.back().second.at("dx")) + "]\nset xlabel " +
							gp_quote(cgs(first) ? "dx (cm; finer ->)" : "dx (finer ->)") +
							"\nset key top left\n";
			for (int f = 0; f < 4; ++f) {
				bool any = false;
				for (auto &[p, m] : series)
					for (auto &n : norms)
						any |= m["norms"][fields[f]][n].get<double>() > 0;
				s += "set title " + gp_quote(field_label(first, f)) +
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
					s += data_path(case_dir / "convergence.dat") + " using 1:(" + col(c) + ">0?" + col(c) +
						 ":1/0) with linespoints pt " + std::to_string(5 + k) + " title " +
						 gp_quote(norms[k]);
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
			plot_pair(case_dir / "convergence", s, 1320, 1000, o);
			page += image_tag(name + "/convergence.png", "Convergence");
		}
		Json previous;
		page += "<div class=\"scroll\"><table><tr><th>N</th><th>E L1</th><th>E L2</th><th>E "
				"L∞</th><th>p(L1)</th><th>p(L2)</th><th>p(L∞)</th></tr>";
		std::string details;
		for (auto &[folder, m] : series) {
			auto target = case_dir / ("l" + std::to_string(m.at("level").get<int>()));
			auto b = read_conservation(folder, m);
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
				reuse = read_json(target / "plot-inputs.json") == signature;
			} catch (const std::exception &) {
			}
			for (auto stem : {"profiles", "slice_er", "slice_fx", "slice_fy", "slice_fz"})
				for (auto ext : {".png", ".pdf"})
					reuse &= fs::is_regular_file(target / (std::string(stem) + ext));
			reuse &= fs::is_regular_file(target / "slice.csv");
			if (b)
				for (auto ext : {".png", ".pdf", ".csv", "-summary.csv", "-summary.json", ".units.json"})
					reuse &= fs::is_regular_file(target / ("conservation" + std::string(ext)));
			if (!reuse) {
				render_run(folder, m, target, b, o);
				write_json(target / "plot-inputs.json", signature);
			} else if (!b)
				for (auto ext : {".png", ".pdf", ".csv", "-summary.csv", "-summary.json", ".units.json"})
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
					   image_tag(rel + "/slice_er.png", "Energy slice") +
					   image_tag(rel + "/profiles.png", "Profiles and errors");
			details += "<details><summary>Flux maps</summary>";
			for (int f = 1; f < 4; ++f)
				details += image_tag(rel + "/slice_" + fields[f] + ".png", fields[f]);
			details += "</details><h3>Radiation conservation</h3>";
			if (b) {
				details += "<p>R = Q(t) − Q(0) + B(t) − S(t). B is cumulative outward transport; S is the "
						   "signed source change. Flux integrals divided by c² give radiation momentum. "
						   "Energy normalization uses max(|E₀|,|E|,|B_E|,|S_E|); flux normalization also "
						   "includes c times that scale and the component's totals, boundary and source "
						   "terms.</p><div class=\"scroll\"><table><tr><th>Field</th><th>Final "
						   "residual</th><th>Final |R|/scale</th><th>Maximum |R|/scale</th></tr>";
				for (auto record : b->summary) {
					details += "<tr><td>" + record["field"].get<std::string>() + "</td><td>" +
							   sci(record["residual"]) + "</td><td>" + sci(record["normalized_error"]) +
							   "</td><td>" + sci(record["max_normalized_error"]) + "</td></tr>";
					record.update({{"case", name},
								   {"level", m.at("level")},
								   {"cells_per_side", m.at("cells")},
								   {"time", m.at("time")},
								   {"origin", m.at("origin")}});
					budgets.push_back(record);
				}
				details += "</table></div><p><a href=\"" + rel +
						   "/conservation-summary.csv\">Budget summary CSV</a> · <a href=\"" + rel +
						   "/conservation.csv\">History CSV</a></p>" +
						   image_tag(rel + "/conservation.png", "Conservation budgets");
			} else
				details += "<p>Conservation diagnostics unavailable for this run.</p>";
			details += "</details>";
			previous = m;
		}
		page += "</table></div>" + details + "</section>";
	}
	records_csv(output / "errors.csv", errors);
	write_json(output / "errors.json", errors);
	records_csv(output / "conservation.csv", budgets,
				{"case", "level", "cells_per_side", "time", "origin", "field", "initial", "final",
				 "raw_change", "boundary", "source", "residual", "normalization_scale", "normalized_error",
				 "max_abs_residual", "max_normalized_error", "integral_units"});
	write_json(output / "conservation.json", budgets);
	atomic_text(output / "index.html", page + "</body></html>\n");
	report_pages(batch);
	return output / "index.html";
}
void publish(const fs::path &output, Json &state)
{
	state["updated_utc"] = stamp();
	write_json(output / "live.json", state);
	report_pages(output, true);
}
} // namespace rr
