#include "problem_descriptions.hpp"
#include <tuple>
namespace rr
{
namespace
{
using Run = std::pair<fs::path, Json>;
std::string show(const Json &v)
{
	if (v.is_null())
		return "—";
	if (v.is_string())
		return html(v.get<std::string>());
	if (v.is_number_float()) {
		std::ostringstream s;
		s << std::scientific << std::setprecision(6) << v.get<double>();
		return s.str();
	}
	return html(v.dump());
}
std::string start(const std::string &title)
{
	return R"(<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>)" +
		   html(title) + R"( · Octo-TIGER</title><style>
:root{color-scheme:light;--ink:#172b35;--muted:#536772;--line:#d7e1e5;--accent:#076e76;--paper:#fff;--wash:#f1f5f5}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:20px}body{margin:0;color:var(--ink);background:var(--wash);font:16px/1.65 system-ui,-apple-system,sans-serif}a{color:var(--accent);text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #c48012;outline-offset:5px}header,main,footer{max-width:1240px;margin:auto;padding:26px 32px}header{padding-top:30px;padding-bottom:8px}.brand{font-size:12px;letter-spacing:.16em;font-weight:750;text-transform:uppercase;color:var(--accent)}h1,h2,h3,h4{line-height:1.2;letter-spacing:-.025em}h1{font-size:clamp(30px,4vw,48px);margin:18px 0 16px}h2{font-size:27px;margin:0 0 18px}h3{font-size:21px}h4{font-size:17px}.lead{font-size:19px;max-width:840px;color:var(--muted)}p{margin:12px 0 20px}.muted,figcaption{color:var(--muted);font-size:14px}.status{display:flex;flex-wrap:wrap;align-items:center;gap:8px 20px;border-block:1px solid var(--line);padding:12px 0;margin-top:22px;font-size:14px}.tag{display:inline-block;font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:750;color:var(--accent)}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}.card-content{padding:24px;display:flex;flex-direction:column;flex:1}.card h2{font-size:27px;margin:8px 0 12px}.card p{color:var(--muted);margin:0 0 15px}.card .cta{font-weight:700;margin-top:auto}.preview{background:#fff;border-bottom:1px solid var(--line);padding:10px 8px 5px;margin:0}.preview img{display:block;width:100%;aspect-ratio:1800/650;object-fit:contain}.preview figcaption{font-size:12px;padding:4px 10px}.placeholder{min-height:170px;display:flex;justify-content:center;align-items:center;background:linear-gradient(145deg,#e4eded,#f6f8f9);color:var(--muted);font-size:14px;border-bottom:1px solid var(--line)}.section{background:var(--paper);padding:32px;border:1px solid var(--line);border-radius:12px;margin:0 0 24px}.section>p,.section>ul{max-width:1000px}.equation{font:19px/1.9 Georgia,serif;overflow-x:auto;padding:16px 20px;background:#f2f6f6;border-left:3px solid var(--accent);margin:20px 0;white-space:normal}nav{display:flex;gap:8px 20px;flex-wrap:wrap;font-size:14px;margin:12px 0}nav a[aria-current=page]{font-weight:750;color:var(--ink)}.contents{padding:16px 0;margin:8px 0 24px;border-block:1px solid var(--line)}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f2f6f6;font-weight:650}td{white-space:nowrap}td.wrap{white-space:normal;overflow-wrap:anywhere}code{font-size:.88em;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;padding:16px;background:#f2f6f6}figure{margin:24px 0}figure img{display:block;width:100%;height:auto}figcaption{margin-top:8px}video{display:block;width:100%;max-height:720px;background:#0c1820}.downloads{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:14px;margin:18px 0}.note{border-left:3px solid #c28a31;padding:12px 18px;background:#fff9ed;margin:20px 0}.error{color:#922727;white-space:pre-wrap}.run{border-top:2px solid var(--line);padding-top:22px;margin-top:36px}.run:first-of-type{margin-top:10px}details{margin:18px 0}summary{cursor:pointer;font-weight:650}footer{font-size:13px;color:var(--muted);padding-bottom:38px}.refresh{font-size:13px;display:block;margin:12px 0}.setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.setup-grid div{border-bottom:1px solid var(--line);padding:12px 0}.setup-grid dt{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.setup-grid dd{margin:4px 0 0;font-weight:600;overflow-wrap:anywhere}.small{font-size:13px}
@media(max-width:700px){header,main,footer{padding:20px 16px}.cards{grid-template-columns:1fr}.card-content,.section{padding:22px}.section{border-radius:8px}h2{font-size:24px}.equation{font-size:17px;padding:12px}.setup-grid{grid-template-columns:1fr}.status{align-items:flex-start}.contents{gap:10px 16px}}@media print{body{background:#fff}.section,.card{break-inside:avoid}video,.refresh{display:none}header,main,footer{max-width:none}.cards{grid-template-columns:1fr 1fr}a{color:inherit}}
</style></head><body>)";
}
std::string finish(const Json &state)
{
	std::string s = "</main><footer>Octo-TIGER · Radiation verification · Descriptions reviewed against the "
					"September 17, 2026 source. "
					"Recorded run settings and provenance accompany the results.";
	if (state.value("active", false))
		s +=
			R"(<label class="refresh"><input id="refresh" type="checkbox" checked> Refresh every 10 seconds; paused during movie playback</label><script>try{window.scrollTo(0,Number(sessionStorage.getItem('scroll:'+location.pathname)||0));}catch(e){}setInterval(()=>{if(!document.getElementById('refresh').checked||Array.from(document.querySelectorAll('video')).some(v=>!v.paused&&!v.ended))return;try{sessionStorage.setItem('scroll:'+location.pathname,String(window.scrollY));}catch(e){}location.reload();},10000);</script>)";
	return s + "</footer></body></html>\n";
}
std::string link(const fs::path &batch, const fs::path &p, const std::string &label)
{
	if (!fs::is_regular_file(p))
		return "";
	return "<a href=\"" + url(fs::relative(p, batch).generic_string()) + "\">" + html(label) + "</a>";
}
std::string figure(const fs::path &batch, const fs::path &stem, const std::string &caption)
{
	auto png = fs::path(stem.string() + ".png");
	if (!fs::is_regular_file(png))
		return "";
	auto rel = url(fs::relative(png, batch).generic_string());
	return "<figure><a href=\"" + rel + "\"><img loading=\"lazy\" src=\"" + rel + "\" alt=\"" +
		   html(caption) + "\"></a><figcaption>" + html(caption) + " · " +
		   link(batch, fs::path(stem.string() + ".pdf"), "PDF") + "</figcaption></figure>";
}
bool comparable(const std::vector<Run> &runs)
{
	if (runs.size() < 2)
		return false;
	std::set<double> dx;
	for (auto &[p, m] : runs) {
		if (!dx.insert(m.at("dx").get<double>()).second)
			return false;
		for (auto key : {"time", "length", "c", "background", "comparison_signature", "executable_sha256",
						 "origin", "units"})
			if (m.value(key, Json()) != runs.front().second.value(key, Json()))
				return false;
	}
	return true;
}
std::string plansTable(const Json &state, const std::string &name)
{
	std::string rows;
	for (auto &r : state.value("runs", Json::array()))
		if (r.value("case", "") == name)
			rows += "<tr><td>" + show(r.value("level", Json())) + "</td><td>" +
					show(r.value("cells", Json())) + "³</td><td class=\"wrap\">" +
					show(r.value("stage", Json())) + "</td></tr>";
	if (rows.empty())
		return "";
	return "<div class=\"scroll\"><table><thead><tr><th>Level</th><th>Grid</th><th>Batch "
		   "status</th></tr></thead><tbody>" +
		   rows + "</tbody></table></div>";
}
std::string settings(const Json &m)
{
	auto cfg = m.value("config", Json::object());
	const std::string lu = cgs(m) ? " cm" : " (code length)", tu = cgs(m) ? " s" : " (code time)";
	auto row = [](const std::string &name, const std::string &v) {
		return "<tr><th scope=\"row\">" + name + "</th><td class=\"wrap\">" + v + "</td></tr>";
	};
	std::string s = "<div class=\"scroll\"><table><tbody>";
	s += row("Origin", show(m.value("origin", Json())));
	s += row("Domain", "Cube of side " + show(m.at("length")) + lu + "; centered at the origin");
	s += row("Resolution", show(m.at("cells")) + "³ cells; level " + show(m.at("level")) +
							   "; Δx = " + show(m.at("dx")) + lu);
	s += row("Final comparison time", show(m.at("time")) + tu);
	s += row("Light speed", show(m.at("c")) + (cgs(m) ? " cm/s" : " (code speed)"));
	for (const auto &[key, title] :
		 std::vector<std::pair<std::string, std::string>>{{"mesh.boundary.periodic", "Periodic boundaries"},
														  {"hydro.enabled", "Hydrodynamics"},
														  {"gravity.enabled", "Gravity"},
														  {"radiation.implicit", "Implicit radiation source"},
														  {"mesh.unigrid", "Uniform grid"},
														  {"hydro.cfl", "CFL"},
														  {"timestep.fixed", "Timestep cap"},
														  {"output.interval", "Output interval"},
														  {"rad_reconstruction", "Reconstruction"}}) {
		if (cfg.contains(key)) {
			s += row(title, show(cfg.at(key)) +
							((key == "timestep.fixed" || key == "output.interval") ? tu : ""));
		}
	}
	if (!m.at("case").get<std::string>().starts_with("streaming")) {
		for (const auto &[key, title, unit] : std::vector<std::tuple<std::string, std::string, std::string>>{
				 {"radiation.test.background", "Background energy density", "erg/cm³"},
				 {"radiation.test.extinction", "Scattering coefficient χ", "cm⁻¹"},
				 {"radiation.test.width", "Gaussian width w", "cm"},
				 {m.at("case") == "gaussian_pulse" ? "radiation.test.amplitude" :
																	 "radiation.test.luminosity",
				  m.at("case") == "gaussian_pulse" ? "Pulse amplitude A" : "All-space luminosity ℒ",
				  m.at("case") == "gaussian_pulse" ? "erg/cm³" : "erg/s"}}) {
			s += row(title, cfg.contains(key) ? show(cfg.at(key)) + " " + (cgs(m) ? unit : "(code units)")
											  : "Not recorded");
		}
	}
    s += row("Opacity model and coefficients", m.contains("opacity") ? html(m.at("opacity").dump()) : "Not recorded (older run)");
	if (cfg.empty())
		s += row("Effective configuration", "Not recorded in this run's metadata");
	return s + "</tbody></table></div>";
}
std::string normsTable(const std::vector<Run> &runs, bool compare, Json &errors)
{
	std::string s = "<div "
					"class=\"scroll\"><table><thead><tr><th>Grid</th><th>Field</th><th>Norm</th><th>Error</"
					"th><th>Order</th><th>Error units</th></tr></thead><tbody>";
	Json previous;
	for (auto &[p, m] : runs) {
		for (int f = 0; f < 4; ++f)
			for (auto &n : norms) {
				const double error = m.at("norms").at(fields[f]).at(n);
				Json order;
				if (compare && !previous.is_null() && error > 0 &&
					previous["norms"][fields[f]][n].get<double>() > 0)
					order = std::log(previous["norms"][fields[f]][n].get<double>() / error) /
							std::log(previous.at("dx").get<double>() / m.at("dx").get<double>());
				const std::string units = cgs(m) ? (f == 0 ? "erg/cm³" : "erg/(cm² s)") : "code units";
				errors.push_back({{"case", m.at("case")},
								  {"level", m.at("level")},
								  {"cells_per_side", m.at("cells")},
								  {"dx", m.at("dx")},
								  {"time", m.at("time")},
								  {"field", fields[f]},
								  {"norm", n},
								  {"error", error},
								  {"order", order},
								  {"error_units", units},
								  {"origin", m.value("origin", "")}});
				s += "<tr><td>" + show(m.at("cells")) + "³</td><td>" + fields[f] + "</td><td>" + n +
					 "</td><td>" + show(error) + "</td><td>" + show(order) + "</td><td>" + units +
					 "</td></tr>";
			}
		previous = m;
	}
	return s + "</tbody></table></div>";
}
std::string movies(const fs::path &batch, const fs::path &folder)
{
	if (!fs::is_directory(folder / "movies"))
		return "<p class=\"muted\">No completed movie is available for this resolution.</p>";
	std::vector<fs::path> manifests;
	for (auto &e : fs::recursive_directory_iterator(folder / "movies"))
		if (e.is_regular_file() && e.path().filename() == "movie.json")
			manifests.push_back(e.path());
	std::sort(manifests.begin(), manifests.end());
	std::string s;
	for (auto &p : manifests) {
		auto m = readJson(p);
		auto file = p.parent_path() / "movie.mp4";
		if (m.value("status", "") != "complete" || !fs::is_regular_file(file))
			continue;
		auto rel = url(fs::relative(file, batch).generic_string());
		s += "<figure><video controls preload=\"none\" src=\"" + rel + "\"></video><figcaption>" +
			 show(m.at("field")) + " · " +
			 (m.value("view", "slice") == "3d" ? "exterior surface"
											   : "slice normal to " + show(m.value("axis", "z"))) +
			 " · " + show(m.at("source_snapshots")) + " recorded snapshots · " + show(m.at("seconds")) +
			 " s playback";
		if (m.value("color_scale", "linear") == "log")
			s += ". Logarithmic energy excess E − Ebg; display floor " + show(m.at("color_limits")[0]) + " " +
				 show(m.at("field_units")) + ". Values at or below the floor use the lowest color.";
		else if (m.value("case", "") == "gaussian_pulse" && m.value("field", "") == "er")
			s += ". Existing linear-scale movie; regenerate movies to apply the logarithmic energy-excess "
				 "display.";
		s += ". " + link(batch, file, "MP4") + " · " + link(batch, p, "Movie metadata") +
			 "</figcaption></figure>";
	}
	return s.empty() ? "<p class=\"muted\">No completed movie is available for this resolution.</p>" : s;
}
std::string budgetTable(const Budget &b)
{
	std::string s = "<p>All values below are dimensionless fractions of the fixed initial scale.</p>"
					"<div "
					"class=\"scroll\"><table><thead><tr><th>Field</th><th>Scale</th><th>Q(0)/scale</"
					"th><th>Q(t)/scale</th><th>Boundary "
					"B/scale</th><th>Source S/scale</th><th>Residual R/scale</th><th>Maximum "
					"|R|/scale</th></tr></thead><tbody>";
	for (auto &r : b.summary) {
		s += "<tr><th scope=\"row\">" + show(r.at("field")) + "</th><td>" +
			 (r.at("normalization_basis") == "c * abs(initial_energy)"
				  ? "c |E₀| (zero/negligible initial flux)"
				  : "|Q(0)|") +
			 "</td>";
		for (auto key : {"initial_fraction", "final_fraction", "boundary_fraction", "source_fraction",
						 "normalized_residual", "max_normalized_error"})
			s += "<td>" + show(r.at(key)) + "</td>";
		s += "</tr>";
	}
	return s + "</tbody></table></div>";
}
std::string methodNotes()
{
	return R"(<section class="section" id="diagnostics"><span class="tag">Reading the results</span><h2>Errors, budgets, and movies</h2>
<p>Every norm uses the full three-dimensional domain, with cell-volume weighting. Here <var>e</var> = numerical − reference and <var>V</var> is the domain volume:</p>
<div class="equation">L1 = Σ |e| ΔV / V, &nbsp; L2 = √(Σ e² ΔV / V), &nbsp; L∞ = max |e|<br>p = ln(error<sub>coarse</sub> / error<sub>fine</sub>) / ln(Δx<sub>coarse</sub> / Δx<sub>fine</sub>)</div>
<p>Orders require two completed runs with matching physical time, configuration, units, executable, and origin. An exact zero error has no logarithmic order. The central slice and line profiles are visual diagnostics: the report slice is the cell layer at <var>z</var> = Δ<var>x</var>/2, and its line profile is at <var>y</var> = <var>z</var> = Δ<var>x</var>/2.</p>
<p>Conservation is assessed using the measured domain integrals and the signed cumulative boundary and source budgets:</p>
<div class="equation">R(t) = Q(t) − Q(0) + B(t) − S(t)</div>
<p><var>B</var> is outward transport and <var>S</var> is the source change. Conservation plots and this page's budget tables show <strong>fractions of the initial totals</strong>: Q(t)/D, [Q(0) − B(t) + S(t)]/D, and the signed residual R(t)/D. The denominator D = |Q(0)| is fixed for the entire history. It never grows with the current total or accumulated transport/source budgets. A negative initial flux therefore starts at −1, and the plotted fraction retains its sign.</p>
<p>A component whose initial net flux is zero or negligible cannot be divided by its own initial total. It instead uses the fixed scale D = c |E₀|, where E₀ is the initial integrated radiation energy. This fallback is labeled on its panels and in the table. “Negligible” means |Q(0)|/(c |E₀|) ≤ 64ε ≈ 1.421085 × 10<sup>−14</sup>, with ε the double-precision machine epsilon. This prevents roundoff left by cancellation in a symmetric field from becoming the denominator. A nonzero initial radiation energy is required.</p>
<p>Raw histories and dimensional summary columns remain in physical units: energy integrals in erg, physical-flux integrals in erg cm/s for CGS runs. Dividing a flux integral by c² gives radiation momentum. Fraction-history CSV files are dimensionless except for their time column; summary files record each denominator and its basis. Maximum fractional residuals use the same fixed denominator as the final residual.</p>
<p>Movies show the numerical Silo states with a fixed color range for each movie. Recorded states are held for their actual simulation-time gaps; intermediate solutions are not interpolated. The 3D option shows the domain's exterior surface. Movies and plots retain their recorded physical units.</p></section>)";
}
} // namespace
fs::path reportPages(const fs::path &batch, bool liveUpdate)
{
	require(fs::is_directory(batch), "Missing batch/run directory: " + batch.string());
	Json state = fs::is_regular_file(batch / "live.json") ? readJson(batch / "live.json") : Json::object();
	auto all = completedRuns(batch);
	require(!all.empty() || fs::is_regular_file(batch / "live.json"),
			"No completed runs or live batch in " + batch.string());
	std::map<std::string, std::vector<Run>> groups;
	std::map<std::string, std::string> unreadable;
	for (auto &[p, m] : all) {
		try {
			m["norms"] = readNorms(p, m);
			if (liveUpdate)
				readConservation(p, m);
			groups[m.at("case")].emplace_back(p, m);
		} catch (const std::exception &e) {
			// Publication precedes resume validation. Do not present invalid results;
			// let the runner's validation report the failure and clear the live state.
			if (!liveUpdate)
				throw;
			unreadable[m.at("case").get<std::string>()] +=
				"<p class=\"error\">Result unavailable: " + html(p.filename().string()) + ": " +
				html(e.what()) + "</p>";
		}
	}
	bool synthetic = false;
	for (auto &[p, m] : all)
		synthetic |= m.value("origin", "").find("fixture") != std::string::npos;
	const std::string fixtureNote =
		synthetic ? "<p class=\"note\"><strong>Synthetic validation data.</strong> These files exercise the "
					"report generator; they are not a solver accuracy result.</p>"
				  : "";
	std::string index =
		start("Radiation test problems") +
		"<header><div class=\"brand\">Octo-TIGER / Radiation verification</div><h1>Radiation test "
		"problems</h1><p class=\"lead\">Transport, relaxation, and equilibrium. Open a problem for its "
		"physical setup, reference solution, sources, and results at every completed resolution.</p>";
	index += "<div class=\"status\"><strong>" + html(state.value("message", "Saved results")) +
			 "</strong><span>" + std::to_string(all.size()) + " completed runs</span><span>" +
			 html(batch.filename().string()) + "</span></div>";
	if (state.contains("error"))
		index += "<p class=\"error\">" + show(state.at("error")) + "</p>";
	if (!unreadable.empty())
		index += "<p class=\"error\">Some saved results could not be validated. Their problem pages identify "
				 "the affected runs.</p>";
	index += fixtureNote + "</header><main><div class=\"cards\">";
	for (auto &d : problemDescriptions) {
		auto &runs = groups[d.name];
		std::sort(runs.begin(), runs.end(), [](auto &a, auto &b) {
			return a.second.at("dx").template get<double>() > b.second.at("dx").template get<double>();
		});
		const auto caseDir = batch / "plots" / d.name;
		std::string preview;
		for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
			const auto img =
				caseDir / ("l" + std::to_string(it->second.at("level").get<int>())) / "slice_er.png";
			if (!fs::is_regular_file(img))
				continue;
			preview = "<figure class=\"preview\"><a href=\"" + d.name +
					  ".html\"><img loading=\"lazy\" src=\"" +
					  url(fs::relative(img, batch).generic_string()) + "\" alt=\"" + d.title +
					  " energy comparison\"></a><figcaption>Energy comparison · " +
					  show(it->second.at("cells")) + "³ cells · t = " + show(it->second.at("time")) +
					  (cgs(it->second) ? " s" : " code time") + "</figcaption></figure>";
			break;
		}
		if (preview.empty())
			preview =
				"<div class=\"placeholder\">" +
				std::string(runs.empty() ? "Awaiting completed results" : "Plots have not been rendered") +
				"</div>";
		index += "<article class=\"card\">" + preview + "<div class=\"card-content\"><span class=\"tag\">" +
				 d.regime + "</span><h2><a href=\"" + d.name + ".html\">" + d.title + "</a></h2><p>" +
				 d.summary + "</p><p class=\"small\">" + std::to_string(runs.size()) +
				 " completed resolution" + (runs.size() == 1 ? "" : "s") + "</p><a class=\"cta\" href=\"" +
				 d.name + ".html\">Setup &amp; all results →</a></div></article>";
		std::string page = start(d.title) + "<header><a class=\"brand\" href=\"index.html\">← Octo-TIGER / "
											"All test problems</a><nav aria-label=\"Test problems\">";
		for (auto &other : problemDescriptions)
			page += "<a href=\"" + other.name + ".html\"" +
					(d.name == other.name ? " aria-current=\"page\"" : "") + ">" + other.title + "</a>";
		page += "</nav><span class=\"tag\">" + d.regime + "</span><h1>" + d.title +
				"</h1><p class=\"lead\">" + d.summary + "</p>" + fixtureNote +
				"</header><main><nav class=\"contents\" aria-label=\"On this page\"><a "
				"href=\"#setup\">Setup</a><a href=\"#reference\">Reference solution</a><a "
				"href=\"#results\">Results</a><a href=\"#diagnostics\">Diagnostics</a><a "
				"href=\"#sources\">Sources &amp; provenance</a></nav>";
		page += "<section class=\"section\" id=\"setup\"><span class=\"tag\">The "
				"experiment</span><h2>Physical setup</h2>" +
				d.setup +
				"<p>The standard runner uses a uniform cubic grid with hydrodynamics and gravity disabled. "
				"Its CGS box spans −3 × 10¹⁰ to +3 × 10¹⁰ cm on each axis, with <var>c</var> = 2.99792458 × "
				"10¹⁰ cm/s and default final time 4 s. A level has <var>N</var> = INX × 2<sup>level</sup> "
				"cells per side. Each completed run below lists its recorded parameters; template defaults "
				"are not substituted for missing metadata.</p></section>";
		page += "<section class=\"section\" id=\"reference\"><span class=\"tag\">The "
				"comparison</span><h2>How the reference is computed</h2>" +
				d.reference + "<h3>What this test measures</h3>" + d.interpretation + "</section>";
		page += "<section class=\"section\" id=\"results\"><span class=\"tag\">Measured "
				"results</span><h2>All resolutions</h2>" +
				unreadable[d.name] + plansTable(state, d.name);
		if (runs.empty()) {
			page += "<p>No completed run is available for this problem in this batch. Its description is "
					"available above; results will appear here as runs finish.</p>";
		} else {
			const bool compare = comparable(runs);
			if (compare) {
				auto chart = figure(batch, caseDir / "convergence",
									"Full-volume convergence for energy and all three flux components");
				page += chart.empty() ? "<p>Comparable runs are available. The norm table is current; use "
										"the plot command to render the convergence figure.</p>"
									  : chart;
			} else
				page +=
					"<p class=\"note\">" +
					std::string(runs.size() < 2
									? "Convergence requires at least two completed, comparable resolutions."
									: "These runs have different comparison settings or duplicate "
									  "resolutions; convergence orders are not computed.") +
					"</p>";
			Json errors = Json::array(), budgets = Json::array();
			page += "<h3>Full-volume norms and orders</h3>" + normsTable(runs, compare, errors);
			fs::create_directories(caseDir);
			recordsCsv(caseDir / "errors.csv", errors);
			writeJson(caseDir / "errors.json", errors);
			page += "<div class=\"downloads\">" +
					link(batch, caseDir / "errors.csv", "This problem: norms/orders CSV") +
					link(batch, caseDir / "errors.json", "JSON") + "</div>";
			for (auto &[folder, m] : runs) {
				const std::string level = std::to_string(m.at("level").get<int>());
				const auto target = caseDir / ("l" + level);
				page += "<article class=\"run\" id=\"level-" + level + "\"><h3>" + show(m.at("cells")) +
						"³ cells <span class=\"muted\">/ level " + level +
						"</span></h3><details><summary>Recorded setup and run provenance</summary>" +
						settings(m) + "<pre>" + html(m.dump(2)) + "</pre></details>";
				page += "<h4>Final fields and profiles</h4>";
				if (d.name == "gaussian_pulse") {
					auto styleFile = target / "display.json";
					if (fs::is_regular_file(styleFile) &&
						readJson(styleFile).value("energy_scale", "") == "log") {
						auto style = readJson(styleFile);
						page += "<p class=\"muted\">Gaussian energy is shown as E − Ebg on a logarithmic "
								"scale. Values at or below " +
								show(style.at("energy_floor")) +
								" use the display floor, including zero and negative excesses. Raw values, "
								"signed errors, and error norms are unchanged.</p>";
					} else
						page += "<p class=\"muted\">Existing energy plots may use a linear scale. Regenerate "
								"plots to apply the logarithmic energy-excess display.</p>";
				}
				if (!fs::is_regular_file(target / "slice_er.png"))
					page += "<p class=\"muted\">Plots are not yet available. Saved numerical norms and "
							"budgets are listed here.</p>";
				page += figure(batch, target / "slice_er",
							   "Numerical, reference, and signed-error energy slices at z = Δx/2");
				page += figure(
					batch, target / "profiles",
					"Numerical/reference line profiles and signed errors for all four radiation fields");
				page += "<details><summary>All three flux maps</summary>";
				for (int f = 1; f < 4; ++f)
					page += figure(batch, target / ("slice_" + fields[f]),
								   fields[f] + ": numerical, reference, and signed-error slice");
				page += "</details><h4>Movies</h4>" + movies(batch, folder) + "<h4>Conservation</h4>";
				if (auto b = readConservation(folder, m)) {
					page +=
						budgetTable(*b) +
						figure(batch, target / "conservation",
							   "Integrated radiation budgets and their transport/source-corrected residuals");
					for (auto r : b->summary) {
						r.update({{"case", d.name},
								  {"level", m.at("level")},
								  {"cells_per_side", m.at("cells")},
								  {"time", m.at("time")},
								  {"origin", m.value("origin", "")}});
						budgets.push_back(r);
					}
				} else
					page += "<p class=\"muted\">Conservation diagnostics were not recorded for this run.</p>";
				page += "<h4>Data and reproducibility</h4><div class=\"downloads\">";
				for (auto &[file, title] : std::vector<std::pair<fs::path, std::string>>{
						 {folder / "run.ini", "Effective configuration"},
						 {folder / "run.json", "Run metadata"},
						 {folder / "run.log", "Solver log"},
						 {target / "slice.csv", "Paired slice values CSV"},
						 {target / "slice.units.json", "Slice units"},
						 {folder / "radiation-conservation.csv", "Raw conservation history"},
						 {target / "conservation-summary.csv", "Budget summary CSV"},
						 {target / "conservation-fractions.csv", "Conservation fractions CSV"},
						 {target / "conservation-fractions.units.json", "Fraction units"},
						 {target / "conservation.units.json", "Conservation units"},
						 {folder / "L1.dat", "L1 norms"},
						 {folder / "L2.dat", "L2 norms"},
						 {folder / "Linf.dat", "L∞ norms"},
						 {folder / "reference.bin", "Gaussian reference binary"},
						 {folder / "final.silo", "Final numerical Silo"},
						 {folder / "analytic.silo", "Reference Silo"}})
					page += link(batch, file, title);
				page += "</div></article>";
			}
			recordsCsv(caseDir / "conservation.csv", budgets,
						{"case",
						 "level",
						 "cells_per_side",
						 "time",
						 "field",
						 "initial",
						 "final",
						 "boundary",
						 "source",
						 "residual",
						 "normalization_scale",
						 "normalization_basis",
						 "initial_fraction",
						 "final_fraction",
						 "boundary_fraction",
						 "source_fraction",
						 "normalized_residual",
						 "normalized_error",
						 "max_normalized_error",
						 "integral_units"});
			writeJson(caseDir / "conservation.json", budgets);
			if (!budgets.empty())
				page += "<div class=\"downloads\">" +
						link(batch, caseDir / "conservation.csv",
							 "This problem: all conservation summaries CSV") +
						link(batch, caseDir / "conservation.json", "JSON") + "</div>";
		}
		page += "</section>" + methodNotes();
		page += "<section class=\"section\" id=\"sources\"><span class=\"tag\">Sources &amp; "
				"provenance</span><h2>Where this problem comes from</h2>" +
				d.source;
		page +=
			R"(<h3>Radiation method context</h3><p><a href="https://arxiv.org/abs/1306.0010">Skinner &amp; Ostriker (2013), “A Two-moment Radiation Hydrodynamics Module in Athena Using a Time-explicit Godunov Method,” ApJS 206, 21</a>, provides context for the two-moment M1 system and source integration. The suite's fixed-medium damping uses the algebra of their equations (42b)–(43) with θ = 1 and prescribed χ. Their paper neglects scattering; the suite reuses that damping formula for its prescribed scattering medium.</p>
<p>The solver's M1 implementation also cites <a href="https://doi.org/10.1016/j.jqsrt.2014.04.014">Hanawa &amp; Audit (2014), “Reformulation of the M1 model of radiative transfer,” JQSRT 145, 9–16</a>. These references describe the numerical/physical framework; the exact test definitions are the source functions identified above.</p>
<h3>Reproducing this batch</h3><p>Descriptions document the reviewed source implementation. Older or edited solver versions can use different profile constants or averaging rules; these are not all encoded in <code>run.ini</code>. Use the run metadata, executable hash, and saved batch source hashes to identify the implementation behind a result. A source path or descriptive formula alone is not proof of the executable's contents.</p>)";
		page += "<div class=\"downloads\">" +
				link(batch, batch / "batch.json", "Batch settings and source hashes") +
				link(batch, batch / "live.json", "Batch status") + "</div></section>" + finish(state);
		atomicText(batch / (d.name + ".html"), page);
	}
	index += "</div><div class=\"downloads\">" +
			 link(batch, batch / "plots/index.html", "Combined plot report") +
			 link(batch, batch / "movies.html", "Movie gallery") +
			 link(batch, batch / "plots/errors.csv", "All norms CSV") +
			 link(batch, batch / "batch.json", "Batch metadata") + "</div>" + finish(state);
	atomicText(batch / "index.html", index);
	return batch / "index.html";
}
} // namespace rr
