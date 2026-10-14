#include "problem_descriptions.hpp"
#include <tuple>
namespace rr
{
namespace
{
using Run = std::pair<fs::path, Json>;
std::string validation_notes(const std::vector<Run> &runs)
{
    size_t short_runs = 0;
    std::set<double> times;
    std::set<int> cells;
    for (const auto &[path, m] : runs) {
        if (coupled_case(m.at("case")) && cgs(m) && m.value("time", 4.0) < m.value("benchmark_time", 4.0)) {
            ++short_runs;
        }
        if (m.contains("time")) {
            times.insert(m.at("time").get<double>());
        }
        if (m.contains("cells")) {
            cells.insert(m.at("cells").get<int>());
        }
    }
    std::string notes;
    if (short_runs != 0) {
        const bool preview = short_runs == runs.size() && !cells.empty() && *cells.rbegin() <= 8;
        notes = "<p class=\"note\"><strong>" + std::string(preview ?
            "Short-run implementation preview." : "Includes short-duration runs.") + "</strong>";
        if (times.size() == 1 && cells.size() == 1) {
            notes += " These saved results use " + std::to_string(*cells.begin()) +
                "³ cells and end at " + html(number(*times.begin())) + " s.";
        }
        if (cells.size() == 1) {
            notes += " Only one spatial resolution is shown, so these results do not measure a convergence order.";
        }
        notes += " Runs shorter than the benchmark duration do not establish developed shock or blast "
            "structure. Full-duration benchmark validation remains separate.</p>";
    }
    const bool fld = std::any_of(runs.begin(), runs.end(), [](const auto &run) { return fld_model(run.second); });
    if (fld) {
        notes += "<p class=\"note\"><strong>FLD method scope.</strong> Grey radiation at physical light speed on CPU, backward-Euler diffusion/material exchange, and the Levermore–Pomraning limiter. Moving cases include first-order gas transport, radiation force/work, and a derived lab-frame flux with radiation enthalpy advection. Where supplied, analytic diffusion-limit references omit nonlinear-limiter and boundary-model effects. Thin streaming cases have no supplied analytic spatial FLD profile and use exact energy integrals instead. Paired M1 RK1 and FLD timings use different stability limits and do not establish performance at matched error.</p>";
    }
    return notes;
}
std::string show(const Json &v)
{
	if (v.is_null()) {
    return "—";
  }
  if (v.is_string()) {
    return html(v.get<std::string>());
  }
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
					"local source implementation. "
					"Recorded run settings and provenance accompany the results.";
	if (state.value("active", false)) {
    s +=
			R"(<label class="refresh"><input id="refresh" type="checkbox" checked> Refresh every 10 seconds; paused during movie playback</label><script>try{window.scrollTo(0,Number(sessionStorage.getItem('scroll:'+location.pathname)||0));}catch(e){}setInterval(()=>{if(!document.getElementById('refresh').checked||Array.from(document.querySelectorAll('video')).some(v=>!v.paused&&!v.ended))return;try{sessionStorage.setItem('scroll:'+location.pathname,String(window.scrollY));}catch(e){}location.reload();},10000);</script>)";
  }
  return s + "</footer></body></html>\n";
}
std::string link(const fs::path &batch, const fs::path &p, const std::string &label)
{
	if (!fs::is_regular_file(p)) {
    return "";
  }
  return "<a href=\"" + url(fs::relative(p, batch).generic_string()) + "\">" + html(label) + "</a>";
}
std::string figure(const fs::path &batch, const fs::path &stem, const std::string &caption)
{
	auto png = fs::path(stem.string() + ".png");
	if (!fs::is_regular_file(png)) {
    return "";
  }
  auto rel = url(fs::relative(png, batch).generic_string());
	return "<figure><a href=\"" + rel + "\"><img loading=\"lazy\" src=\"" + rel + "\" alt=\"" +
		   html(caption) + "\"></a><figcaption>" + html(caption) + " · " +
		   link(batch, fs::path(stem.string() + ".pdf"), "PDF") + "</figcaption></figure>";
}
bool comparable(const std::vector<Run> &runs)
{
	if (runs.size() < 2) {
    return false;
  }
  std::set<double> dx;
	for (auto &[p, m] : runs) {
		if (m.value("adaptive", false)) {
			return false;
		}
		if (!dx.insert(m.at("dx").get<double>()).second) {
      return false;
    }
    for (auto key : {"time", "length", "c", "background", "comparison_signature", "executable_sha256",
						 "origin", "units"}) {
      if (m.value(key, Json()) != runs.front().second.value(key, Json())) {
        return false;
      }
    }
  }
	return true;
}
std::string plans_table(const Json &state, const std::string &name)
{
	std::string rows;
	for (auto &r : state.value("runs", Json::array())) {
    if (r.value("case", "") == name) {
      rows += "<tr><td>" + show(r.value("level", Json())) + "</td><td>" +
					show(r.value("cells", Json())) + "³</td><td class=\"wrap\">" +
					show(r.value("stage", Json())) + "</td></tr>";
    }
  }
  if (rows.empty()) {
    return "";
  }
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
	s += row("Resolution", html(mesh_label(m)) + "; finest Δx = " + show(m.at("dx")) + lu);
	if (m.value("adaptive", false)) {
		s += row("Slice geometry", html(slice_geometry_note(m)));
	}
	s += row("Final comparison time", show(m.at("time")) + tu);
    const auto method = cfg.value("radiation_integrator", "euler");
    s += row("Radiation model", fld_model(m) ? "FLD" : "M1");
    s += row("Radiation integration", fld_model(m) ? "FLD backward Euler (first order in time)" :
        method == "euler" ? "M1 RK1 (first order in time)" : "Historical method: " + html(method));
    s += row("Flux semantics", fld_model(m) ? (coupled_case(m.at("case")) ?
        "Derived lab-frame FLD flux: F = -D grad E + (E I + P) v" : "Derived FLD flux: F = -D grad E") : "Evolved M1 flux moment");
    if (fld_model(m)) {
        if (fld_reference_case(m.at("case"))) {
            s += row("Analytic reference", html(m.value("reference_model", "")) +
                "; linear diffusion-limit reference, not exact for the nonlinear FLD limiter");
            s += row("Reference norm scope", "Area-weighted projected positive-side slice; no full-volume norm or convergence order inferred");
        } else {
            s += row("Analytic reference", coupled_case(m.at("case")) ? "No independent analytic or digitized benchmark profile supplied" : "No analytic spatial FLD solution supplied; exact periodic energy integral checked separately");
            s += row("Reference norm scope", "No analytic-profile norms or convergence orders");
        }
    }
    s += row("Radiation backend", html(cfg.value("radiation_backend", "cpu")));
	s += row("Physical light speed", show(m.at("c")) + (cgs(m) ? " cm/s" : " (code speed)"));
    if (m.contains("radiation_reduced_c_factor")) {
        s += row("Reduced-speed factor α", show(m.at("radiation_reduced_c_factor")));
        s += row("Radiation transport speed", show(m.at("transport_c")) +
            (cgs(m) ? " cm/s" : " (code speed)"));
    }
	for (const auto &[key, title] :
		 std::vector<std::pair<std::string, std::string>>{{"periodic", "Periodic boundaries"},
														  {"hydro", "Hydrodynamics"},
														  {"gravity", "Gravity"},
														  {"rad_implicit", "Implicit radiation source"},
														  {"unigrid", "Uniform grid"},
														  {"cfl", "CFL"},
														  {"hard_dt", "Timestep cap"},
														  {"odt", "Output interval"},
														  {"rad_reconstruction", "Reconstruction"}}) {
    if (cfg.contains(key)) {
      s += row(title, show(cfg.at(key)) + ((key == "hard_dt" || key == "odt") ? tu : ""));
    }
  }
  if (m.at("case") == "gaussian_pulse" || m.at("case") == "equilibrium_sphere") {
		for (const auto &[key, title, unit] : std::vector<std::tuple<std::string, std::string, std::string>>{
				 {"rad_test_background", "Background energy density", "erg/cm³"},
				 {"rad_test_chi", "Scattering coefficient χ", "cm⁻¹"},
				 {"rad_test_width", "Gaussian width w", "cm"},
				 {m.at("case") == "gaussian_pulse" ? "rad_test_amplitude" : "rad_test_luminosity",
				  m.at("case") == "gaussian_pulse" ? "Pulse amplitude A" : "All-space luminosity ℒ",
				  m.at("case") == "gaussian_pulse" ? "erg/cm³" : "erg/s"}}) {
      s += row(title, cfg.contains(key) ? show(cfg.at(key)) + " " + (cgs(m) ? unit : "(code units)")
											  : "Not recorded");
    }
  }
	if (cfg.empty()) {
    s += row("Effective configuration", "Not recorded in this run's metadata");
  }
  return s + "</tbody></table></div>";
}
std::string norms_table(const std::vector<Run> &runs, bool compare, Json &errors)
{
	std::string s = "<div "
					"class=\"scroll\"><table><thead><tr><th>Grid</th><th>Field</th><th>Norm</th><th>Error</"
					"th><th>Order</th><th>Error units</th></tr></thead><tbody>";
	Json previous;
	for (auto &[p, m] : runs) {
		std::vector<std::string> measured_fields(fields.begin(), fields.end());
        if (m.at("norms").contains("material_egas")) { measured_fields.push_back("material_egas"); }
        for (size_t f = 0; f < measured_fields.size(); ++f) {
      for (auto &n : norms) {
				const double error = m.at("norms").at(measured_fields[f]).at(n);
				Json order;
				if (compare && !previous.is_null() && error > 0 &&
					previous["norms"][measured_fields[f]][n].get<double>() > 0) {
          order = std::log(previous["norms"][measured_fields[f]][n].get<double>() / error) /
							std::log(previous.at("dx").get<double>() / m.at("dx").get<double>());
        }
        const std::string units = cgs(m) ? (f == 0 ? "erg/cm³" : "erg/(cm² s)") : "code units";
				errors.push_back({{"case", m.at("case")},
								  {"level", m.at("level")},
								  {"cells_per_side", m.at("cells")},
								  {"dx", m.at("dx")},
								  {"time", m.at("time")},
								  {"field", measured_fields[f]},
								  {"norm", n},
								  {"error", error},
								  {"order", order},
								  {"error_units", units},
								  {"origin", m.value("origin", "")}});
				s += "<tr><td>" + html(mesh_label(m)) + "</td><td>" + measured_fields[f] + "</td><td>" + n +
					 "</td><td>" + show(error) + "</td><td>" + show(order) + "</td><td>" + units +
					 "</td></tr>";
			}
    }
    previous = m;
	}
	return s + "</tbody></table></div>";
}
std::string movies(const fs::path &batch, const fs::path &folder)
{
	if (!fs::is_directory(folder / "movies")) {
    return "<p class=\"muted\">No completed movie is available for this resolution.</p>";
  }
  std::vector<fs::path> manifests;
	for (auto &e : fs::recursive_directory_iterator(folder / "movies")) {
    if (e.is_regular_file() && e.path().filename() == "movie.json") {
      manifests.push_back(e.path());
    }
  }
  std::sort(manifests.begin(), manifests.end());
	std::string s;
	for (auto &p : manifests) {
		auto m = read_json(p);
		auto file = p.parent_path() / "movie.mp4";
		if (m.value("status", "") != "complete" || !fs::is_regular_file(file)) {
      continue;
    }
    auto rel = url(fs::relative(file, batch).generic_string());
		s += "<figure><video controls preload=\"none\" src=\"" + rel + "\"></video><figcaption>" +
			 show(m.at("field")) + " · " +
			 (m.value("view", "slice") == "3d" ? "exterior surface"
											   : "slice normal to " + show(m.value("axis", "z"))) +
			 " · " + show(m.at("source_snapshots")) + " recorded snapshots · " + show(m.at("seconds")) +
			 " s playback";
		if (m.value("color_scale", "linear") == "log") {
      s += ". Logarithmic energy excess E − Ebg; display floor " + show(m.at("color_limits")[0]) + " " +
				 show(m.at("field_units")) + ". Values at or below the floor use the lowest color.";
    } else if (m.value("case", "") == "gaussian_pulse" && m.value("field", "") == "er") {
      s += ". Existing linear-scale movie; regenerate movies to apply the logarithmic energy-excess "
				 "display.";
    }
    s += ". " + link(batch, file, "MP4") + " · " + link(batch, p, "Movie metadata") +
			 "</figcaption></figure>";
	}
	return s.empty() ? "<p class=\"muted\">No completed movie is available for this resolution.</p>" : s;
}
std::string budget_table(const Budget &b, const Json &m)
{
	std::string s = conservation_scope_note(b, m) +
        "<p>All values below are dimensionless fractions of the fixed normalization scale at t_ref = " +
        html(number(b.history.rows.front()[0])) + (cgs(m) ? " s.</p>" : ".</p>") +
					"<div "
					"class=\"scroll\"><table><thead><tr><th>Field</th><th>Scale</th><th>Q(t_ref)/scale</"
					"th><th>Q(t)/scale</th><th>Boundary "
					"B/scale</th><th>Source S/scale</th><th>Residual R/scale</th><th>Maximum "
					"|R|/scale</th></tr></thead><tbody>";
	for (auto &r : b.summary) {
		s += "<tr><th scope=\"row\">" + show(r.at("field")) + "</th><td>" +
			 (r.at("normalization_basis") == "bath_energy * volume" ? "E<sub>bath</sub> V" :
              r.at("normalization_basis") == "c * bath_energy * volume" ? "c E<sub>bath</sub> V" :
              (r.at("normalization_basis") == "c * abs(initial_energy)" ||
               r.at("normalization_basis") == "c * abs(segment_initial_energy)")
				  ? "c |E(t_ref)| (zero/negligible reference flux)"
				  : "|Q(t_ref)|") +
			 "</td>";
		for (auto key : {"initial_fraction", "final_fraction", "boundary_fraction", "source_fraction",
						 "normalized_residual", "max_normalized_error"}) {
      s += "<td>" + show(r.at(key)) + "</td>";
    }
    s += "</tr>";
	}
	return s + "</tbody></table></div>";
}
std::string method_notes()
{
	return R"(<section class="section" id="diagnostics"><span class="tag">Reading the results</span><h2>Errors, budgets, and movies</h2>
<p>Every norm uses the full three-dimensional domain, with cell-volume weighting. Here <var>e</var> = numerical − reference and <var>V</var> is the domain volume:</p>
<div class="equation">L1 = Σ |e| ΔV / V, &nbsp; L2 = √(Σ e² ΔV / V), &nbsp; L∞ = max |e|<br>p = ln(error<sub>coarse</sub> / error<sub>fine</sub>) / ln(Δx<sub>coarse</sub> / Δx<sub>fine</sub>)</div>
<p>Orders require two completed runs with matching physical time, configuration, units, executable, and origin. An exact zero error has no logarithmic order. The central slice and line profiles are visual diagnostics: the report slice is the cell layer at <var>z</var> = Δ<var>x</var>/2, and its line profile is at <var>y</var> = <var>z</var> = Δ<var>x</var>/2.</p>
<p>Conservation is assessed using the measured domain integrals and the signed cumulative boundary and source budgets:</p>
<p>A labeled checkpoint continuation uses its checkpoint as t_ref, with counters and fixed normalization starting there. Its reported residuals cover that resumed segment only; original histories remain separate.</p>
<div class="equation">R(t) = Q(t) − Q(0) + B(t) − S(t)</div>
<p><var>B</var> is outward transport and <var>S</var> is the source change. Conservation plots and this page's budget tables show <strong>fractions of the initial totals</strong>: Q(t)/D, [Q(0) − B(t) + S(t)]/D, and the signed residual R(t)/D. The denominator D = |Q(0)| is fixed for the entire history. It never grows with the current total or accumulated transport/source budgets. A negative initial flux therefore starts at −1, and the plotted fraction retains its sign.</p>
<p>A component whose initial net flux is zero or negligible cannot be divided by its own initial total. It instead uses the fixed scale D = c |E₀|, where E₀ is the initial integrated radiation energy. This fallback is labeled on its panels and in the table. “Negligible” means |Q(0)|/(c |E₀|) ≤ 64ε ≈ 1.421085 × 10<sup>−14</sup>, with ε the double-precision machine epsilon. This prevents roundoff left by cancellation in a symmetric field from becoming the denominator. For the initially cold Marshak case, the prescribed bath sets D = E<sub>bath</sub> V for energy and D = c E<sub>bath</sub> V for flux, with E<sub>bath</sub> = 4/c. These scales remain fixed even though radiation enters the initially empty domain.</p>
<p>Raw histories and dimensional summary columns remain in physical units: energy integrals in erg, physical-flux integrals in erg cm/s for CGS runs. Dividing a flux integral by c² gives radiation momentum. Fraction-history CSV files are dimensionless except for their time column; summary files record each denominator and its basis. Maximum fractional residuals use the same fixed denominator as the final residual.</p>
<p>For runs labeled RSLA, these budgets check the modified radiation equations. They do not establish conservation of physical gas-plus-radiation energy. With reduced-speed factor α, the closed-system coupled invariants are Egas + Er/α and gas momentum + F/(αc²), with physical c in the closure. Compare multiple α values before interpreting an RSLA result as an approximation to physical-c evolution.</p>
<p>Movies show the numerical Silo states with a fixed color range for each movie. Recorded states are held for their actual simulation-time gaps; intermediate solutions are not interpolated. The 3D option shows the domain's exterior surface. Movies and plots retain their recorded physical units.</p></section>)";
}
} // namespace
fs::path report_pages(const fs::path &batch, bool live_update)
{
	require(fs::is_directory(batch), "Missing batch/run directory: " + batch.string());
	Json state = fs::is_regular_file(batch / "live.json") ? read_json(batch / "live.json") : Json::object();
	auto all = completed_runs(batch);
	require(!all.empty() || fs::is_regular_file(batch / "live.json"),
			"No completed runs or live batch in " + batch.string());
	std::map<std::string, std::vector<Run>> groups;
	std::map<std::string, std::string> unreadable;
	for (auto &[p, m] : all) {
		try {
			if (fld_model(m)) {
                m["fld_diagnostics"] = read_fld_diagnostics(p, m);
                if (coupled_case(m.at("case"))) { read_hydro_profile(p, m); }
                else { read_slice(p, m); }
                if (m.at("case") == "marshak") { read_marshak_material(p, m); }
            } else if (coupled_case(m.at("case"))) {
                read_hydro_profile(p, m);
            } else {
                m["norms"] = read_norms(p, m);
            }
            if (live_update) {
        read_conservation(p, m);
      }
      groups[m.at("case")].emplace_back(p, m);
		} catch (const std::exception &e) {
			// Publication precedes resume validation. Do not present invalid results;
			// let the runner's validation report the failure and clear the live state.
			if (!live_update) {
        throw;
      }
      unreadable[m.at("case").get<std::string>()] +=
				"<p class=\"error\">Result unavailable: " + html(p.filename().string()) + ": " +
				html(e.what()) + "</p>";
		}
	}
	bool synthetic = false;
	for (auto &[p, m] : all) {
    synthetic |= m.value("origin", "").find("fixture") != std::string::npos;
  }
  const std::string fixture_note =
		synthetic ? "<p class=\"note\"><strong>Synthetic validation data.</strong> These files exercise the "
					"report generator; they are not a solver accuracy result.</p>"
				  : "";
	std::string index =
		start("Radiation test problems") +
		"<header><div class=\"brand\">Octo-TIGER / Radiation verification</div><h1>Radiation test "
		"problems</h1><p class=\"lead\">M1 RK1 and FLD. Open a problem for its "
		"physical setup, comparison method, sources, and results at every completed resolution.</p>";
	index += "<div class=\"status\"><strong>" + html(state.value("message", "Saved results")) +
			 "</strong><span>" + std::to_string(all.size()) + " completed runs</span><span>" +
			 html(batch.filename().string()) + "</span></div>";
	if (state.contains("error")) {
    index += "<p class=\"error\">" + show(state.at("error")) + "</p>";
  }
  if (!unreadable.empty()) {
    index += "<p class=\"error\">Some saved results could not be validated. Their problem pages identify "
				 "the affected runs.</p>";
  }
  index += fixture_note + validation_notes(all) + "</header><main><div class=\"cards\">";
    const bool fld_batch = !all.empty() && fld_model(all.front().second);
    for (const auto &[path, m] : all) { require(fld_model(m) == fld_batch, "Use separate M1 and FLD batches"); }
	for (const auto &base_description : problem_descriptions) {
        const auto d = fld_batch ? fld_problem_description(base_description) : base_description;
        const bool fld = fld_batch && fld_case(d.name);
        const bool coupled = coupled_case(d.name);
		auto &runs = groups[d.name];
		std::sort(runs.begin(), runs.end(), [](auto &a, auto &b) {
			return a.second.at("dx").template get<double>() > b.second.at("dx").template get<double>();
		});
		const auto case_dir = batch / "plots" / d.name;
		std::string preview;
		for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
			const auto img =
				case_dir / ("l" + std::to_string(it->second.at("level").get<int>())) / (coupled ? "hydro_slice.png" : "slice_er.png");
			if (!fs::is_regular_file(img)) {
        continue;
      }
      preview = "<figure class=\"preview\"><a href=\"" + d.name +
					  ".html\"><img loading=\"lazy\" src=\"" +
					  url(fs::relative(img, batch).generic_string()) + "\" alt=\"" + d.title +
					  (coupled ? " numerical fields\"></a><figcaption>Numerical fields · " : " energy comparison\"></a><figcaption>Energy comparison · ") +
					  html(mesh_label(it->second)) + " · t = " + show(it->second.at("time")) +
					  (cgs(it->second) ? " s" : " code time") + "</figcaption></figure>";
			break;
		}
		if (preview.empty()) {
      preview =
				"<div class=\"placeholder\">" +
				std::string(runs.empty() ? "Awaiting completed results" : "Plots have not been rendered") +
				"</div>";
    }
    index += "<article class=\"card\">" + preview + "<div class=\"card-content\"><span class=\"tag\">" +
				 d.regime + "</span><h2><a href=\"" + d.name + ".html\">" + d.title + "</a></h2><p>" +
				 d.summary + "</p><p class=\"small\">" + std::to_string(runs.size()) +
				 " completed resolution" + (runs.size() == 1 ? "" : "s") + "</p><a class=\"cta\" href=\"" +
				 d.name + ".html\">Setup &amp; all results →</a></div></article>";
		std::string page = start(d.title) + "<header><a class=\"brand\" href=\"index.html\">← Octo-TIGER / "
											"All test problems</a><nav aria-label=\"Test problems\">";
		for (auto &other : problem_descriptions) {
      page += "<a href=\"" + other.name + ".html\"" +
					(d.name == other.name ? " aria-current=\"page\"" : "") + ">" + other.title + "</a>";
    }
    page += "</nav><span class=\"tag\">" + d.regime + "</span><h1>" + d.title +
				"</h1><p class=\"lead\">" + d.summary + "</p>" + fixture_note + validation_notes(runs) +
				"</header><main><nav class=\"contents\" aria-label=\"On this page\"><a "
				"href=\"#setup\">Setup</a><a href=\"#reference\">Comparison method</a><a "
				"href=\"#results\">Results</a><a href=\"#diagnostics\">Diagnostics</a><a "
				"href=\"#sources\">Sources &amp; provenance</a></nav>";
        page += "<section class=\"section\" id=\"setup\"><span class=\"tag\">The experiment</span>"
            "<h2>Physical setup</h2>" + d.setup;
        page += d.name == "marshak" ?
            "<p>This case uses normalized units with c = 1, opacity 100, and equal material/radiation energy capacities. Hydrodynamics and gravity are disabled. The default cube spans −1 to +1, and the default final time is 1. Adaptive runs retain their actual leaf spacing in exported data.</p></section>" : coupled ?
            "<p>These cases enable gas dynamics and radiation coupling on a Cartesian grid; gravity is disabled. Uniform runs are the default; adaptive runs are identified in their recorded setup. "
            "Each completed run records its actual configuration, time, units, and executable hash below. "
            "The configured finest equivalent cells per side are N = INX × 2<sup>level</sup>.</p></section>" :
            "<p>The standard runner uses a uniform cubic grid with hydrodynamics and gravity disabled. "
            "Its CGS box spans −3 × 10¹⁰ to +3 × 10¹⁰ cm on each axis, with c = 2.99792458 × 10¹⁰ cm/s "
            "and default final time 4 s. Opt-in adaptive runs are identified below; N = INX × 2<sup>level</sup> then describes their configured finest equivalent resolution. "
            "Each completed run below lists its recorded parameters.</p></section>";
		page += "<section class=\"section\" id=\"reference\"><span class=\"tag\">The "
				"comparison</span><h2>Comparison and interpretation</h2>" +
				d.reference + "<h3>What this test measures</h3>" + d.interpretation + "</section>";
		page += "<section class=\"section\" id=\"results\"><span class=\"tag\">Measured "
				"results</span><h2>All resolutions</h2>" +
				unreadable[d.name] + plans_table(state, d.name);
		if (runs.empty()) {
			page += "<p>No completed run is available for this problem in this batch. Its description is "
					"available above; results will appear here as runs finish.</p>";
            if (fld_batch && !fld) { page += "<p class=\"note\">Unsupported FLD problem; no FLD run is planned for this problem.</p>"; }
        } else if (fld && !coupled) {
            const bool references = fld_reference_case(d.name);
            page += references ?
                "<p class=\"note\">Analytic diffusion-limit reference comparison. The reference is exact for its stated linear diffusion equations, not for the nonlinear FLD limiter. Flux remains a derived diagnostic; no separate M1 momentum conservation test applies.</p>" :
                "<p class=\"note\">No analytic spatial FLD solution is supplied for this vacuum problem. The translated M1 solution is not an FLD reference. Exact total-energy conservation and physical flux bounds are reported instead; no analytic-profile errors or convergence orders are inferred.</p>";
            for (const auto &[folder, m] : runs) {
                const auto target = case_dir / ("l" + std::to_string(m.at("level").get<int>()));
                const auto diagnostics = read_fld_diagnostics(folder, m);
                page += "<article class=\"run\"><h3>FLD backward Euler · " + html(mesh_label(m)) + "</h3><p>" +
                    html(slice_geometry_note(m)) + "</p><p>" + html(radiation_method_note(m)) +
                    "</p><details><summary>Recorded setup and run provenance</summary>" + settings(m) +
                    "<pre>" + html(m.dump(2)) + "</pre></details>";
                page += figure(batch, target / "slice_er", (references ?
                    "FLD radiation energy, analytic diffusion-limit reference, and signed difference. " :
                    "Numerical FLD radiation energy. ") + slice_geometry_note(m));
                page += figure(batch, target / "profiles", references ?
                    "Radiation energy and derived FLD flux against analytic diffusion-limit profiles, with signed differences. The reference does not include the nonlinear flux limiter." :
                    "Numerical radiation energy and derived FLD flux. No analytic spatial reference is supplied.");
                if (d.name == "marshak") {
                    page += figure(batch, target / "material_profiles", "Radiation and material energy relative to bath energy, compared with the Su–Olson analytic linear diffusion reference.");
                }
                page += "<details><summary>Derived FLD flux maps</summary>";
                for (int f = 1; f < 4; ++f) {
                    page += figure(batch, target / ("slice_" + fields[f]), "Derived FLD " + fields[f] +
                        (references ? ": numerical flux, analytic diffusion-reference flux, and signed difference." : ": numerical gradient-driven flux."));
                }
                page += "</details><h4>Accepted implicit solves</h4><p>Accepted steps: " + show(diagnostics.at("accepted_steps")) +
                    "; substeps: " + show(diagnostics.at("substeps")) + "; timestep retries: " + show(diagnostics.at("retries")) +
                    "; maximum nonlinear residual: " + show(diagnostics.at("maximum_nonlinear_residual")) +
                    "; maximum absolute coupled energy error per accepted step: " +
                    show(diagnostics.at("maximum_step_coupled_conservation_error")) + ".</p>";
                if (references) {
                    page += "<h4>Projected-slice differences from the analytic reference</h4><p>Area-weighted L1, L2, and maximum differences on the exported positive-side slice; these are not full-volume norms or a pure discretization error. No convergence order is inferred. Energy-density and flux units are those recorded in the slice-units download.</p><div class=\"scroll\"><table><tr><th>Field</th><th>L1 difference</th><th>L2 difference</th><th>Maximum difference</th></tr>";
                    const auto &differences = diagnostics.at("slice_reference_norms");
                    for (auto field = differences.begin(); field != differences.end(); ++field) {
                        page += "<tr><th>" + html(field.key()) + "</th><td>" + show(field.value().at("L1")) +
                            "</td><td>" + show(field.value().at("L2")) + "</td><td>" + show(field.value().at("Linf")) + "</td></tr>";
                    }
                    page += "</table></div>";
                } else {
                    const auto &invariants = diagnostics.at("analytic_invariants");
                    const auto &bounds = diagnostics.at("slice_physical_bounds");
                    page += "<h4>Analytic integral and physical bounds</h4><p>For this source-free periodic problem the exact mean radiation energy is " +
                        show(invariants.at("expected_mean_radiation_energy")) +
                        ". The maximum relative deviation of the recorded full-volume radiation energy from that exact integral is " +
                        show(invariants.at("maximum_relative_integrated_energy_error")) +
                        ". This checks an integral invariant, not the spatial solution.</p><p>On the exported projected slice, minimum E = " +
                        show(bounds.at("minimum_radiation_energy")) + "; maximum |F|/(cE) = " +
                        show(bounds.at("maximum_reduced_flux")) +
                        ". The expected bounds are E ≥ 0 and |F|/(cE) ≤ 1. These are slice checks, not full-volume extrema.</p>";
                }
                const auto budget = *read_conservation(folder, m);
                page += "<h4>Radiation energy conservation</h4>" + budget_table(budget, m) +
                    figure(batch, target / "conservation", "Radiation energy after signed boundary transport, material exchange, and any prescribed source. Flux moments are not conserved variables in FLD.");
                page += "<h4>Movies</h4>" + movies(batch, folder) + "<h4>Data and reproducibility</h4><div class=\"downloads\">";
                for (const auto &[file, label] : std::vector<std::pair<fs::path, std::string>>{
                    {folder / "run.ini", "Effective configuration"}, {folder / "run.json", "Run metadata"},
                    {folder / "run.log", "Solver log"}, {target / "slice.csv", references ? "Numerical/reference/difference slice CSV" : "Numerical slice CSV"},
                    {target / "slice.units.json", "Slice units"}, {target / "fld-diagnostics.json", "FLD diagnostics"},
                    {target / "fld-steps.csv", "Accepted solve history CSV"},
                    {target / "reference-differences.csv", "Slice reference-difference norms CSV"},
                    {target / "analytic-invariants.json", "Analytic integral invariant checks"},
                    {target / "conservation.csv", "Radiation energy history CSV"},
                    {target / "conservation-summary.csv", "Energy budget summary CSV"},
                    {folder / "reference.bin", "Gaussian analytic diffusion-limit reference binary"},
                    {folder / "final.silo", "Final numerical Silo"}}) {
                    page += link(batch, file, label);
                }
                page += "</div></article>";
            }
        } else if (coupled) {
            page += "<p class=\"note\">These are numerical solutions without a supplied analytic reference. "
                "No error norms or convergence orders are inferred from the profiles. "
                "Short runs used for implementation checks do not establish the published late-time shock or blast structure.</p>";
            for (const auto &[folder, m] : runs) {
                const auto level = std::to_string(m.at("level").get<int>());
                const auto target = case_dir / ("l" + level);
                page += "<article class=\"run\" id=\"level-" + level + "\"><h3>" + html(mesh_label(m)) +
                    " · actual final time " + show(m.at("time")) + " s</h3><p>" + html(slice_geometry_note(m)) + "</p>";
                if (m.contains("validation_scope")) {
                    page += "<p class=\"note\">" + show(m.at("validation_scope")) + "</p>";
                }
                page += "<p class=\"note\">" + html(radiation_method_note(m)) + "</p>";
                page += "<details><summary>Recorded setup and run provenance</summary>" + settings(m) +
                    "<pre>" + html(m.dump(2)) + "</pre></details>";
                page += figure(batch, target / "hydro_slice", "Numerical gas temperature, radiation temperature, and density. " + slice_geometry_note(m));
                page += figure(batch, target / "profiles", "Numerical profiles; no reference curves are supplied. " + slice_geometry_note(m));
                page += "<h4>Movies</h4>" + movies(batch, folder);
                if (fld_model(m)) {
                    const auto diagnostics = read_fld_diagnostics(folder, m);
                    page += "<h4>Implicit solve and energy balances</h4><p>Radiation-relative imbalance and the raw accepted-step coupled error are reported separately. A gas-dominated shock can have a much larger radiation-relative imbalance than its residual normalized by gas plus radiation energy.</p><pre>" + html(diagnostics.dump(2)) + "</pre>";
                }
                if (m.contains("checkpoint_resume")) {
                    const auto budget = read_conservation(folder, m);
                    require(bool(budget), "Missing checkpoint conservation segment");
                    page += "<h4>Conservation of the resumed segment</h4>" + budget_table(*budget, m);
                }
                page += "<h4>Data and reproducibility</h4><div class=\"downloads\">";
                for (const auto &[file, label] : std::vector<std::pair<fs::path, std::string>>{
                    {folder / "run.ini", "Effective configuration"}, {folder / "run.json", "Run metadata"},
                    {folder / "radiation-conservation.csv", "Raw conservation segment"},
                    {folder / "conservation-segments/original-attempt.csv", "Original raw ledger (tail after checkpoint is abandoned)"},
                    {folder / "run.log", "Solver log"}, {target / "hydro-slice.csv", "Numerical slice CSV"},
                    {target / "hydro-slice.units.json", "Data units"}, {folder / "final.silo", "Final numerical Silo"}}) {
                    page += link(batch, file, label);
                }
                page += "</div></article>";
            }
		} else {
			const bool compare = comparable(runs);
			if (compare) {
				auto chart = figure(batch, case_dir / "convergence",
									"Full-volume convergence for energy and all three flux components");
				page += chart.empty() ? "<p>Comparable runs are available. The norm table is current; use "
										"the plot command to render the convergence figure.</p>"
									  : chart;
			} else {
        page +=
					"<p class=\"note\">" +
					std::string(runs.size() < 2
									? "Convergence requires at least two completed, comparable resolutions."
									: "These runs have different comparison settings or duplicate "
									  "resolutions; convergence orders are not computed.") +
					"</p>";
      }
      Json errors = Json::array(), budgets = Json::array();
			page += "<h3>Full-volume norms and orders</h3>" + norms_table(runs, compare, errors);
			fs::create_directories(case_dir);
			records_csv(case_dir / "errors.csv", errors);
			write_json(case_dir / "errors.json", errors);
			page += "<div class=\"downloads\">" +
					link(batch, case_dir / "errors.csv", "This problem: norms/orders CSV") +
					link(batch, case_dir / "errors.json", "JSON") + "</div>";
			for (auto &[folder, m] : runs) {
				const std::string level = std::to_string(m.at("level").get<int>());
				const auto target = case_dir / ("l" + level);
				page += "<article class=\"run\" id=\"level-" + level + "\"><h3>" + html(mesh_label(m)) +
						" <span class=\"muted\">/ level " + level +
						"</span></h3><details><summary>Recorded setup and run provenance</summary>" +
						settings(m) + "<pre>" + html(m.dump(2)) + "</pre></details>";
				page += "<h4>Final fields and profiles</h4><p>" + html(slice_geometry_note(m)) + "</p>";
				if (d.name == "gaussian_pulse") {
					auto style_file = target / "display.json";
					if (fs::is_regular_file(style_file) &&
						read_json(style_file).value("energy_scale", "") == "log") {
						auto style = read_json(style_file);
						page += "<p class=\"muted\">Gaussian energy is shown as E − Ebg on a logarithmic "
								"scale. Values at or below " +
								show(style.at("energy_floor")) +
								" use the display floor, including zero and negative excesses. Raw values, "
								"signed errors, and error norms are unchanged.</p>";
					} else {
            page += "<p class=\"muted\">Existing energy plots may use a linear scale. Regenerate "
								"plots to apply the logarithmic energy-excess display.</p>";
          }
        }
				if (!fs::is_regular_file(target / "slice_er.png")) {
          page += "<p class=\"muted\">Plots are not yet available. Saved numerical norms and "
							"budgets are listed here.</p>";
        }
        page += figure(batch, target / "slice_er",
							   "Numerical, reference, and signed-error energy slices. " + slice_geometry_note(m));
				page += figure(
					batch, target / "profiles",
					"Numerical/reference line profiles and signed errors for all four radiation fields");
                if (d.name == "marshak") {
                    page += figure(batch, target / "material_profiles", "Normalized radiation and material energies with their semi-analytic diffusion references");
                }
				page += "<details><summary>All three flux maps</summary>";
				for (int f = 1; f < 4; ++f) {
          page += figure(batch, target / ("slice_" + fields[f]),
								   fields[f] + ": numerical, reference, and signed-error slice");
        }
        page += "</details><h4>Movies</h4>" + movies(batch, folder) + "<h4>Conservation</h4>";
				if (auto b = read_conservation(folder, m)) {
					page +=
						budget_table(*b, m) +
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
				} else {
          page += "<p class=\"muted\">Conservation diagnostics were not recorded for this run.</p>";
        }
        page += "<h4>Data and reproducibility</h4><div class=\"downloads\">";
				for (auto &[file, title] : std::vector<std::pair<fs::path, std::string>>{
						 {folder / "run.ini", "Effective configuration"},
						 {folder / "run.json", "Run metadata"},
						 {folder / "run.log", "Solver log"},
						 {target / "slice.csv", "Paired slice values CSV"},
						 {target / "slice.units.json", "Slice units"},
						 {folder / "radiation-conservation.csv", "Raw conservation history"},
                         {folder / "conservation-segments/original-attempt.csv", "Original raw ledger (tail after checkpoint is abandoned)"},
						 {target / "conservation-summary.csv", "Budget summary CSV"},
						 {target / "conservation-fractions.csv", "Conservation fractions CSV"},
						 {target / "conservation-fractions.units.json", "Fraction units"},
						 {target / "conservation.units.json", "Conservation units"},
						 {folder / "L1.dat", "L1 norms"},
						 {folder / "L2.dat", "L2 norms"},
						 {folder / "Linf.dat", "L∞ norms"},
						 {folder / "reference.bin", "Gaussian reference binary"},
						 {folder / "final.silo", "Final numerical Silo"},
						 {folder / "analytic.silo", "Reference Silo"}}) {
          page += link(batch, file, title);
        }
        page += "</div></article>";
			}
			records_csv(case_dir / "conservation.csv", budgets,
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
			write_json(case_dir / "conservation.json", budgets);
			if (!budgets.empty()) {
        page += "<div class=\"downloads\">" +
						link(batch, case_dir / "conservation.csv",
							 "This problem: all conservation summaries CSV") +
						link(batch, case_dir / "conservation.json", "JSON") + "</div>";
      }
    }
		page += "</section>";
        if (!coupled) {
            page += "<p class=\"note\">Momentum plots are omitted for this fixed-material problem. The prescribed medium acts as an external momentum reservoir, so radiation momentum exchange does not imply that a compensating gas momentum was evolved.</p>";
        }
        page += (fld && !coupled) ? R"(<section class="section" id="diagnostics"><h2>Reading FLD diagnostics</h2>
<p>Gaussian, sphere, and Marshak profiles include analytic linear diffusion-limit references; Marshak also includes material energy. Those references omit the nonlinear limiter, and Marshak assumes a semi-infinite domain. Slice discrepancies combine these model differences with discretization and reference-evaluation errors. Thin positive-opacity wave/front cases instead have numerical-only profiles and exact periodic energy-integral checks. Numerical flux is derived as F = −D∇E and satisfies |F| ≤ cE; it is not an evolved M1 momentum variable.</p>
<p>The radiation energy ledger checks R = E(t) − E(t_ref) + B − S, with outward boundary energy B and radiation sources S, including material exchange and any prescribed emitter. Its fixed normalization is initial radiation energy, or bath energy times domain volume for cold Marshak heating. Coupled gas-plus-radiation errors per accepted implicit step are recorded separately in the solver diagnostics.</p>
<p>Nonlinear residuals, substeps, and retries are measured solver diagnostics. The short pilot checks the implementation and reporting pipeline; comparison of models is distinct from an accuracy benchmark or speedup at matched error.</p></section>)" : coupled ? R"(<section class="section" id="diagnostics"><h2>Reading the numerical profiles</h2>
<p>Gas and radiation temperatures are distinct diagnostics; the radiation temperature is defined by E = a<sub>r</sub>T<sub>r</sub><sup>4</sup>. The central ray lies at y = z = Δx/2 and is offset from the exact origin by half a cell in each transverse direction. The two-dimensional plane is z = Δx/2.</p>
<p>Check positivity, the recorded-frame radiation flux envelope, conservation with the appropriate boundary budget, thermal equilibration, shock speed and precursor structure, and sensitivity to resolution and timestep. A finite, positive short run is an implementation check, not evidence that a subcritical or supercritical state has developed. Comparisons to published figures also require matching geometry, opacities, boundaries, and physical time.</p></section>)" : method_notes();
		page += "<section class=\"section\" id=\"sources\"><span class=\"tag\">Sources &amp; "
				"provenance</span><h2>Where this problem comes from</h2>" +
				d.source;
		page +=
			R"(<h3>Radiation method context</h3><p><strong>M1 RK1</strong> evolves radiation energy and flux moments with the retained first-order radiation update. <strong>FLD</strong> uses backward-Euler radiation diffusion/material exchange with the Levermore–Pomraning limiter. Moving cases add first-order conservative gas transport and radiation force/work; their reconstructed lab-frame flux includes (E I+P) v as well as −D∇E. The scalar implicit transport solve uses composite-grid multigrid and an iterative linear solver. Both time integrators are first order.</p>
<p>The suite includes Gaussian scattering, Marshak heating, a Gaussian emitter, explicit thin positive-opacity wave/front comparisons, and moving-gas shock/blast problems on CPU at physical light speed. A configured case does not establish that a solve converged; only validated completed runs appear as results. Flux diagnostics in FLD have no separate M1 momentum conservation test. A difference between paired M1 and FLD solutions is a model comparison, not an analytic error. ADER2 has been removed from the current solver; historical result files retain their original provenance.</p>
<p><a href="https://arxiv.org/abs/1306.0010">Skinner &amp; Ostriker (2013)</a> and <a href="https://doi.org/10.1016/j.jqsrt.2014.04.014">Hanawa &amp; Audit (2014)</a> provide context for the M1 equations. The implementation and saved executable/source hashes identify the actual numerical methods used in this batch.</p>
<h3>Reproducing this batch</h3><p>Descriptions document the reviewed source implementation. Older or edited solver versions can use different profile constants or averaging rules; these are not all encoded in <code>run.ini</code>. Use the run metadata, executable hash, and saved batch source hashes to identify the implementation behind a result. A source path or descriptive formula alone is not proof of the executable's contents.</p>)";
		page += "<div class=\"downloads\">" +
				link(batch, batch / "batch.json", "Batch settings and source hashes") +
				link(batch, batch / "live.json", "Batch status") + "</div></section>" + finish(state);
		atomic_text(batch / (d.name + ".html"), page);
	}
	index += "</div><div class=\"downloads\">" +
			 link(batch, batch / "plots/index.html", "Combined plot report") +
			 link(batch, batch / "movies.html", "Movie gallery") +
			 (fld_batch ? std::string() : link(batch, batch / "plots/errors.csv", "All norms CSV")) +
			 link(batch, batch / "batch.json", "Batch metadata") + "</div>" + finish(state);
	atomic_text(batch / "index.html", index);
	return batch / "index.html";
}
} // namespace rr
