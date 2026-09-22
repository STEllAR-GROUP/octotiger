#include "common.hpp"
namespace rr
{
namespace
{
void replaceOnce(std::string &s, const std::string &old, const std::string &value)
{
	auto pos = s.find(old);
	require(pos != s.npos && s.find(old, pos + old.size()) == s.npos,
			"Source differs from expected context: " + old);
	s.replace(pos, old.size(), value);
}
std::string stripIncludes(const std::string &source)
{
	std::istringstream s(source);
	std::string line, out;
	while (std::getline(s, line))
		if (!line.starts_with("#include") && !line.starts_with("#pragma once"))
			out += line + '\n';
	return out;
}
std::string method(const std::string &source, const std::string &marker)
{
	auto begin = source.find(marker);
	require(begin != source.npos, "Serial harness cannot find method: " + marker);
	auto end = source.find('{', begin);
	require(end != source.npos, "Missing function body");
	int depth = 1;
	for (++end; end < source.size() && depth; ++end) {
		if (source[end] == '{')
			++depth;
		else if (source[end] == '}')
			--depth;
	}
	require(depth == 0, "Unbalanced function body");
	return source.substr(begin, end - begin);
}
} // namespace
int maintenance(const Options &o)
{
	auto root = o.root;
	auto source = sourceRoot(root);
	auto support = toolsRoot(root) / "support";
	if (o.command == "install") {
		require(fs::is_regular_file(source / "octotiger/test_problems/radiation/reference.hpp"),
				"Install radiation reference update first");
		auto file = source / "src/grid.cpp";
		auto old = readText(file), changed = old;
		if (old.find("RADIATION_PLOT_EXPORT_BEGIN") != old.npos)
			require(old.find("radiationSlice.capture(") != old.npos &&
						old.find("radiationSlice.finish();") != old.npos,
					"Incomplete existing plot hook");
		else {
			auto begin = old.find("analytic_t grid::compute_analytic(Real t) {");
			require(begin != old.npos, "Missing compute_analytic");
			auto end = old.find("\nvoid grid::allocate()", begin);
			require(end != old.npos, "Missing grid::allocate anchor");
			auto part = old.substr(begin, end - begin);
			std::string anchor = "\tconst Real dv = dx * dx * dx;";
			replaceOnce(part, anchor, anchor + R"(
	// RADIATION_PLOT_EXPORT_BEGIN: opt in by creating datadir/radiation-slices.
	radiationTests::SliceOutput radiationSlice(radiationRegressionProblem(),
		opts().data_dir, double(t), double(dx), 2 * double(opts().xscale));)");
			std::string loop = "\t\t\t\tfor (integer field = 0; field != opts().n_fields; ++field) {";
			replaceOnce(part, loop,
						 R"(				radiationSlice.capture(double(X[XDIM][iii]), double(X[YDIM][iii]),
					double(X[ZDIM][iii]), A, opts().n_fields, [&](int f) {
						return double(rad_grid_ptr->get_field(f, i - H_BW + R_BW,
							j - H_BW + R_BW, k - H_BW + R_BW));
					});
)" + loop);
			replaceOnce(part, "\treturn a;", "\tradiationSlice.finish();\n\treturn a;");
			changed = "#include \"octotiger/test_problems/radiation.hpp\"\n#include "
					  "\"octotiger/test_problems/radiation/plot_output.hpp\"\n" +
					  old.substr(0, begin) + part + old.substr(end);
		}
		if (o.check) {
			if (old == changed) {
				std::cout << "Plot hook already installed\n";
				return 0;
			}
			auto tmp = fs::temp_directory_path() / ("rr-grid-" + stamp() + ".cpp");
			atomicText(tmp, changed);
			int code = execute({"diff", "-u", "--label", "a/src/grid.cpp", "--label", "b/src/grid.cpp",
								file.string(), tmp.string()},
							   source, {}, {}, false);
			fs::remove(tmp);
			require(code <= 1, "diff failed");
			return 0;
		}
		auto header = source / "octotiger/test_problems/radiation/plot_output.hpp";
		// An installed hook belongs to the solver; never replace its implementation.
		if (old == changed && fs::is_regular_file(header)) {
			std::cout << "Radiation slice export already installed\n";
			return 0;
		}
		auto supplied = readText(support / "plot_output.hpp");
		require(!fs::exists(header) || readText(header) == supplied,
				"Existing plot_output.hpp differs; no files changed");
		if (old != changed) {
			auto backup = fs::path(file.string() + ".before-radiation-plots");
			require(!fs::exists(backup), "Backup already exists; no files changed");
			fs::copy_file(file, backup);
			atomicText(header, supplied);
			atomicText(file, changed);
		} else if (!fs::exists(header))
			atomicText(header, supplied);
		std::cout << "Radiation slice export installed; rebuild octotiger\n";
		return 0;
	}
	require(!o.reference.empty() && !o.output.empty(), "validate-serial requires --reference and --output");
	require(o.cells >= 4 && o.cells <= 512 && o.cells % 2 == 0, "Invalid serial mesh size");
	auto radiationSource = readText(source / "src/radiation/rad_grid.cpp");
	std::string bodies;
	for (auto marker :
		 {"void rad_grid::allocate()", "void rad_grid::set_dx(", "void rad_grid::set_X(",
		  "Real rad_grid::maxTimestep(", "void rad_grid::compute_flux(", "void rad_grid::advance(",
		  "void rad_grid::sanity_check()", "void rad_grid::applyRegressionSource(",
		  "radiationConservation::Totals rad_grid::takeConservation(", "void rad_grid::accountBoundaryFlux(",
		  "void rad_grid::set_physical_boundaries(", "rad_grid::rad_grid(Real _dx)", "rad_grid::rad_grid()"})
		bodies += method(radiationSource, marker) + "\n";
	auto temp = fs::temp_directory_path() / ("radiation-regression-" + stamp());
	fs::create_directory(temp);
	try {
		auto fixture = "#include " + gpQuote((support / "plot_output.hpp").string()) + "\n" +
					   readText(support / "serial_fixture.inc") +
					   stripIncludes(readText(source / "test_problems/radiation.hpp")) +
					   stripIncludes(readText(source / "src/test_problems/radiation/radiation.cpp")) +
					   stripIncludes(readText(source / "octotiger/radiation/rad_grid.hpp")) + bodies +
					   readText(support / "serial_checks.inc");
		atomicText(temp / "test.cpp", fixture);
		std::vector<std::string> cmd{o.cxx, "-std=c++23"};
		if (o.sanitize)
			cmd.insert(cmd.end(), {"-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"});
		else
			cmd.push_back("-O3");
		cmd.insert(cmd.end(), {"-I" + source.string(), "-DTEST_CELLS=" + std::to_string(o.cells),
							   (temp / "test.cpp").string(), "-o", (temp / "test").string()});
		execute(cmd, source);
		execute(
			{(temp / "test").string(), rr::absolute(o.reference).string(), rr::absolute(o.output).string()},
			source);
		fs::remove_all(temp);
	} catch (...) {
		std::error_code e;
		fs::remove_all(temp, e);
		throw;
	}
	return 0;
}
} // namespace rr
