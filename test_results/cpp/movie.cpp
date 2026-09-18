#include "silo_reader.hpp"
#include <silo.h>
namespace rr
{
namespace
{
using File = std::unique_ptr<DBfile, decltype(&DBClose)>;
using Var = std::unique_ptr<DBquadvar, decltype(&DBFreeQuadvar)>;
using Mesh = std::unique_ptr<DBquadmesh, decltype(&DBFreeQuadmesh)>;
using Multi = std::unique_ptr<DBmultivar, decltype(&DBFreeMultivar)>;
File open_silo(const fs::path &p)
{
	File f(DBOpen(p.c_str(), DB_UNKNOWN, DB_READ), DBClose);
	require(bool(f), "Cannot open Silo: " + p.string());
	return f;
}
double scalar(void *p, int datatype, std::size_t i)
{
	require(p, "Missing Silo array");
	double v;
	if (datatype == DB_DOUBLE)
		v = static_cast<double *>(p)[i];
	else if (datatype == DB_FLOAT)
		v = static_cast<float *>(p)[i];
	else
		throw std::runtime_error("Unsupported Silo scalar type");
	require(std::isfinite(v), "Nonfinite Silo value");
	return v;
}
std::vector<std::pair<fs::path, std::string>> blocks(DBfile *root, const fs::path &file,
													 const std::string &field)
{
	std::vector<std::pair<fs::path, std::string>> out;
	int type = DBInqVarType(root, field.c_str());
	if (type == DB_QUADVAR) {
		out.emplace_back(file, field);
		return out;
	}
	require(type == DB_MULTIVAR, "Expected Silo quadvar/multivar: " + field);
	Multi mv(DBGetMultivar(root, field.c_str()), DBFreeMultivar);
	require(bool(mv) && mv->nvars > 0 && mv->varnames, "Missing explicit multivar block names");
	for (int i = 0; i < mv->nvars; ++i) {
		std::string s = mv->varnames[i];
		if (s == "EMPTY")
			continue;
		auto split = s.find(':');
		if (split == s.npos)
			out.emplace_back(file, s);
		else
			out.emplace_back(rr::absolute(file.parent_path() / s.substr(0, split)), s.substr(split + 1));
	}
	require(!out.empty(), "Empty Silo multivar");
	return out;
}
std::size_t offset(const DBquadvar &v, int i, int j, int k)
{
	// Silo supplies strides in elements, independent of array-major ordering.
	require(v.stride[0] > 0 && v.stride[1] > 0 && v.stride[2] > 0, "Invalid Silo strides");
	auto n = std::size_t(i) * v.stride[0] + std::size_t(j) * v.stride[1] + std::size_t(k) * v.stride[2];
	require(n < std::size_t(v.nels), "Silo index out of bounds");
	return n;
}
std::string palette(const std::string &color)
{
	if (color == "hot")
		return "set palette defined (0 '#000000', 0.33 '#d40000', 0.66 '#ffff00', 1 '#ffffff')\n";
	if (color == "viridis")
		return "set palette defined (0 '#440154', 0.25 '#3b528b', 0.5 '#21918c', 0.75 '#5ec962', 1 "
			   "'#fde725')\n";
	if (color == "gray" || color == "grey")
		return "set palette gray\n";
	if (color == "RdBu_r" || color == "diverging")
		return "set palette defined (0 '#2166ac', 0.5 '#f7f7f7', 1 '#b2182b')\n";
	throw std::runtime_error("Unknown palette; choose hot, viridis, gray or diverging");
}
Json input_signature(const std::vector<fs::path> &files, const Options &o, const Json &m)
{
	Json entries = Json::array();
	auto add = [&](const fs::path &p) {
		entries.push_back({{"path", p.string()},
						   {"size", fs::file_size(p)},
						   {"mtime", fs::last_write_time(p).time_since_epoch().count()}});
	};
	for (auto &p : files) {
		add(p);
		auto dir = fs::path(p.string() + ".data");
		if (fs::is_directory(dir)) {
			std::vector<fs::path> data;
			for (auto &e : fs::recursive_directory_iterator(dir))
				if (e.is_regular_file())
					data.push_back(e.path());
			std::sort(data.begin(), data.end());
			for (auto &d : data)
				add(d);
		}
	}
	return {
		{"inputs", entries},
		{"renderer", sha256("/proc/self/exe")},
		{"gnuplot", executable(o.gnuplot).string()},
		{"field", o.field},
		{"view", o.view},
		{"axis", o.axis},
		{"position", o.position},
		{"width", o.width},
		{"height", o.height},
		{"color", o.color},
		{"minimum", o.minimum ? Json(*o.minimum) : Json()},
		{"maximum", o.maximum ? Json(*o.maximum) : Json()},
		{"units", m.value("units", Json::object())},
		{"length", m.at("length")},
		{"time", m.at("time")},
		{"minimum_snapshots",
		 o.allow_sparse
			 ? 2
			 : std::max(2, m.value("movie_capture", Json::object()).value("requested_snapshots", 3) - 1)}};
}
std::string frame_name(std::size_t i)
{
	std::ostringstream s;
	s << "frame_" << std::setfill('0') << std::setw(6) << i;
	return s.str();
}
} // namespace
Snapshot read_silo(const fs::path &p, const Options &o, const Json &m)
{
	auto root = open_silo(p);
	auto names = blocks(root.get(), p, o.field == "fluxmag" ? "fx" : o.field);
	Snapshot out;
	bool have_time = false;
	int axis = o.axis == "x" ? 0 : o.axis == "y" ? 1 : 2;
	double half = m.at("length").get<double>() / 2;
	std::map<fs::path, File> opened;
	for (auto &[file, name] : names) {
		check_interrupt();
		auto it = opened.find(file);
		if (it == opened.end())
			it = opened.emplace(file, open_silo(file)).first;
		auto db = it->second.get();
		Var v(DBGetQuadvar(db, name.c_str()), DBFreeQuadvar);
		require(bool(v) && v->ndims == 3 && v->nvals == 1 && v->centering == DB_ZONECENT,
				"Expected 3D zone-centered scalar Silo variable");
		if (!have_time) {
			out.time = v->dtime;
			have_time = true;
		}
		require(close(v->dtime, out.time) && std::isfinite(out.time) && out.time >= 0,
				"Inconsistent Silo block times");
		fs::path var_dir = fs::path(name).parent_path();
		fs::path mesh_name = v->meshname;
		require(mesh_name.string().find(':') == std::string::npos,
				"External mesh references are unsupported");
		if (!mesh_name.is_absolute())
			mesh_name = var_dir / mesh_name;
		Mesh mesh(DBGetQuadmesh(db, mesh_name.c_str()), DBFreeQuadmesh);
		require(bool(mesh) && mesh->coordtype == DB_COLLINEAR && mesh->ndims == 3,
				"Expected a 3D rectilinear Silo mesh");
		std::array<std::vector<double>, 3> coords;
		for (int d = 0; d < 3; ++d) {
			require(v->dims[d] > 0 && mesh->dims[d] == v->dims[d] + 1,
					"Silo mesh/variable dimensions differ");
			for (int i = 0; i < mesh->dims[d]; ++i) {
				double x = scalar(mesh->coords[d], mesh->datatype, i);
				require(!i || x > coords[d].back(), "Silo coordinates must increase");
				coords[d].push_back(x);
			}
		}
		std::array<int, 3> first{0, 0, 0}, last{v->dims[0], v->dims[1], v->dims[2]};
		if (o.view == "slice") {
			auto &x = coords[axis];
			if (o.position < x.front() || o.position >= x.back()) {
				if (!(o.position == half && x.back() == half))
					continue;
			}
			int index = std::min(int(std::upper_bound(x.begin(), x.end(), o.position) - x.begin()) - 1,
								 v->dims[axis] - 1);
			if (index < 0)
				continue;
			first[axis] = index;
			last[axis] = index + 1;
		}
		std::vector<Var> extra;
		if (o.field == "fluxmag")
			for (auto field : {"fy", "fz"}) {
				auto other = (var_dir / field).string();
				Var w(DBGetQuadvar(db, other.c_str()), DBFreeQuadvar);
				require(bool(w) && w->ndims == 3 && w->nvals == 1 && w->centering == DB_ZONECENT &&
							close(w->dtime, out.time),
						"Invalid flux component");
				for (int d = 0; d < 3; ++d)
					require(w->dims[d] == v->dims[d], "Flux dimensions differ");
				extra.push_back(std::move(w));
			}
		for (int k = first[2]; k < last[2]; ++k)
			for (int j = first[1]; j < last[1]; ++j)
				for (int i = first[0]; i < last[0]; ++i) {
					std::array<int, 3> index{i, j, k};
					Cell cell;
					bool exterior = false;
					for (int d = 0; d < 3; ++d) {
						cell.lo[d] = coords[d][index[d]];
						cell.hi[d] = coords[d][index[d] + 1];
						exterior |= close(cell.lo[d], -half) || close(cell.hi[d], half);
					}
					if (o.view == "3d" && !exterior)
						continue;
					cell.value = scalar(v->vals[0], v->datatype, offset(*v, i, j, k));
					if (o.field == "fluxmag")
						cell.value = std::hypot(
							cell.value,
							scalar(extra[0]->vals[0], extra[0]->datatype, offset(*extra[0], i, j, k)),
							scalar(extra[1]->vals[0], extra[1]->datatype, offset(*extra[1], i, j, k)));
					out.cells.push_back(cell);
				}
	}
	require(!out.cells.empty(), "Slice does not intersect any numerical cells");
	if (o.view == "slice") {
		int a = (axis + 1) % 3, b = (axis + 2) % 3;
		if (axis == 1) {
			a = 0;
			b = 2;
		}
		if (axis == 2) {
			a = 0;
			b = 1;
		}
		int n = m.at("cells");
		double dx = m.at("dx");
		std::set<std::pair<long long, long long>> indices;
		for (auto &cell : out.cells) {
			auto i = std::llround((cell.lo[a] + half) / dx), j = std::llround((cell.lo[b] + half) / dx);
			require(i >= 0 && i < n && j >= 0 && j < n && close(cell.hi[a] - cell.lo[a], dx) &&
						close(cell.hi[b] - cell.lo[b], dx) &&
						close(cell.lo[a], -half + i * dx, 0, dx * 1e-10) &&
						close(cell.lo[b], -half + j * dx, 0, dx * 1e-10),
					"Movie slice differs from uniform run grid");
			require(indices.emplace(i, j).second, "Overlapping movie slice cells");
		}
		require(indices.size() == std::size_t(n) * n, "Movie slice has missing cells");
	}
	return out;
}
void check_dependencies(const Options &o, bool movie)
{
	executable(o.gnuplot);
	palette(o.color);
	if (!movie)
		return;
	executable(o.ffmpeg);
	// Check codec support before spending time running the simulation.
	auto tmp = fs::temp_directory_path() / ("rr-encoders-" + stamp() + ".log");
	try {
		execute({o.ffmpeg, "-hide_banner", "-encoders"}, fs::current_path(), tmp);
		require(read_text(tmp).find("libx264") != std::string::npos, "FFmpeg needs libx264");
		fs::remove(tmp);
	} catch (...) {
		std::error_code e;
		fs::remove(tmp, e);
		throw;
	}
}
Json make_movie(const fs::path &folder, const Options &o)
{
	auto m = read_json(folder / "run.json");
	const bool logarithmic = m.at("case") == "gaussian_pulse" && o.field == "er";
	const double background = logarithmic ? m.at("background").get<double>() : 0;
	require(m.at("status") == "complete", "Simulation is incomplete");
	read_norms(folder, m);
	frame_schedule({0, 1}, o.seconds, o.fps, o.hold);
	require(o.width >= 128 && o.height >= 128, "Movie dimensions must be at least 128");
	if (o.minimum && o.maximum)
		require(*o.minimum < *o.maximum, "Invalid movie color range");
	auto files = numerical_silos(folder);
	auto signature = input_signature(files, o, m);
	auto key = o.field + "-" + o.view + (o.view == "slice" ? "-" + o.axis : "");
	auto destination = folder / "movies" / key;
	fs::create_directories(destination);
	Json rendered;
	fs::path frames_json;
	if (o.reuse_frames) {
		auto saved = read_json(destination / "render.json");
		require(saved.at("signature") == signature,
				"Saved frames have different inputs/style; omit --reuse-frames");
		frames_json = saved.at("frames_json").get<std::string>();
		rendered = read_json(frames_json);
	} else {
		auto frames_dir = destination / "renders" / stamp();
		fs::create_directories(frames_dir);
		Json frames = Json::array();
		double low = INFINITY, high = -INFINITY;
		int axis = o.axis == "x" ? 0 : o.axis == "y" ? 1 : 2, a = axis == 0 ? 1 : 0, b = axis == 2 ? 1 : 2;
		double half = m.at("length").get<double>() / 2;
		// Export once, then use one global color range for every frame.
		for (std::size_t i = 0; i < files.size(); ++i) {
			auto snapshot = read_silo(files[i], o, m);
			if (!frames.empty()) {
				double prev = frames.back().at("time");
				if (files[i].filename() == "final.silo" && close(snapshot.time, prev))
					frames.erase(frames.size() - 1);
				else
					require(snapshot.time > prev, "Snapshot times must strictly increase");
			}
			std::ostringstream data;
			data << std::scientific << std::setprecision(17);
			for (auto &cell : snapshot.cells) {
				// Keep exported samples physical. Background subtraction and the floor
				// are display expressions, never changes to the numerical data.
				low = std::min(low, cell.value - background);
				high = std::max(high, cell.value - background);
				if (o.view == "slice")
					data << (cell.lo[a] + cell.hi[a]) / 2 << ' ' << (cell.lo[b] + cell.hi[b]) / 2 << ' '
						 << cell.lo[a] << ' ' << cell.hi[a] << ' ' << cell.lo[b] << ' ' << cell.hi[b] << ' '
						 << cell.value << '\n';
				else
					for (int d = 0; d < 3; ++d)
						for (int side = 0; side < 2; ++side)
							if (close(side ? cell.hi[d] : cell.lo[d], side ? half : -half)) {
								int u = (d + 1) % 3, v = (d + 2) % 3;
								for (auto uv : std::array<std::array<int, 2>, 5>{
										 {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}}) {
									std::array<double, 3> x;
									x[d] = side ? cell.hi[d] : cell.lo[d];
									x[u] = uv[0] ? cell.hi[u] : cell.lo[u];
									x[v] = uv[1] ? cell.hi[v] : cell.lo[v];
									data << x[0] << ' ' << x[1] << ' ' << x[2] << ' ' << cell.value << '\n';
								}
								data << '\n';
							}
			}
			auto dat = frames_dir / (frame_name(i) + ".dat");
			atomic_text(dat, data.str());
			frames.push_back({{"time", snapshot.time},
							  {"silo", files[i].string()},
							  {"data", dat.string()},
							  {"image", (frames_dir / (frame_name(i) + ".png")).string()}});
			std::cout << "Exported " << i + 1 << '/' << files.size() << " snapshots\n";
		}
		require(frames.size() >= 2, "Need two distinct snapshot times");
		require(close(frames.front().at("time"), 0, 0, 1e-10 * m.at("time").get<double>()) &&
					close(frames.back().at("time"), m.at("time"), 1e-9, 1e-12),
				"Snapshot initial/final time mismatch");
		require(frames.size() >= signature.at("minimum_snapshots").get<std::size_t>(),
				"Too few snapshots; check cadence or use --allow-sparse");
		if (logarithmic) {
			high = high > 0 ? high : std::max(std::abs(background), 1.) * 1e-12;
			if (o.maximum)
				high = *o.maximum;
			low = high * 1e-6;
		}
		if (o.minimum)
			low = *o.minimum;
		if (o.maximum)
			high = *o.maximum;
		if (!logarithmic && low == high) {
			double pad = std::max(std::abs(low) * 1e-8, 1e-30);
			low -= pad;
			high += pad;
		}
		require(low < high, "Invalid color limits");
		require(!logarithmic || low > 0, "Gaussian log display requires positive energy-excess color limits");
		std::string units = cgs(m) ? (o.field == "er" ? "erg/cm^3" : "erg/(cm^2 s)") : "Silo units";
		std::ostringstream floor_label;
		floor_label << "E-Ebg <= " << std::scientific << std::setprecision(3) << low
					<< " (including <=0) uses the floor color";
		std::string script =
			"set terminal pngcairo size " + std::to_string(o.width) + "," + std::to_string(o.height) +
			" noenhanced font 'Sans,12'\nset encoding utf8\nunset key\nset format x '%.2e'\nset format y "
			"'%.2e'\nset format cb '%.2e'\nset cblabel " +
			gp_quote(o.field + (logarithmic ? " - background; log (" : " (") + units + ")") + "\n" +
			(logarithmic
				 ? "set logscale cb\nclip_energy(v)=(v-" + number(background) + ">" + number(low) + "?v-" +
					   number(background) + ":" + number(low) + ")\nset label 1 " +
					   gp_quote(floor_label.str()) + " at screen 0.5,0.025 center font ',9'\nset bmargin 5\n"
				 : "") +
			palette(o.color) + "set cbrange [" + number(low) + ":" + number(high) + "]\nset xrange [" +
			number(-half) + ":" + number(half) + "]\nset yrange [" + number(-half) + ":" + number(half) +
			"]\nset xtics " + number(half) + "\nset ytics " + number(half) + "\nset xlabel " +
			gp_quote((o.view == "slice" ? std::string(1, "xyz"[a]) : "x") + (cgs(m) ? " (cm)" : "")) +
			"\nset ylabel " +
			gp_quote((o.view == "slice" ? std::string(1, "xyz"[b]) : "y") + (cgs(m) ? " (cm)" : "")) + "\n";
		if (o.view == "slice")
			script += "set size ratio -1\nset style fill solid 1 noborder\n";
		else
			script += "set zrange [" + number(-half) + ":" + number(half) +
					  "]\nset zlabel 'z'\nset view 65,35,1,1\nset view equal xyz\nset pm3d depthorder\n";
		for (auto &f : frames) {
			script +=
				"set output " + gp_quote(f.at("image")) + "\nset title " +
				gp_quote(
					m.at("case").get<std::string>() + " | " + std::to_string(m.at("cells").get<int>()) +
					"^3 | t=" + number(f.at("time")) + (cgs(m) ? " s" : "") +
					(o.view == "slice" ? " | " + o.axis + "=" + number(o.position) : " | exterior surface")) +
				"\n" + (o.view == "slice" ? "plot " : "splot ") + gp_quote(f.at("data")) +
				(o.view == "slice"
					 ? std::string(" using 1:2:3:4:5:6:") + (logarithmic ? "(clip_energy($7))" : "7") +
						   " with boxxyerror lc palette\n"
					 : std::string(" using 1:2:3:") + (logarithmic ? "(clip_energy($4))" : "4") +
						   " with polygons fc palette\n") +
				"unset output\n";
		}
		gnuplot_script(frames_dir / "frames.gnuplot", script, o);
		for (auto &f : frames)
			require(fs::is_regular_file(f.at("image").get<std::string>()) &&
						fs::file_size(f.at("image").get<std::string>()) > 0,
					"Missing gnuplot frame");
		rendered = {{"status", "complete"},
					{"color_scale", logarithmic ? "log" : "linear"},
					{"background_subtracted", background},
					{"frames", frames},
					{"color_limits", {low, high}},
					{"time_units", cgs(m) ? "s" : "Silo time"},
					{"field_units", units}};
		frames_json = frames_dir / "frames.json";
		write_json(frames_json, rendered);
		write_json(destination / "render.json",
				   {{"signature", signature}, {"frames_json", frames_json.string()}});
	}
	require(rendered.at("status") == "complete", "Incomplete frame manifest");
	std::vector<double> times;
	for (auto &f : rendered.at("frames")) {
		times.push_back(f.at("time"));
		require(fs::is_regular_file(f.at("image").get<std::string>()) &&
					fs::file_size(f.at("image").get<std::string>()) > 0,
				"Missing cached frame");
	}
	auto sequence = frame_schedule(times, o.seconds, o.fps, o.hold);
	auto temp = destination / ("encode-" + stamp());
	fs::create_directory(temp);
	auto output = destination / "movie.mp4";
	try {
		for (std::size_t i = 0; i < sequence.size(); ++i) {
			fs::path src = rendered["frames"][sequence[i]]["image"].get<std::string>();
			auto dest = temp / (frame_name(i) + ".png");
			std::error_code ec;
			fs::create_hard_link(src, dest, ec);
			if (ec)
				fs::create_symlink(rr::absolute(src), dest);
		}
		std::size_t encoded = 0;
		std::string speed = "N/A";
		auto progress = [&](const std::string &line) {
			auto s = trim(line);
			if (s.starts_with("frame="))
				encoded = std::stoull(trim(s.substr(6)));
			else if (s.starts_with("speed="))
				speed = trim(s.substr(6));
			else if (s.starts_with("progress="))
				std::cout << "Encoding " << encoded << '/' << sequence.size() << " frames ("
						  << std::min(std::size_t(100), 100 * encoded / sequence.size()) << "%) | speed "
						  << speed << std::endl;
			else if (!s.starts_with("fps=") && !s.starts_with("stream_") && !s.starts_with("bitrate=") &&
					 !s.starts_with("total_size=") && !s.starts_with("out_time") &&
					 !s.starts_with("dup_frames=") && !s.starts_with("drop_frames="))
				std::cerr << line << std::flush;
		};
		std::cout << "Encoding " << sequence.size() << " frames to MP4..." << std::endl;
		execute({o.ffmpeg,
				 "-hide_banner",
				 "-nostdin",
				 "-nostats",
				 "-stats_period",
				 "1",
				 "-progress",
				 "pipe:1",
				 "-loglevel",
				 "warning",
				 "-y",
				 "-framerate",
				 std::to_string(o.fps),
				 "-start_number",
				 "0",
				 "-i",
				 (temp / "frame_%06d.png").string(),
				 "-frames:v",
				 std::to_string(sequence.size()),
				 "-an",
				 "-c:v",
				 "libx264",
				 "-crf",
				 "18",
				 "-preset",
				 "medium",
				 "-pix_fmt",
				 "yuv420p",
				 "-vf",
				 "pad=ceil(iw/2)*2:ceil(ih/2)*2",
				 "-movflags",
				 "+faststart",
				 (temp / "movie.mp4").string()},
				folder, destination / "encode.log", progress, true, false);
		require(fs::is_regular_file(temp / "movie.mp4") && fs::file_size(temp / "movie.mp4") > 0,
				"FFmpeg produced no movie");
		fs::rename(temp / "movie.mp4", output);
		fs::remove_all(temp);
	} catch (...) {
		std::error_code ec;
		fs::remove_all(temp, ec);
		throw;
	}
	std::set<std::size_t> displayed(sequence.begin(), sequence.end());
	Json result{{"path", output.string()},
				{"color_scale", rendered.value("color_scale", "linear")},
				{"background_subtracted", rendered.value("background_subtracted", 0.)},
				{"seconds", double(sequence.size()) / o.fps},
				{"fps", o.fps},
				{"video_frames", sequence.size()},
				{"source_snapshots", times.size()},
				{"displayed_snapshots", displayed.size()},
				{"endpoint_hold_seconds", double(std::llround(o.hold * o.fps)) / o.fps},
				{"timing", "Actual Silo time gaps; recorded states held without interpolation"},
				{"status", "complete"},
				{"case", m.at("case")},
				{"cells", m.at("cells")},
				{"field", o.field},
				{"view", o.view},
				{"axis", o.axis},
				{"frames_json", frames_json.string()},
				{"simulation_directory", folder.string()},
				{"color_limits", rendered.at("color_limits")},
				{"time_units", rendered.at("time_units")},
				{"field_units", rendered.at("field_units")},
				{"units", m.value("units", Json::object())}};
	write_json(destination / "movie.json", result);
	return result;
}
void movie_index(const fs::path &batch)
{
	std::string page = page_start("Octo-TIGER radiation movies") +
					   "<p>Numerical Silo states; fixed camera and color range. Actual time spacing is "
					   "retained without interpolating solutions. 3D views show the exterior surface.</p>";
	std::vector<fs::path> paths;
	for (auto &e : fs::recursive_directory_iterator(batch))
		if (e.is_regular_file() && e.path().filename() == "movie.json")
			paths.push_back(e.path());
	std::sort(paths.begin(), paths.end());
	for (auto &p : paths) {
		auto m = read_json(p);
		if (m.value("status", "") != "complete")
			continue;
		auto file = p.parent_path() / "movie.mp4";
		if (!fs::is_regular_file(file))
			continue;
		auto rel = url(fs::relative(file, batch).generic_string());
		page += "<section><h2>" + html(m.at("case")) + " | " + std::to_string(m.at("cells").get<int>()) +
				"³ | " + html(m.at("field")) + "</h2><p>" + number(m.at("seconds")) + " seconds; " +
				std::to_string(m.at("source_snapshots").get<int>()) + " stored snapshots. <a href=\"" + rel +
				"\">MP4</a></p>";
		if (m.value("color_scale", "linear") == "log")
			page += "<p>Logarithmic energy excess E − Ebg. Values at or below " +
					number(m.at("color_limits")[0]) + " " + html(m.at("field_units")) +
					" (including nonpositive values) use the floor color.</p>";
		page += "<video controls preload=\"metadata\" src=\"" + rel + "\"></video></section>";
	}
	atomic_text(batch / "movies.html", page + "</body></html>\n");
	report_pages(batch);
}
} // namespace rr
