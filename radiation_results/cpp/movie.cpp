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
		{"backend", "VisIt-session-3.4.2"},
		{"visit", visit_executable(o).string()},
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
		{"case", m.at("case")},
		{"cells", m.at("cells")},
		{"background", m.at("background")},
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
	if (o.command != "movies")
		executable(o.gnuplot);
	if (!movie)
		return;
	visit_executable(o);
	executable(o.ffmpeg);
	// Check codec support before spending time running the simulation.
	auto tmp = fs::temp_directory_path() / ("rr-encoders-" + stamp() + ".log");
	try {
		execute({o.ffmpeg, "-hide_banner", "-encoders"}, fs::current_path(), tmp, {}, true, false);
		require(read_text(tmp).find("libx264") != std::string::npos, "FFmpeg needs libx264");
		fs::remove(tmp);
	} catch (...) {
		std::error_code e;
		fs::remove(tmp, e);
		throw;
	}
}
Json make_session(const fs::path &folder, const Options &o)
{
	auto m = read_json(folder / "run.json");
	require(m.at("status") == "complete", "Simulation is incomplete");
	read_norms(folder, m);
	const bool logarithmic = m.at("case") == "gaussian_pulse" && o.field == "er";
	const double background = logarithmic ? m.at("background").get<double>() : 0;
	require(o.width >= 128 && o.height >= 128, "Movie dimensions must be at least 128");
	require(std::set<std::string>{"hot", "viridis", "gray", "grey", "diverging", "RdBu_r"}.contains(o.color),
			"Unknown color table; choose hot, viridis, gray or diverging");
	auto files = numerical_silos(folder);
	Json frames = Json::array();
	double low = INFINITY, high = -INFINITY;
	for (std::size_t i = 0; i < files.size(); ++i) {
		auto snapshot = read_silo(files[i], o, m);
		if (!frames.empty()) {
			double previous = frames.back().at("time");
			if (files[i].filename() == "final.silo" && close(snapshot.time, previous))
				frames.erase(frames.size() - 1);
			else
				require(snapshot.time > previous, "Snapshot times must strictly increase");
		}
		for (auto &cell : snapshot.cells) {
			low = std::min(low, cell.value - background);
			high = std::max(high, cell.value - background);
		}
		frames.push_back({{"time", snapshot.time}, {"silo", rr::absolute(files[i]).string()}});
		std::cout << "Scanned " << i + 1 << '/' << files.size() << " snapshots for VisIt" << std::endl;
	}
	require(frames.size() >= 2, "Need two distinct snapshot times");
	require(close(frames.front().at("time"), 0, 0, 1e-10 * m.at("time").get<double>()) &&
				close(frames.back().at("time"), m.at("time"), 1e-9, 1e-12),
			"Snapshot initial/final time mismatch");
	const std::size_t minimum =
		o.allow_sparse
			? 2
			: std::max(2, m.value("movie_capture", Json::object()).value("requested_snapshots", 3) - 1);
	require(frames.size() >= minimum, "Too few snapshots; check cadence or use --allow-sparse");
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
	auto units = cgs(m) ? (o.field == "er" ? "erg/cm^3" : "erg/(cm^2 s)") : "Silo units";
	Json rendered{{"backend", "VisIt"},
				  {"color_scale", logarithmic ? "log" : "linear"},
				  {"background_subtracted", background},
				  {"frames", frames},
				  {"color_limits", {low, high}},
				  {"time_units", cgs(m) ? "s" : "Silo time"},
				  {"field_units", units},
				  {"case", m.at("case")},
				  {"cells", m.at("cells")},
				  {"length", m.at("length")},
				  {"field", o.field},
				  {"view", o.view},
				  {"axis", o.axis},
				  {"position", o.position},
				  {"color", o.color},
				  {"width", o.width},
				  {"height", o.height},
				  {"run", rr::absolute(folder).string()}};
	// Stable across rerenders; changes of source path, view, or limits cannot collide.
	auto digest = sha256_text(rendered.dump()).substr(0, 12);
	auto safe = [](std::string s) {
		for (auto &c : s)
			if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
				c = '_';
		return s.substr(0, 64);
	};
	auto name = safe(folder.parent_path().parent_path().filename().string()) + "--" + safe(m.at("case")) +
				"--" + safe(folder.filename().string()) + "-n" + std::to_string(m.at("cells").get<int>()) +
				"--" + o.field + "-" + o.view +
				(o.view == "slice" ? "-" + o.axis + "-at" + safe(number(o.position)) : "") + "--" + digest;
	auto dir =
		rr::absolute(o.session_dir.empty() ? o.root / "radiation_results/visit_sessions" : o.session_dir);
	auto session = dir / (name + ".session"), database = dir / (name + ".visit");
	std::string listing;
	for (auto &f : frames) {
		auto file = f.at("silo").get<std::string>();
		require(file.find_first_of("\r\n") == std::string::npos, "Silo paths cannot contain newlines");
		listing += file + "\n";
	}
	atomic_text(database, listing);
	atomic_text(session, visit_session_xml(database, rendered, o));
	rendered["session_file"] = session.string();
	rendered["database_file"] = database.string();
	write_json(dir / (name + ".json"), rendered);
	std::cout << "VisIt session: " << session << std::endl;
	return rendered;
}
Json make_movie(const fs::path &folder, const Options &o)
{
	auto m = read_json(folder / "run.json");
	frame_schedule({0, 1}, o.seconds, o.fps, o.hold);
	auto files = numerical_silos(folder);
	auto signature = input_signature(files, o, m);
	auto key = o.field + "-" + o.view + (o.view == "slice" ? "-" + o.axis : "");
	auto destination = folder / "movies" / key;
	fs::create_directories(destination);
	// Also recreate missing sessions when reusing previously rendered frames.
	auto session = make_session(folder, o);
	Json rendered;
	fs::path frames_json;
	if (o.reuse_frames) {
		auto saved = read_json(destination / "render.json");
		require(saved.at("signature") == signature,
				"Saved frames have different inputs/style; omit --reuse-frames");
		frames_json = saved.at("frames_json").get<std::string>();
		rendered = read_json(frames_json);
	} else {
		auto frames_dir = rr::absolute(destination / "renders" / stamp());
		fs::create_directories(frames_dir);
		const auto count = session.at("frames").size();
		auto log = frames_dir / "visit.log";
		std::cout << "VisIt rendering " << count << " snapshots..." << std::endl;
		int visit_status = execute({visit_executable(o).string(), "-movie", "-nowin", "-ignoresessionengines",
									"-sessionfile", session.at("session_file"), "-format", "png", "-geometry",
									std::to_string(o.width) + "x" + std::to_string(o.height), "-start", "0",
									"-end", std::to_string(count - 1), "-framestep", "1", "-output",
									(frames_dir / "frame_").string()},
								   frames_dir, log, {}, false);
		auto messages = read_text(log);
		require(messages.find("VisIt: Error") == std::string::npos &&
					messages.find("VisIt could not") == std::string::npos &&
					messages.find("Traceback (most recent call last)") == std::string::npos &&
					messages.find("There was an error when trying to") == std::string::npos &&
					messages.find("SetTimeSliderState was called when there was no time slider") ==
						std::string::npos,
				"VisIt reported a rendering failure; see " + log.string());
		require(visit_status == 0 ||
					(visit_status == 250 &&
					 messages.find("VisIt completed generating frames.") != std::string::npos),
				"VisIt exited with code " + std::to_string(visit_status) + "; see " + log.string());
		rendered = session;
		rendered["visit_exit_code"] = visit_status;
		// VisIt's native -movie uses 4 digits, increasing to 5/6/7 for long sequences.
		int digits = count > 999999 ? 7 : count > 99999 ? 6 : count > 9999 ? 5 : 4;
		for (std::size_t i = 0; i < count; ++i) {
			std::ostringstream name;
			name << "frame_" << std::setfill('0') << std::setw(digits) << i << ".png";
			auto file = frames_dir / name.str();
			require(fs::is_regular_file(file) && fs::file_size(file) > 0,
					"Missing VisIt frame " + file.string() + "; see " + log.string());
			rendered["frames"][i]["image"] = file.string();
		}
		if (visit_status != 0) {
			// Some 3.4.2 binaries abort during shutdown after saving every frame.
			// Never accept partial/error renders: require the native completion marker,
			// all expected images, and a successful full-image decode before proceeding.
			execute({o.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-start_number", "0",
					 "-i", (frames_dir / ("frame_%0" + std::to_string(digits) + "d.png")).string(),
					 "-frames:v", std::to_string(count), "-f", "null", "-"},
					frames_dir, frames_dir / "verify-frames.log");
			std::cerr << "VisIt exited with code " << visit_status << " after reporting completion; all "
					  << count << " saved frames decoded successfully. See " << log << std::endl;
		}
		rendered["status"] = "complete";
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
				 "-xerror",
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
				{"renderer", "VisIt"},
				{"visit_exit_code", rendered.value("visit_exit_code", 0)},
				{"session_file", session.at("session_file")},
				{"database_file", session.at("database_file")},
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
