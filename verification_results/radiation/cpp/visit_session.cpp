#include "common.hpp"
#include <cstdlib>

namespace rr
{
namespace
{
std::string object(const std::string &name, const std::string &body)
{
	return "<Object name=\"" + html(name) + "\">\n" + body + "</Object>\n";
}
std::string field(const std::string &name, const std::string &type, const std::string &value, int length = 0)
{
	return "<Field name=\"" + html(name) + "\" type=\"" + type + "\"" +
		   (length ? " length=\"" + std::to_string(length) + "\"" : "") + ">" +
		   [&] {
			   std::string escaped;
			   for (char c : value) {
				   if (c == '&')
					   escaped += "&amp;";
				   else if (c == '<')
					   escaped += "&lt;";
				   else if (c == '>')
					   escaped += "&gt;";
				   else
					   escaped += c;
			   }
			   return escaped;
		   }() +
		   "</Field>\n";
}
std::string str(const std::string &name, const std::string &value)
{
	// VisIt's DataNode string tokenizer requires quotes around whitespace.
	return field(name, "string", Json(value).dump());
}
std::string real(const std::string &name, double value)
{
	return field(name, "double", number(value));
}
std::string flag(const std::string &name, bool value)
{
	return field(name, "bool", value ? "true" : "false");
}
std::string array(const std::string &name, std::initializer_list<double> values)
{
	std::string s;
	for (double value : values)
		s += number(value) + " ";
	return field(name, "doubleArray", s, values.size());
}
std::string color(const std::string &name, bool white)
{
	return object(name, object("ColorAttribute", field("color", "unsignedCharArray",
													   white ? "255 255 255 255" : "0 0 0 255", 4)));
}
std::string axis(const std::string &name, const std::string &title, bool physical)
{
	auto font = object("font", object("FontAttributes", str("font", "Arial") + flag("bold", false) +
															flag("italic", false) + real("scale", 0.9)));
	return object(
		name, object("AxisAttributes",
					 object("title", object("AxisTitles", flag("userTitle", true) + flag("userUnits", true) +
															  str("title", title) +
															  str("units", physical ? "cm" : "") + font)) +
						 object("label", object("AxisLabels", font))));
}
std::string textLabel(const std::string &name, const std::string &text, double y, double height)
{
	return object("AnnotationObject",
				  str("objectName", name) + str("objectType", "Text2D") + flag("visible", true) +
					  flag("active", false) + array("position", {0.15, y, 0}) +
					  array("position2", {height, 0, 0}) + color("textColor", false) +
					  flag("useForegroundForTextColor", true) +
					  field("text", "stringVector", Json(text).dump()) + str("fontFamily", "Arial"));
}
} // namespace
fs::path visitExecutable(const Options &o)
{
	if (o.visit != "visit" || o.explicitOptions.contains("--visit"))
		return executable(o.visit);
	if (const char *value = std::getenv("VISIT"))
		if (*value)
			return executable(value);
	try {
		return executable("visit");
	} catch (const std::exception &) {
	}
	if (const char *value = std::getenv("HOME")) {
		auto installed = fs::path(value) / "visit3_4_2.linux-x86_64/bin/visit";
		if (fs::is_regular_file(installed))
			return executable(installed.string());
	}
	throw std::runtime_error("VisIt is required for movies; pass --visit /path/to/bin/visit or set VISIT");
}
std::string visitSessionXml(const fs::path &database, const Json &m, const Options &o)
{
	const bool logarithmic = m.at("color_scale") == "log";
	const double low = m.at("color_limits")[0], high = m.at("color_limits")[1];
	const double half = m.at("length").get<double>() / 2;
	const bool physical = m.at("time_units") == "s";
	std::string variable = o.field, expressions;
	if (logarithmic || o.field == "fluxmag") {
		variable = logarithmic ? "radiation_energy_excess" : "radiation_flux_magnitude";
		std::string definition =
			logarithmic ? "max(er-(" + number(m.at("background_subtracted")) + ")," + number(low) + ")"
						: "sqrt(fx*fx+fy*fy+fz*fz)";
		expressions = object("ExpressionList",
							 object("Expression", str("name", variable) + str("definition", definition) +
													  str("type", "ScalarMeshVar") + flag("fromDB", false) +
													  flag("fromOperator", false) + flag("hidden", false)));
	}
	std::string palette = o.color;
	if (palette == "grey")
		palette = "gray";
	if (palette == "RdBu_r" || palette == "diverging")
		palette = "difference";
	auto plot =
		field("cacheIndex", "int", "0") + flag("followsTime", true) + array("bgColor", {1, 1, 1}) +
		array("fgColor", {0, 0, 0}) +
		object("PseudocolorAttributes",
			   str("scaling", logarithmic ? "Log" : "Linear") + flag("minFlag", true) + real("min", low) +
				   flag("maxFlag", true) + real("max", high) + str("centering", "Zonal") +
				   str("limitsMode", "CurrentPlot") + str("colorTableName", palette) +
				   flag("legendFlag", true) + flag("lightingFlag", false));
	if (o.view == "slice") {
		int d = o.axis == "x" ? 0 : o.axis == "y" ? 1 : 2;
		auto slice = str("originType", "Intercept") + real("originIntercept", o.position) +
					 str("axisType", d == 0	  ? "XAxis"
									 : d == 1 ? "YAxis"
											  : "ZAxis") +
					 array("normal", {d == 0 ? 1. : 0., d == 1 ? 1. : 0., d == 2 ? 1. : 0.}) +
					 array("upAxis", {0., d == 2 ? 1. : 0., d == 2 ? 0. : 1.}) + flag("project2d", true) +
					 flag("flip", false);
		plot +=
			object("Operators",
				   field("activeOperatorIndex", "int", "0") +
					   object("operator00", str("operatorType", "Slice_1.0") +
												object("ViewerOperator", object("SliceAttributes", slice))));
	}
	plot = object("ViewerPlotList",
				  field("nPlots", "int", "1") + str("activeSource", "SOURCE00") +
					  flag("keyframeMode", false) + object("timeSliders", field("SOURCE00", "int", "0")) +
					  str("activeTimeSlider", "SOURCE00") +
					  object("plot00", str("plotName", "Plot0000") + str("pluginID", "Pseudocolor_1.0") +
										   str("sourceID", "SOURCE00") + str("variableName", variable) +
										   flag("active", true) + flag("hidden", false) +
										   flag("realized", true) + object("ViewerPlot", plot)));
	auto annotations =
		object("axes2D",
			   object("Axes2D", flag("visible", true) + axis("xAxis", o.axis == "x" ? "y" : "x", physical) +
									axis("yAxis", o.axis == "z" ? "y" : "z", physical))) +
		object("axes3D", object("Axes3D", flag("visible", false) + flag("triadFlag", true) +
											  flag("bboxFlag", true) + axis("xAxis", "x", physical) +
											  axis("yAxis", "y", physical) + axis("zAxis", "z", physical))) +
		flag("userInfoFlag", false) + flag("databaseInfoFlag", false) + flag("timeInfoFlag", false) +
		color("backgroundColor", true) + color("foregroundColor", false) + str("backgroundMode", "Solid");
	auto title = m.at("case").get<std::string>() + " | " + std::to_string(m.at("cells").get<int>()) +
				 "^3 | " + o.field + " | t=$time" + (physical ? " s" : "") +
				 (o.view == "slice" ? " | " + o.axis + "=" + number(o.position) : " | exterior surface");
	auto labels =
		textLabel("RadiationTitle", title, 0.96, 0.024) +
		textLabel(
			"RadiationUnits",
			(logarithmic ? "E - background; log scale" : o.field) + std::string(" (") +
				m.at("field_units").get<std::string>() + ")" +
				(o.view == "3d" ? " | domain side " + number(m.at("length")) + (physical ? " cm" : "") : ""),
			0.055, 0.020);
	if (logarithmic) {
		std::ostringstream floor;
		floor << "E - background <= " << std::scientific << std::setprecision(3) << low
			  << " (including <= 0) uses the floor color";
		labels += textLabel("RadiationFloor", floor.str(), 0.023, 0.016);
	}
	auto view =
		object("View2DAttributes", array("windowCoords", {-half, half, -half, half}) +
									   array("viewportCoords", {0.20, 0.95, 0.15, 0.89}) +
									   str("fullFrameActivationMode", "Off") + flag("windowValid", true)) +
		object("View3DAttributes", array("viewNormal", {0.55, -0.7, 0.45}) + array("focus", {0, 0, 0}) +
									   array("viewUp", {0, 0, 1}) + real("parallelScale", half * 2.1) +
									   real("nearPlane", -half * 4) + real("farPlane", half * 4) +
									   flag("perspective", false) + flag("windowValid", true));
	std::string dimensions = std::to_string(o.width) + " " + std::to_string(o.height);
	auto window =
		object("ViewerWindow", field("windowSize", "intArray", dimensions, 2) +
								   field("windowImageSize", "intArray", dimensions, 2) +
								   flag("maintainView", true) + object("AnnotationAttributes", annotations) +
								   view + object("AnnotationObjectList", labels) + plot);
	auto source = "localhost:" + rr::absolute(database).string();
	auto subject =
		object("SourceMap", str("SOURCE00", source)) + object("SourcePlugins", str(source, "Silo_1.0")) +
		object("ViewerWindowManager", field("activeWindow", "int", "0") + object("Windows", window));
	return "<?xml version=\"1.0\"?>\n" +
		   object("VisIt",
				  str("Version", "3.4.2") +
					  object("VIEWER",
							 object("DEFAULT_VALUES",
									expressions +
										object("SaveWindowAttributes",
											   field("pixelData", "int", "1") + flag("screenCapture", false) +
												   flag("stereo", false) + flag("saveTiled", false) +
												   flag("advancedMultiWindowSave", false))) +
								 object("ViewerSubject", subject)));
}
} // namespace rr
