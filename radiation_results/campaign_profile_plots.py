"""VisIt-native, synchronized shock temperature profiles for campaign movies.

The embedded script is joined to campaign_movies.VISIT_SCRIPT before execution
in VisIt's Python interpreter. It uses native mesh intersections, not a uniform
sample count or interpolated movie states. The exported gas-temperature ``T``
expression implements the solver's composition, EOS, and dual-energy choice
(src/grid.cpp, grid::get_scalar_expressions). Radiation temperature uses the
same CGS radiation constant as src/physcon.cpp and src/radiation/rad_grid.cpp.
"""

SHOCK_CASES = frozenset({"subcritical_shock", "supercritical_shock"})


# Globals supplied by the movie renderer: request, metadata, database, half,
# slice_z, checked, check_stop, text_label, and the VisIt Python API.
PROFILE_SCRIPT = r'''
def _temperature_display_domain():
    """Keep the whole precursor visible unless profile cropping is opted into."""
    raw = request.get("view_bounds")
    if raw is None:
        return [-half, half]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("Temperature view_bounds must contain xmin,xmax,ymin,ymax")
    bounds = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Temperature view bounds must be finite")
    if not (-half <= bounds[0] < bounds[1] <= half and
            -half <= bounds[2] < bounds[3] <= half):
        raise ValueError("Temperature view bounds must lie inside the simulation domain")
    return [-half, half] if request.get("profile_full_domain", True) else bounds[:2]

def _temperature_profile_window():
    if 2 not in GetGlobalAttributes().windows:
        checked(AddWindow(), "Create temperature profile window")
    checked(SetActiveWindow(1), "Select movie source window")
    # These production shock runs use code_to_cm=code_to_g=code_to_s=1.
    # Silo coordinates, er, and the exported T expression are already in CGS.
    checked(DefineScalarExpression("campaign_gas_temperature", "<T>"),
            "Define gas temperature from stored Silo expression")
    checked(DefineScalarExpression("campaign_radiation_temperature",
            "(er / %.17g)^0.25" % (4.0 * 5.67051e-5 / 2.99792458e10)),
            "Define radiation temperature")
    checked(DefineScalarExpression("campaign_profile_position_x",
            'recenter(coord(quadmesh)[0], "zonal")'), "Define native cell-center x")
    lineout = GetGlobalLineoutAttributes()
    lineout.Dynamic = 0
    lineout.createWindow = 0
    lineout.windowId = 2
    lineout.samplingOn = 0
    lineout.createReflineLabels = 0
    lineout.freezeInTime = 1
    checked(SetGlobalLineoutAttributes(lineout), "Set native mesh lineout")

def _temperature_lineout(frame):
    check_stop()
    checked(SetActiveWindow(2), "Select temperature profile window")
    DeleteAllPlots()
    checked(SetActiveWindow(1), "Select temperature source window")
    DeleteAllPlots()
    checked(AddPlot("Pseudocolor", "campaign_gas_temperature", 1, 0),
            "Add gas temperature source")
    checked(SetTimeSliderState(frame["state"]), "Set temperature source state")
    checked(DrawPlots(), "Draw temperature source")
    checked(Query("Time"), "Query temperature source time")
    actual_time = float(GetQueryOutputValue())
    if not math.isfinite(actual_time) or abs(actual_time - frame["time"]) > 1e-10 * max(1, abs(frame["time"])):
        raise RuntimeError("Temperature profile and slice snapshot times differ")
    # A small positive offset avoids intersections exactly on shared AMR faces.
    # Omitting num_samples uses native cell intersections (VisIt Lineout docs).
    checked(Lineout(start_point=(-half, slice_z, slice_z),
                    end_point=(half, slice_z, slice_z),
                    vars=("campaign_gas_temperature", "campaign_radiation_temperature",
                          "campaign_profile_position_x"),
                    use_sampling=0), "Extract native shock temperature profiles")
    checked(SetActiveWindow(2), "Select extracted temperature profiles")
    checked(DrawPlots(), "Draw native temperature profiles")
    if GetPlotList().GetNumPlots() != 3:
        raise RuntimeError("Expected gas, radiation, and physical-position profiles")

def scan_temperature_profiles(frames):
    """Scan native 1D data at every recorded state and return fixed plot bounds."""
    display_domain = _temperature_display_domain()
    _temperature_profile_window()
    low, high = float("inf"), -float("inf")
    frame_ranges = []
    for number, frame in enumerate(frames):
        _temperature_lineout(frame)
        limits = {}
        curves = {}
        checked(SetActivePlots(2), "Select native physical x positions")
        positions = list(GetPlotInformation()["Curve"])
        if len(positions) < 4 or len(positions) % 2 or not all(math.isfinite(value) for value in positions):
            raise RuntimeError("Invalid native cell-center positions")
        for index, name in enumerate(("gas", "radiation")):
            checked(SetActivePlots(index), "Select " + name + " temperature")
            checked(Query("MinMax", use_actual_data=1), "Query profile extrema")
            values = [float(value) for value in GetQueryOutputValue()]
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise RuntimeError("Invalid " + name + " temperature profile extrema")
            if values[0] < 0 or values[1] < values[0]:
                raise RuntimeError("Unphysical " + name + " temperature profile range")
            low, high = min(low, values[0]), max(high, values[1])
            limits[name] = values
            pairs = list(GetPlotInformation()["Curve"])
            if len(pairs) < 4 or len(pairs) % 2 or not all(math.isfinite(value) for value in pairs):
                raise RuntimeError("Invalid native " + name + " curve data")
            if len(pairs) != len(positions):
                raise RuntimeError("Temperature and position lineouts have different sample counts")
            for offset in range(0, len(pairs), 2):
                if abs(pairs[offset] - positions[offset]) > 1e-10 * max(1.0, 2 * half):
                    raise RuntimeError("Temperature and coordinate sample locations differ")
                pairs[offset] = positions[offset + 1]
            curves[name] = pairs
        # Retain the exact native Lineout pairs, then display as plain Curve
        # data in window 1. This avoids an offscreen window-2 Curve render crash
        # observed in the installed VisIt 3.4.2 viewer (the extraction succeeds).
        curve_file = "temperature_%06d.curve" % number
        with open(curve_file, "w") as stream:
            for name in ("gas", "radiation"):
                stream.write("# " + name + "_temperature_K\n")
                pairs = curves[name]
                for index in range(0, len(pairs), 2):
                    stream.write("%.17g %.17g\n" % (pairs[index], pairs[index + 1]))
        frame["temperature_curve"] = curve_file
        frame_ranges.append({"state": frame["state"], "time_s": frame["time"],
                             "ranges_K": limits, "curve_file": curve_file,
                             "native_points": {name: len(pairs) // 2 for name, pairs in curves.items()}})
    if not math.isfinite(low) or not math.isfinite(high) or high <= 0:
        raise RuntimeError("Temperature profiles contain no positive temperature")
    scale = "log" if low > 0 and high / low >= 1000 else "linear"
    if scale == "log":
        padding = max(math.log10(high / low) * 0.06, 0.05)
        bounds = [low / (10 ** padding), high * (10 ** padding)]
    else:
        padding = max((high - low) * 0.08, high * 0.02)
        bounds = [max(0.0, low - padding), high + padding]
    checked(SetActiveWindow(1), "Restore slice window after temperature scan")
    return {
        "method": "VisIt Lineout over native mesh intersections, uniform sampling disabled",
        "temporal_interpolation": "none; same recorded Silo state as both slices",
        "spatial_interpolation": "VisIt native Lineout representation of zone-centered data",
        "path_cm": [[-half, slice_z, slice_z], [half, slice_z, slice_z]],
        "horizontal_coordinate": "physical cell-center x from native Lineout of recenter(coord(quadmesh)[0], zonal)",
        "coordinate_transform": "Replace native Lineout distance with separately extracted physical x at identical native samples; no temperature interpolation",
        "domain_cm": [-half, half],
        "display_domain_cm": display_domain,
        "display_crop": display_domain != [-half, half],
        "profile_full_domain": display_domain == [-half, half],
        "display_crop_reason": (request.get("movie_view") or {}).get("reason", "") if display_domain != [-half, half] else "",
        "data_scope": "Full native lineout and full-domain temperature limits retained; optional crop affects display only",
        "temperature_units": "K", "scale": scale, "limits_K": bounds,
        "data_range_K": [low, high], "frame_ranges": frame_ranges,
        "gas_expression": "Silo T (EOS, composition, and dual-energy selection from solver)",
        "radiation_expression": "(er/a_r)^0.25",
        "a_r_cgs": 4.0 * 5.67051e-5 / 2.99792458e10,
        "constant_source": "src/physcon.cpp CGS sigma=5.67051e-5, c=2.99792458e10; a_r=4*sigma/c",
    }

def render_temperature_profiles(frames, profile_metadata):
    """Write one wide profile panel per map frame without changing its time."""
    checked(SetActiveWindow(2), "Select temporary lineout window")
    DeleteAllPlots()
    checked(DeleteWindow(), "Close temporary lineout window")
    checked(SetActiveWindow(1), "Use primary window for temperature profiles")
    for number, frame in enumerate(frames):
        check_stop()
        DeleteAllPlots()
        curve_file = os.path.abspath(frame["temperature_curve"])
        checked(OpenDatabase(curve_file), "Open exact native lineout data")
        for name in ("gas", "radiation"):
            checked(AddPlot("Curve", name + "_temperature_K", 1, 0), "Add " + name + " curve")
        for index, name, color in ((0, "Gas", (205, 78, 0, 255)),
                                   (1, "Radiation", (0, 104, 180, 255))):
            checked(SetActivePlots(index), "Style " + name + " temperature")
            curve = CurveAttributes()
            curve.curveColorSource = curve.Custom
            curve.curveColor = color
            curve.lineWidth = 2
            curve.showPoints = 0
            curve.showLabels = 0
            curve.showLegend = 0
            checked(SetPlotOptions(curve), "Set " + name + " curve style")
        annotation = AnnotationAttributes()
        annotation.userInfoFlag = 0
        annotation.databaseInfoFlag = 0
        annotation.timeInfoFlag = 0
        annotation.backgroundColor = (255, 255, 255, 255)
        annotation.foregroundColor = (0, 0, 0, 255)
        annotation.axes2D.visible = 1
        for axis, title, units in ((annotation.axes2D.xAxis, "x", "cm"),
                                   (annotation.axes2D.yAxis, "Temperature", "K")):
            axis.title.userTitle, axis.title.userUnits = 1, 1
            axis.title.title, axis.title.units = title, units
            axis.title.font.bold, axis.title.font.italic = 0, 0
            axis.title.font.font = axis.title.font.Arial
            axis.label.font.font = axis.label.font.Arial
            axis.label.font.bold, axis.label.font.italic = 0, 0
            axis.title.font.scale = 2.2
            axis.label.font.scale = 2.0
        checked(SetAnnotationAttributes(annotation), "Set temperature axes")
        view = ViewCurveAttributes()
        view.domainCoords = tuple(profile_metadata.get("display_domain_cm", profile_metadata["domain_cm"]))
        view.rangeCoords = tuple(profile_metadata["limits_K"])
        view.viewportCoords = (0.10, 0.97, 0.30, 0.77)
        view.domainScale = view.LINEAR
        view.rangeScale = view.LOG if profile_metadata["scale"] == "log" else view.LINEAR
        checked(SetViewCurve(view), "Set fixed temperature profile limits")
        crop_label = "Cropped x view; " if profile_metadata.get("display_crop") else "Full-domain x profile; "
        labels = [text_label("Central shock temperature profile | t = %.6g s" % frame["time"],
                             0.055, 0.89, 0.040),
                  text_label("Gas temperature", 0.62, 0.89, 0.038),
                  text_label("Radiation temperature", 0.78, 0.89, 0.038),
                  text_label(crop_label + "native cells on y=z=0+; fixed %s scale; synchronized recorded state" % profile_metadata["scale"],
                             0.055, 0.052, 0.042)]
        labels[1].useForegroundForTextColor = 0
        labels[1].textColor = (205, 78, 0, 255)
        labels[2].useForegroundForTextColor = 0
        labels[2].textColor = (0, 104, 180, 255)
        checked(DrawPlots(), "Draw styled temperature profile")
        filename = "temperature_%06d.png" % number
        save = SaveWindowAttributes()
        save.outputToCurrentDirectory = 0
        save.outputDirectory = os.getcwd()
        save.fileName = filename
        save.family = 0
        save.format = save.PNG
        save.width = request["width"] * 2
        save.height = request.get("profile_height", 400)
        save.resConstraint = save.NoConstraint
        save.screenCapture = 0
        checked(SetSaveWindowAttributes(save), "Set temperature image dimensions")
        checked(SaveWindow(), "Save " + filename)
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            raise RuntimeError("Missing temperature profile image " + filename)
        frame["temperature_image"] = filename
        for label in labels:
            label.Delete()
'''
