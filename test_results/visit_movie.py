"""Run inside VisIt: visit -nowin -cli -s visit_movie.py settings.json.

Only the standard library and VisIt's embedded Python API are required here.
The outer movies.py process uses the user's normal Python environment.
"""
import json
import math
import os
import sys
import traceback

SUPPORTS_CGS = True

def checked(result, action):
    if result == 0:
        raise RuntimeError(action + ': ' + str(GetLastError()))
    return result


def main(settings):
    with open(settings) as f: cfg = json.load(f)
    target = cfg['render_directory']
    files = cfg['silos']
    os.makedirs(target, exist_ok=True)
    database = os.path.join(target, 'numerical.visit')
    with open(database, 'w') as f:
        for file in files:
            if '\n' in file or '\r' in file: raise ValueError('Newline in Silo path')
            f.write(file + '\n')
    checked(OpenDatabase(database), 'OpenDatabase')
    field = cfg['field']
    cgs = cfg.get('units', {}).get('system') == 'CGS'
    field_units = ('erg/cm^3' if field == 'er' else 'erg/(cm^2 s)') if cgs else 'Silo units'
    if field == 'fluxmag':
        DefineScalarExpression('fluxmag', 'sqrt(fx*fx+fy*fy+fz*fz)')
    checked(AddPlot('Pseudocolor', field), 'AddPlot')
    plot = PseudocolorAttributes()
    plot.colorTableName = cfg['color_table']
    plot.scaling = plot.Linear
    plot.legendFlag = 1
    SetPlotOptions(plot)
    if cfg['view'] == 'slice':
        checked(AddOperator('Slice'), 'AddOperator')
        plane = SliceAttributes()
        plane.originType = plane.Intercept
        plane.axisType = {'x': plane.XAxis, 'y': plane.YAxis, 'z': plane.ZAxis}[cfg['axis']]
        plane.originIntercept = cfg['slice_position']
        plane.project2d = 1
        SetOperatorOptions(plane)
    checked(DrawPlots(), 'DrawPlots')
    checked(ResetView(), 'ResetView')
    # Set the camera once; changing time must not change the framing.
    if cfg['view'] == '3d':
        view = GetView3D()
        view.viewNormal = (0.55, -0.7, 0.45)
        view.viewUp = (0, 0, 1)
        view.perspective = 0
        SetView3D(view)
    annotation = GetAnnotationAttributes()
    annotation.backgroundColor = (255,255,255,255)
    annotation.foregroundColor = (0,0,0,255)
    annotation.backgroundMode = annotation.Solid
    annotation.userInfoFlag = 0
    annotation.databaseInfoFlag = 0
    annotation.timeInfoFlag = 0
    if cgs:
        # The Silo values are already CGS. Override captions, not the data.
        for axes in (annotation.axes2D, annotation.axes3D):
            for axis_name in ('xAxis', 'yAxis'):
                axis = getattr(axes, axis_name)
                axis.title.userUnits = 1
                axis.title.units = 'cm'
                axis.title.visible = 1
        annotation.axes3D.zAxis.title.userUnits = 1
        annotation.axes3D.zAxis.title.units = 'cm'
        annotation.axes3D.zAxis.title.visible = 1
    SetAnnotationAttributes(annotation)
    title = CreateAnnotationObject('Text2D')
    title.position = (.04,.96)
    title.height = .025
    title.fontBold = 1
    title.text = cfg['title'] + ' | ' + field + ' (' + field_units + ')'
    clock = CreateAnnotationObject('Text2D')
    clock.position = (.04,.91)
    clock.height = .023
    note = CreateAnnotationObject('Text2D')
    note.position = (.04,.035)
    note.height = .020
    note.text = ('%s=%g %s slice' % (cfg['axis'],cfg['slice_position'],'cm' if cgs else '(Silo length units)')
                 if cfg['view']=='slice' else 'Numerical solution | fixed 3D view')
    SuppressQueryOutputOn()
    if TimeSliderGetNStates() != len(files):
        raise RuntimeError('VisIt time-state count differs from numerical Silo list')
    states = []; lower = float('inf'); upper = -float('inf')
    # One complete pass establishes time metadata and a fixed scale for the movie.
    for index, file in enumerate(files):
        checked(SetTimeSliderState(index), 'SetTimeSliderState')
        checked(DrawPlots(), 'DrawPlots')
        Query('Time'); value = GetQueryOutputValue()
        if value is None: raise RuntimeError('Missing Silo time: ' + file)
        time = float(value)
        if not math.isfinite(time) or time < 0: raise RuntimeError('Invalid Silo time: ' + file)
        Query('MinMax'); bounds = GetQueryOutputValue()
        if bounds is None or len(bounds) != 2 or not all(math.isfinite(float(v)) for v in bounds):
            raise RuntimeError('Invalid field range: ' + file)
        lower = min(lower, float(bounds[0])); upper = max(upper, float(bounds[1]))
        row = dict(state=index, time=time, silo=file)
        if states:
            tolerance = 1e-11 * max(abs(time), abs(states[-1]['time']), 1e-300)
            if abs(time-states[-1]['time']) <= tolerance and os.path.basename(file)=='final.silo':
                states[-1] = row  # final may duplicate the last numerical output time.
                continue
            if time <= states[-1]['time']:
                raise RuntimeError('Silo times are not strictly increasing: ' + file)
        states.append(row)
        print('Scanned %d/%d Silo states' % (index+1,len(files)))
        sys.stdout.flush()
    if len(states)<2 or states[-1]['time'] <= states[0]['time']:
        raise RuntimeError('Need at least two distinct physical times')
    if abs(states[0]['time']) > 1e-10*states[-1]['time']:
        raise RuntimeError('Missing initial t=0 snapshot')
    expected = cfg.get('expected_final_time')
    if expected is not None and not math.isclose(states[-1]['time'],expected,rel_tol=1e-9,abs_tol=1e-12):
        raise RuntimeError('Final Silo time %g s differs from requested %g s; check solver units' %
                           (states[-1]['time'],expected))
    required = cfg.get('minimum_snapshots',2)
    if len(states) < required:
        raise RuntimeError('Only %d snapshots; expected at least %d. Check output cadence, or use --allow-sparse.' % (len(states), required))
    if cfg['minimum'] is not None: lower=cfg['minimum']
    if cfg['maximum'] is not None: upper=cfg['maximum']
    if upper < lower: raise ValueError('Invalid color limits')
    if upper==lower:
        pad = max(abs(lower)*1e-8,1e-30); lower-=pad; upper+=pad
    plot.minFlag=1;plot.maxFlag=1;plot.min=lower;plot.max=upper
    SetPlotOptions(plot)
    save = SaveWindowAttributes()
    save.outputToCurrentDirectory=0;save.outputDirectory=target
    save.family=0;save.format=save.PNG;save.screenCapture=0
    save.width=cfg['width'];save.height=cfg['height'];save.resConstraint=save.NoConstraint
    for frame, row in enumerate(states):
        checked(SetTimeSliderState(row['state']), 'SetTimeSliderState')
        clock.text = 't = %.6e s | snapshot %d / %d' % (row['time'],frame+1,len(states))
        checked(DrawPlots(), 'DrawPlots')
        save.fileName='frame_%06d' % frame
        SetSaveWindowAttributes(save)
        filename=SaveWindow()
        expected=os.path.join(target,save.fileName+'.png')
        if not filename or not os.path.isfile(expected) or os.path.getsize(expected)==0:
            raise RuntimeError('VisIt did not save ' + expected)
        row['image']=expected
        print('Rendered %d/%d' % (frame+1,len(states)));sys.stdout.flush()
    checked(SaveSession(os.path.join(target,'movie.session')), 'SaveSession')
    result=dict(status='complete',frames=states,color_limits=[lower,upper],
                time_units='s' if cgs else 'seconds, as stored in Silo',field_units=field_units,
                configuration=cfg)
    with open(os.path.join(target,'frames.json'),'w') as f:
        json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    try:
        arguments=Argv()
        if len(arguments)!=1: raise ValueError('Expected the movie settings JSON filename')
        main(arguments[0])
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        # VisIt may otherwise return success after an embedded-script exception.
        os._exit(1)
    # Use VisIt's shutdown routine so the viewer, metadata server, and engine
    # are closed before the CLI returns.  sys.exit() can leave that teardown
    # racing the frontend launcher and occasionally produces exit code 250
    # even though every frame was written successfully.
    Close()
