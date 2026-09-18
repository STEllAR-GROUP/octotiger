"""Check output timing, state selection, and actual MP4 playback metadata.

The VisIt fixture checks script control flow; it does not test a real Silo reader.
"""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from movie_support import cadence, encode, frame_schedule, numerical_silos
import visit_movie


class MovieTests(unittest.TestCase):
    def test_batched_output_has_enough_states(self):
        # Mimic the solver's output check, five timesteps, then another check.
        capture=cadence(.2,61,.4)
        self.assertEqual(capture['steps_per_output_check'],5)
        for natural_dt in (.01,.001,.00017):
            t=0.; count=0; saved=[]
            while t < .2-1e-14:
                if t/capture['odt'] >= count:
                    saved.append(t);count+=1
                for _ in range(5):
                    t+=min(natural_dt,capture['hard_dt'],max(0.,.2-t))
            saved.append(.2)
            self.assertGreaterEqual(len(saved),60)
            self.assertLessEqual(len(saved),61)
            self.assertEqual(saved[0],0.)
            self.assertLessEqual(max(b-a for a,b in zip(saved,saved[1:])),
                                 2*capture['odt']+1e-12)
        manual=cadence(.2,61,.4,odt=.002)
        self.assertAlmostEqual(manual['hard_dt'],.0004)
        self.assertEqual(manual['requested_snapshots'],101)

    def test_playback_preserves_time_gaps_and_duration(self):
        # Uneven physical intervals must not become equally long in the video.
        sequence=frame_schedule([0.,.1,1.],seconds=12,fps=10,hold=1)
        self.assertEqual(len(sequence),120)
        self.assertEqual(sequence[:10],[0]*10)
        self.assertEqual(sequence[-10:],[2]*10)
        self.assertGreater(sequence.count(1),4*sequence.count(0))
        self.assertEqual(set(sequence),{0,1,2})
        for times in ([0.,0.],[0.,float('nan')],[1.,0.],[]):
            with self.assertRaises(ValueError):frame_schedule(times)
        with self.assertRaises(ValueError):frame_schedule([0.,1.],seconds=2,hold=1)

    def test_silo_order_and_missing_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ('X.10.silo','X.2.silo','final.silo','analytic.silo'):
                (folder/name).write_bytes(b'fixture')
                (folder/(name+'.data')).mkdir()
            self.assertEqual([p.name for p in numerical_silos(folder)],
                             ['X.2.silo','X.10.silo','final.silo'])
            (folder/'X.2.silo.data').rmdir()
            with self.assertRaises(ValueError):numerical_silos(folder)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                         'FFmpeg and ffprobe are required for encoding test')
    def test_encoded_mp4_duration_frames_and_pixel_format(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='movie test ') as temp:
            folder=Path(temp);frames=[]
            for i,color in enumerate(('red','green','blue')):
                path=folder/('input %d.png'%i)
                Image.new('RGB',(161,101),color).save(path)
                frames.append(dict(time=(0.,.1,1.)[i],image=str(path)))
            out=folder/'test.mp4'
            result=encode(dict(frames=frames),out,seconds=20,fps=30,hold=1)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error',
                '-show_streams','-show_format','-of','json',str(out)],text=True))
            stream=probe['streams'][0]
            self.assertEqual((stream['width'],stream['height']),(162,102))
            self.assertEqual(stream['pix_fmt'],'yuv420p')
            self.assertEqual(stream['r_frame_rate'],'30/1')
            self.assertEqual(int(stream['nb_frames']),600)
            self.assertAlmostEqual(float(probe['format']['duration']),20.,places=2)
            self.assertEqual(result['displayed_snapshots'],3)

    def test_visit_state_scan_and_fixed_color_scale(self):
        # Stub only VisIt's embedded API. Exercise our real renderer function.
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); state=[0]; query=['']; options=[]; saved=[]; plots=[]
            files=[str(folder/name) for name in
                   ('X.0.silo','X.1.silo','X.2.silo','final.silo')]
            cfg=dict(render_directory=temp,silos=files,field='er',color_table='hot',
                     view='slice',axis='z',slice_position=0.,title='Fixture',
                     minimum=None,maximum=None,minimum_snapshots=3,width=160,height=128)
            settings=folder/'settings.json';settings.write_text(json.dumps(cfg))
            def attributes():
                return SimpleNamespace(Linear=0,Intercept=0,XAxis=0,YAxis=1,ZAxis=2,
                                       Solid=0,PNG=0,NoConstraint=0)
            def select(index):state[0]=index;return 1
            def value():
                return ([0.,.1,.2,.2][state[0]] if query[0]=='Time'
                        else [(-1.,2.),(-3.,4.),(-2.,3.),(-2.,3.)][state[0]])
            def save():
                a=options[0];path=folder/(a.fileName+'.png')
                Image.new('RGB',(a.width,a.height),'white').save(path)
                saved.append(state[0]);return str(path)
            api={name:(lambda *args:1) for name in
                 ('OpenDatabase','AddPlot','AddOperator','SetOperatorOptions','DrawPlots',
                  'ResetView','SetAnnotationAttributes','SuppressQueryOutputOn','SaveSession')}
            api.update(PseudocolorAttributes=attributes,SliceAttributes=attributes,
                SaveWindowAttributes=attributes,GetAnnotationAttributes=attributes,
                CreateAnnotationObject=lambda *args:SimpleNamespace(),
                SetPlotOptions=lambda a:plots.append(vars(a).copy()),
                TimeSliderGetNStates=lambda:len(files),SetTimeSliderState=select,
                Query=lambda name:query.__setitem__(0,name),GetQueryOutputValue=value,
                SetSaveWindowAttributes=lambda a:options.__setitem__(slice(None),[a]),
                SaveWindow=save)
            with patch.dict(visit_movie.__dict__,api),contextlib.redirect_stdout(io.StringIO()):
                visit_movie.main(str(settings))
            result=json.loads((folder/'frames.json').read_text())
            self.assertEqual([f['time'] for f in result['frames']],[0.,.1,.2])
            self.assertEqual(saved,[0,1,3])  # Duplicate end time uses final.silo.
            self.assertEqual(result['color_limits'],[-3.,4.])
            self.assertEqual((plots[-1]['min'],plots[-1]['max']),(-3.,4.))
            self.assertNotIn('analytic.silo',(folder/'numerical.visit').read_text())


if __name__=='__main__':unittest.main()
