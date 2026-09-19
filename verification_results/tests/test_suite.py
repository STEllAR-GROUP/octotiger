import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from verification_results import runner
from verification_results.adapters import native_suite as suite

class SuiteContract(unittest.TestCase):
    def test_reference_period_and_front_cell_averages(self):
        d=runner.descriptors()['radiation.diagnostics.streaming_wave_1d'][1];p=d['parameters']
        n=32;L=p['length_cm'];x=-L/2+(np.arange(n)+.5)*L/n
        np.testing.assert_allclose(suite.reference(d['name'],p,0,x,n),suite.reference(d['name'],p,L/suite.C,x,n),rtol=2e-15)
        front=suite.reference('streaming_front_1d',p,.25*L/suite.C,x,n)
        self.assertAlmostEqual(float(front.mean()),.5+.5e-10)
        self.assertTrue(np.all((front>=1e-10-1e-14)&(front<=1+1e-14)))

    def test_diffusion_reference_retains_finite_light_speed(self):
        d=runner.descriptors()['radiation.diagnostics.static_diffusion'][1];p=d['parameters']
        x=np.array([0.]);n=32;initial=suite.reference(d['name'],p,0,x,n)[0]
        self.assertAlmostEqual(initial,1+p['amplitude']*np.sinc(1/n))
        self.assertLess(suite.reference(d['name'],p,p['final_time_s'],x,n)[0],initial)

    def test_conditional_is_not_success_or_invented_reference(self):
        d=runner.descriptors()['radiation.ensman.equilibrium_sphere'][1]
        self.assertIsNone(d['parameters']['initial_condition'])
        with tempfile.TemporaryDirectory() as tmp:
            code=suite.execute([('radiation.ensman.equilibrium_sphere',d)],['--output',tmp])
            self.assertEqual(code,3)
            self.assertEqual(json.loads((Path(tmp)/'summary.json').read_text())[0]['status'],'conditional')

    def test_build_failure_is_retained_and_nonzero(self):
        d=runner.descriptors()['radiation.diagnostics.streaming_wave_1d'][1]
        with tempfile.TemporaryDirectory() as tmp, patch.object(suite,'compile_fixture',side_effect=RuntimeError('injected compiler failure')):
            code=suite.execute([('test',d)],['0','--output',tmp])
            self.assertEqual(code,1)
            meta=json.loads((Path(tmp)/'test/l0/run.json').read_text())
            self.assertEqual(meta['status'],'failed')
            self.assertIn('injected compiler failure',(Path(tmp)/'test/l0/run.log').read_text())

    def test_movie_failure_propagates(self):
        d=runner.descriptors()['radiation.diagnostics.streaming_wave_1d'][1];p=d['parameters'];n=8
        x=-p['length_cm']/2+(np.arange(n)+.5)*p['length_cm']/n
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rows=[]
            for frame,t in enumerate([0,p['final_time_s']]):
                E=suite.reference(d['name'],p,t,x,n)
                rows.extend(zip(np.full(n,frame),np.full(n,t),x,E,suite.C*E,np.ones(n),np.zeros(n)))
            np.savetxt(folder/'samples.csv',rows,delimiter=',',header='frame,t,x,E,Fx,gas_energy,momentum',comments='')
            np.savetxt(folder/'comparison.csv',np.column_stack([x,E,E,np.zeros(n)]),delimiter=',',header='x,E,reference,signed_error',comments='')
            with self.assertRaises(subprocess.CalledProcessError):suite.visualize(d,folder,n,'/bin/false')
            self.assertTrue((folder/'movie.log').exists())

    def test_boundary_trace_detects_wrong_epochs_times_and_gas_writes(self):
        dtype=[('gas_step',int),('event','U12'),('rcycle',int),('hcycle',int),('time',float),('interior_E',float),('interior_Fx',float),('halo_valid',int),('hydro_unchanged',int)]
        events=np.array([(0,'hydro',47,14,0,0,0,1,1),
            (0,'radiation',47,14,0,1,1,1,1),(0,'flux',48,14,0,0,0,1,1),
            (0,'radiation',48,14,.5,1,.8,1,1),(0,'flux',49,14,.5,0,0,1,1),
            (0,'radiation',49,14,1,1,.6,1,1)],dtype=dtype)
        history=np.array([(0.,0),(1.,2)],dtype=[('t',float),('subcycles',int)])
        self.assertTrue(all(suite.check_exchange_trace(events,history).values()))
        for field,value in [('rcycle',99),('hcycle',99),('time',.7),('halo_valid',0),('hydro_unchanged',0)]:
            bad=events.copy();bad[3][field]=value
            self.assertFalse(all(suite.check_exchange_trace(bad,history).values()),field)

    def test_unsupported_thread_count_and_output_reuse_fail(self):
        with self.assertRaises(ValueError):suite.execute([],['--threads','2'])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'raw.log').write_text('preserve')
            with self.assertRaises(ValueError):suite.execute([],['--output',tmp])
            self.assertEqual((Path(tmp)/'raw.log').read_text(),'preserve')

if __name__=='__main__':unittest.main()
