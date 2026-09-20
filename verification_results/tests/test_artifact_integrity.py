"""Integrity failures must invalidate retained products and process success."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from verification_results import runner
from verification_results import audit_artifacts
from verification_results.adapters import native_suite as suite


class ArtifactIntegrity(unittest.TestCase):
    def descriptor(self):
        descriptor=copy.deepcopy(runner.descriptors()['radiation.diagnostics.thermal_relaxation'][1])
        descriptor['parameters']['frames']=3
        return descriptor

    def raw(self,folder,descriptor=None,n=8):
        descriptor=descriptor or self.descriptor()
        parameters=descriptor['parameters'];end=parameters['final_time_s']
        times=np.linspace(0,end,5);x=-parameters['length_cm']/2+(np.arange(n)+.5)*parameters['length_cm']/n
        samples=[];history=[]
        for index,t in enumerate(times):
            energy=suite.reference(descriptor['name'],parameters,t,x,n)
            gas=parameters['gas_internal_erg_cm3']+parameters['radiation_energy_erg_cm3']-energy
            history.append((t,energy.mean(),gas.mean(),0,0,energy.min(),0,0 if index==0 else end/4,0,0,47,13,1))
            if index%2==0:
                samples.extend(zip(np.full(n,index//2),np.full(n,t),x,energy,np.zeros(n),gas,np.zeros(n)))
        for name,values in [('samples.csv',samples),('history.csv',history)]:
            stream=io.StringIO()
            np.savetxt(stream,values,delimiter=',',header=','.join(suite.RAW_SCHEMAS[name]),comments='')
            suite.atomic_write(folder/name,stream.getvalue())
        suite.atomic_write(folder/'exchanges.csv',','.join(suite.RAW_SCHEMAS['exchanges.csv'])+'\n')
        suite.atomic_write(folder/'run.log',f'completed {descriptor["name"]} N={n} t={end} steps=4 c=2.99792e+10\n')

    def test_valid_complete_serialized_data(self):
        descriptor=self.descriptor()
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);self.raw(folder,descriptor)
            data=suite.validate_raw(descriptor['name'],descriptor['parameters'],folder,8)
            self.assertEqual(len(data['samples.csv']),24)
            self.assertEqual(len(data['history.csv']),5)
            self.assertEqual(len(data['exchanges.csv']),0)
            self.assertEqual(suite.evaluate(descriptor['name'],descriptor,folder,8)['status'],'passed')

    def test_empty_partial_and_complete_row_truncations_rejected(self):
        descriptor=self.descriptor()
        mutations={
            'empty samples':('samples.csv',lambda raw:b''),
            'partial history':('history.csv',lambda raw:raw[:-13]),
            'missing final history row':('history.csv',lambda raw:b'\n'.join(raw.split(b'\n')[:-2])+b'\n'),
            'missing entire sample frame':('samples.csv',lambda raw:b'\n'.join(raw.split(b'\n')[:-9])+b'\n'),
            'missing cell row':('samples.csv',lambda raw:b'\n'.join(raw.split(b'\n')[:-2])+b'\n'),
            'empty exchange header':('exchanges.csv',lambda raw:b''),
            'empty completion log':('run.log',lambda raw:b''),
        }
        for label,(filename,mutate) in mutations.items():
            with self.subTest(label),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);self.raw(folder,descriptor)
                path=folder/filename;path.write_bytes(mutate(path.read_bytes()))
                with self.assertRaises(ValueError):suite.validate_raw(descriptor['name'],descriptor['parameters'],folder,8)

    def test_nonfinite_nonenergy_fields_and_wrong_schema_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);descriptor=self.descriptor();self.raw(folder,descriptor)
            path=folder/'samples.csv';original=path.read_text();rows=original.splitlines()
            fields=rows[1].split(',');fields[-1]='nan';rows[1]=','.join(fields)
            path.write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError,'nonfinite momentum'):
                suite.validate_raw(descriptor['name'],descriptor['parameters'],folder,8)
            path.write_text(original.replace('gas_energy','unexpected',1))
            with self.assertRaisesRegex(ValueError,'schema'):
                suite.validate_raw(descriptor['name'],descriptor['parameters'],folder,8)

    def test_missing_interior_history_row_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);descriptor=self.descriptor();self.raw(folder,descriptor)
            path=folder/'history.csv';rows=path.read_text().splitlines();del rows[2]
            path.write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError,'history|timesteps'):
                suite.validate_raw(descriptor['name'],descriptor['parameters'],folder,8)

    def test_atomic_publication_readback_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'value.json'
            with patch.object(Path,'read_bytes',return_value=b'corrupted'):
                with self.assertRaisesRegex(RuntimeError,'readback mismatch'):
                    suite.atomic_write(path,b'expected\n')
            self.assertEqual(path.read_bytes(),b'expected\n')

    def execute_fake(self,folder,after_web=None,after_visualize=None):
        descriptor=self.descriptor();original_web=suite.web
        identity={'commit':'test','dirty':True,'sha256':'test','files':{}}
        def run(command,**kwargs):
            self.assertEqual(command[0],'fake-fixture')
            self.raw(Path(command[2]),descriptor)
            return subprocess.CompletedProcess(command,0)
        def visualize(_descriptor,run_folder,n,ffmpeg):
            for name in ['comparison.png','movie.mp4','movie.log','movie-validation.log','frames/0000.png']:
                suite.atomic_write(run_folder/name,b'generated and checked by mock visualizer\n')
            suite.dump(run_folder/'movie-validation.json',{'decoded_frames':3,'artifact':suite.file_record(run_folder/'movie.mp4')})
            if after_visualize:after_visualize(run_folder)
            return ['comparison.png','movie.mp4']
        mutated=False
        def web(output,results,embedded=False):
            nonlocal mutated
            original_web(output,results,embedded)
            if embedded and after_web and not mutated:
                mutated=True;after_web(output/'test/l0')
        with patch.object(suite,'source_identity',return_value=identity), \
             patch.object(suite,'compile_fixture',return_value=('fake-fixture',{})), \
             patch.object(suite.subprocess,'run',side_effect=run), \
             patch.object(suite,'visualize',side_effect=visualize), \
             patch.object(suite,'web',side_effect=web):
            return suite.execute([('test',descriptor)],['0','--output',str(folder)])

    def test_sealed_outputs_can_be_revalidated_without_reexecution(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            self.assertEqual(self.execute_fake(folder),0)
            self.assertTrue(suite.verify_artifacts(folder/'test/l0'))
            manifest=json.loads((folder/'test/l0/artifacts.json').read_text())
            self.assertIn('history.csv',manifest['files'])
            self.assertIn('movie.mp4',manifest['files'])
            self.assertIn('frames/0000.png',manifest['files'])
            self.assertIn('movie-validation.log',manifest['files'])

    def test_post_generation_corruption_produces_failed_summary_and_exit(self):
        for filename in ['history.csv','samples.csv','movie.mp4','run.log','artifacts.json','run.json']:
            with self.subTest(filename),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary)
                def corrupt(run_folder):
                    path=run_folder/filename;path.write_bytes(path.read_bytes()[:8])
                self.assertEqual(self.execute_fake(folder,after_web=corrupt),1)
                summary=json.loads((folder/'summary.json').read_text())[0]
                self.assertEqual(summary['status'],'failed')
                self.assertEqual(summary['runs'][0]['status'],'failed')
                self.assertEqual(summary['runs'][0]['artifact_integrity']['status'],'failed')
                self.assertEqual(json.loads((folder/'test/l0/run.json').read_text())['status'],'failed')
                self.assertIn('Final artifact audit failed',(folder/'report.html').read_text())

    def test_mutation_before_sealing_cannot_replace_evaluated_snapshot(self):
        for filename in ['history.csv','movie.mp4']:
            with self.subTest(filename),tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary)
                def corrupt(run_folder):
                    path=run_folder/filename;data=path.read_bytes()
                    path.write_bytes(data[:-4])
                self.assertEqual(self.execute_fake(folder,after_visualize=corrupt),1)
                self.assertIn('integrity mismatch',json.loads((folder/'summary.json').read_text())[0]['runs'][0]['error'])

    @unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required for real decode regression')
    def test_real_movie_decode_count_and_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);path=folder/'movie.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=16x16:r=4',
                            '-frames:v','3','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
            self.assertEqual(suite.validate_movie(path,3,'ffmpeg')['decoded_frames'],3)
            with self.assertRaisesRegex(RuntimeError,'frame count mismatch'):suite.validate_movie(path,4,'ffmpeg')
            good=path.read_bytes()
            path.write_bytes(good[:48])
            with self.assertRaisesRegex(RuntimeError,'decode failed'):suite.validate_movie(path,3,'ffmpeg')
            path.write_bytes(b'not a movie but nonempty')
            with self.assertRaisesRegex(RuntimeError,'decode failed'):suite.validate_movie(path,3,'ffmpeg')

    def test_source_mismatch_is_strict_but_does_not_skip_product_audit(self):
        descriptor={'adapter':{'name':'native_suite'},'name':'test','parameters':{'frames':3}}
        files={'wrapper.py':hashlib.sha256(b'original').hexdigest()}
        source={'files':files,'sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);current=root/'current';current.mkdir();(current/'wrapper.py').write_text('changed')
            for build in ['Debug','Release','RelWithDebInfo']:
                folder=root/build/'radiation';folder.mkdir(parents=True)
                suite.dump(folder/'source.json',source);runs=[];links=[]
                for level in [0,1,2]:
                    path=folder/'test'/f'l{level}';path.mkdir(parents=True)
                    run={'level':level,'cells':8*2**level,'status':'passed','L1':0.0};runs.append(run)
                    generated=path/'suite.cpp';executable=path/'suite'
                    generated.write_text('generated source');executable.write_text('compiled executable')
                    build_meta={'build_type':build,'command':['cxx',str(generated),'-o',str(executable)],
                                'generated_source_sha256':suite.file_record(generated)['sha256'],
                                'executable_sha256':suite.file_record(executable)['sha256']}
                    meta={**run,'source':source,'descriptor':descriptor,'build':build_meta,'artifact_integrity':{'file_count':14}}
                    for filename in audit_artifacts.EMBEDDED_FILES:
                        payload=json.dumps(meta).encode() if filename=='run.json' else b'{"decoded_frames":3}' if filename=='movie-validation.json' else b'artifact'
                        (path/filename).write_bytes(payload)
                        encoded=base64.b64encode(payload).decode()
                        links.append(f'<a download="test-l{level}-{filename}" href="data:application/octet-stream;base64,{encoded}">download</a>')
                suite.dump(folder/'summary.json',[{'id':'test','status':'passed','runs':runs}])
                suite.dump(root/build/'summary.json',{'families':[{'family':'radiation','status':'passed','returncode':0}]})
                (folder/'report.html').write_text(''.join(links))
            with patch.object(runner,'descriptors',return_value={'test':(None,descriptor)}), \
                 patch.object(suite,'ROOT',current), \
                 patch.object(suite,'verify_artifacts',return_value=True) as manifests, \
                 patch.object(suite,'validate_raw',return_value={}) as raw, \
                 patch.object(suite,'validate_movie',return_value={}) as movies:
                result=audit_artifacts.audit(root,decode=True)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['source_status'],'failed')
            self.assertEqual(result['artifact_status'],'passed')
            self.assertEqual(result['native_runs'],9)
            self.assertEqual(result['decoded_movies'],9)
            self.assertEqual(manifests.call_count,9);self.assertEqual(raw.call_count,9);self.assertEqual(movies.call_count,9)
            self.assertEqual(len(result['errors']),3)
            self.assertTrue(all(error['kind']=='source_mismatch' for error in result['errors']))


if __name__=='__main__':unittest.main()
