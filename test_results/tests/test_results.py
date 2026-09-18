import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from results import FIELDS, NORMS, order, parse_log, read_norms, read_slice

class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.meta=dict(case='streaming_front',time=.2,cells=4,length=2.,dx=.5,level=0)
        self.log='RADIATION_TEST_FINISHED RADIATION_STREAMING_FRONT t=0.2\n'
        self.log+='\n'.join(f'{f} 1e-2 2e-2 3e-2' for f in FIELDS)+'\n'
        (self.folder/'run.log').write_text(self.log)
        for norm,e in zip(NORMS,(.01,.02,.03)):
            (self.folder/(norm+'.dat')).write_text('0.5 0 0 0 '+('%.6e '%e)*4+'\n')
        self.slices=self.folder/'radiation-slices';self.slices.mkdir()
        self.header='t,dx,x,y,z,er,fx,fy,fz,er_ref,fx_ref,fy_ref,fz_ref\n'
        self.rows=[f'.2,.5,{-1+(i+.5)*.5},{-1+(j+.5)*.5},.25,1,0,0,0,1,0,0,0\n'
                   for i in range(4) for j in range(4)]
        (self.slices/'slice-0-0.csv').write_text(self.header+''.join(reversed(self.rows)))

    def test_valid_and_order(self):
        self.assertEqual(read_norms(self.folder,self.meta)['er']['L1'],.01)
        self.assertEqual(read_slice(self.folder,self.meta).shape,(4,4))
        self.assertAlmostEqual(order(.04,.01,.2,.1),2)
        self.assertIsNone(order(0,0,.2,.1))
        self.assertLess(order(.01,.02,.2,.1),0)

    def test_incomplete_and_duplicate_log(self):
        for bad in (self.log.replace('t=0.2','t=0.1'),self.log+'er 1 2 3\n',
                    self.log.replace('fx 1e-2','fx nan'),self.log.replace('fz 1e-2 2e-2 3e-2','')):
            with self.assertRaises(ValueError): parse_log(bad,'streaming_front',.2)

    def test_stale_norms(self):
        with (self.folder/'L1.dat').open('a') as f:f.write('0.5 0 0 0 0 0\n')
        with self.assertRaises(ValueError): read_norms(self.folder,self.meta)

    def test_wrong_resolution(self):
        meta=dict(self.meta,dx=.25)
        with self.assertRaises(ValueError): read_norms(self.folder,meta)

    def test_duplicate_slice(self):
        rows=self.rows[:-1]+[self.rows[0]]
        (self.slices/'slice-0-0.csv').write_text(self.header+''.join(rows))
        with self.assertRaises(ValueError): read_slice(self.folder,self.meta)

    def test_wrong_slice_time_and_legacy_fields(self):
        path=self.slices/'slice-0-0.csv'
        path.write_text(self.header+''.join(self.rows).replace('.2,.5,','.1,.5,'))
        with self.assertRaises(ValueError): read_slice(self.folder,self.meta)
        path.write_text(self.header.replace('fz_ref','wx')+''.join(self.rows))
        with self.assertRaises(ValueError): read_slice(self.folder,self.meta)

    def test_nonfinite_slice(self):
        path=self.slices/'slice-0-0.csv';path.write_text(path.read_text().replace(',1,0,0,0,1',',nan,0,0,0,1'))
        with self.assertRaises(ValueError): read_slice(self.folder,self.meta)

    def test_writer_concurrent_leaves_and_opt_in(self):
        header=Path(__file__).resolve().parents[1]/'support/plot_output.hpp'
        source='#include "'+str(header)+'"\n'+r'''
#include <thread>
#include <vector>
int main(int argc,char** argv){
    std::string dir=argv[1];
    { radiationTests::SliceOutput disabled(false,dir,.2,.5,2);
      disabled.capture(0,0,.25,std::array<double,4>{1,0,0,0},0,
        [](int)->double{throw std::runtime_error("Disabled exporter accessed fields");}); }
    { radiationTests::SliceOutput missing(true,dir+"/absent",.2,.5,2);
      missing.capture(0,0,.25,std::array<double,4>{1,0,0,0},0,
        [](int)->double{throw std::runtime_error("No opt-in directory");}); }
    std::vector<std::thread> workers;
    for(int bi=0;bi<2;++bi)for(int bj=0;bj<2;++bj) workers.emplace_back([=]{
        radiationTests::SliceOutput slice(true,dir,.2,.5,2);
        for(int i=2*bi;i<2*bi+2;++i)for(int j=2*bj;j<2*bj+2;++j)for(int k=0;k<4;++k){
            std::array<double,4> actual{2,0,0,0},reference{1,0,0,0};
            slice.capture(-1+(i+.5)*.5,-1+(j+.5)*.5,-1+(k+.5)*.5,
                reference,0,[&](int f){return actual[f];});
            actual=reference;
        }
        slice.finish();
    });
    for(auto& worker:workers)worker.join();
}
'''
        for f in self.slices.glob('*.csv'):f.unlink()
        src=self.folder/'writer.cpp';src.write_text(source);exe=self.folder/'writer'
        subprocess.run(['g++','-std=c++23','-pthread','-I'+str(Path(__file__).resolve().parents[2]),str(src),'-o',str(exe)],check=True)
        subprocess.run([str(exe),str(self.folder)],check=True)
        self.assertEqual(len(list(self.slices.glob('*.csv'))),4)
        data=read_slice(self.folder,self.meta)
        self.assertTrue((data['er']==2).all())
        self.assertTrue((data['er_ref']==1).all())

if __name__=='__main__':unittest.main()
