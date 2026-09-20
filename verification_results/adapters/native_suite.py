"""Serial production-method suite. No claim of HPX/MPI or full application coverage."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf
from verification_results.radiation.damping_reference import damping_checks
from verification_results.radiation.validate_so import production_source

ROOT=Path(__file__).resolve().parents[2]
C=2.99792458e10

RAW_SCHEMAS={
    'samples.csv':'frame,t,x,E,Fx,gas_energy,momentum'.split(','),
    'history.csv':'t,mean_E,mean_gas,mean_momentum,mean_Fx,min_E,max_reduced_flux,gas_dt,rad_dt,subcycles,rcycle,hcycle,hydro_ghost_unchanged'.split(','),
    'exchanges.csv':'gas_step,event,rcycle,hcycle,time,interior_E,interior_Fx,halo_valid,hydro_unchanged'.split(','),
}

def sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(descriptor)
    finally:os.close(descriptor)

def atomic_write(path,payload):
    """Publish closed, synced bytes, then verify what the published path returns.

    This detects failed/changed writes at the check, not arbitrary corruption
    after the suite exits. The retained manifests support later revalidation.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(payload,str):payload=payload.encode('utf8')
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.pending',delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path);sync_directory(path.parent)
    if path.read_bytes()!=payload:raise RuntimeError(f'Artifact readback mismatch: {path}')

def dump(path,value):
    atomic_write(path,json.dumps(value,indent=2,allow_nan=False)+'\n')

def file_record(path):
    payload=Path(path).read_bytes()
    return {'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}

def check_file_records(folder,records):
    for name,expected in records.items():
        path=folder/name
        if not path.is_file() or file_record(path)!=expected:
            raise RuntimeError(f'Artifact integrity mismatch: {path}')

def read_csv_strict(path,columns,allow_empty=False):
    """Reject malformed, partial, nonfinite, or unexpected native CSV data."""
    raw=Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):
        raise ValueError(f'{path}: empty or unterminated CSV')
    rows=list(csv.reader(io.StringIO(raw.decode('utf8')),strict=True))
    if not rows or rows[0]!=columns:raise ValueError(f'{path}: incorrect CSV schema')
    if not allow_empty and len(rows)==1:raise ValueError(f'{path}: no data rows')
    integer_fields={'frame','gas_step','subcycles','rcycle','hcycle','hydro_ghost_unchanged','halo_valid','hydro_unchanged'}
    values=[]
    for number,row in enumerate(rows[1:],2):
        if len(row)!=len(columns) or any(not value.strip() for value in row):
            raise ValueError(f'{path}:{number}: incomplete CSV row')
        converted=[]
        for column,value in zip(columns,row):
            if column=='event':
                if value not in {'hydro','radiation','flux'}:raise ValueError(f'{path}:{number}: invalid event')
                converted.append(value);continue
            value=float(value)
            if not math.isfinite(value):raise ValueError(f'{path}:{number}: nonfinite {column}')
            if column in integer_fields and value!=int(value):raise ValueError(f'{path}:{number}: noninteger {column}')
            converted.append(value)
        values.append(tuple(converted))
    return np.array(values,dtype=[(column,'U16' if column=='event' else float) for column in columns])

def validate_raw(name,p,folder,n):
    data={filename:read_csv_strict(folder/filename,columns,allow_empty=filename=='exchanges.csv' and name!='boundary_subcycles')
          for filename,columns in RAW_SCHEMAS.items()}
    s=data['samples.csv'];h=data['history.csv'];events=data['exchanges.csv']
    frames=p['frames'];end=p['final_time_s']
    # These are serialization/time-grid checks, not physics acceptance bounds.
    time_atol=32*np.finfo(float).eps*abs(end)
    if len(s)!=frames*n or not np.array_equal(s['frame'],np.repeat(np.arange(frames),n)):
        raise ValueError(f'{folder}/samples.csv: incomplete or unordered frame/cell coverage')
    targets=np.linspace(0,end,frames)
    if not np.allclose(s['t'],np.repeat(targets,n),rtol=32*np.finfo(float).eps,atol=time_atol):
        raise ValueError(f'{folder}/samples.csv: incorrect frame times or missing final time')
    x=-p['length_cm']/2+(np.arange(n)+.5)*p['length_cm']/n
    if not np.allclose(s['x'],np.tile(x,frames),rtol=32*np.finfo(float).eps,atol=32*np.finfo(float).eps*p['length_cm']):
        raise ValueError(f'{folder}/samples.csv: incorrect cell coordinates')
    if len(h)<frames or h['t'][0]!=0 or not np.isclose(h['t'][-1],end,rtol=32*np.finfo(float).eps,atol=time_atol) or np.any(np.diff(h['t'])<=0):
        raise ValueError(f'{folder}/history.csv: incomplete/nonmonotone time history')
    if np.any(h['gas_dt'][1:]<=0) or not np.allclose(np.diff(h['t']),h['gas_dt'][1:],rtol=32*np.finfo(float).eps,atol=time_atol):
        raise ValueError(f'{folder}/history.csv: missing/inconsistent timesteps')
    if not all(np.any(np.isclose(h['t'],target,rtol=32*np.finfo(float).eps,atol=time_atol)) for target in targets):
        raise ValueError(f'{folder}/history.csv: missing frame history')
    if name=='boundary_subcycles':
        if not np.array_equal(np.unique(events['gas_step']),np.arange(len(h)-1)):
            raise ValueError(f'{folder}/exchanges.csv: incomplete gas-step coverage')
    elif len(events):raise ValueError(f'{folder}/exchanges.csv: unexpected exchange records')
    log=(folder/'run.log').read_text()
    completed=re.search(r'^completed '+re.escape(name)+r' N=(\d+) t=(\S+) steps=(\d+) c=(\S+)$',log,re.MULTILINE)
    if not completed or int(completed[1])!=n or int(completed[3])!=len(h)-1 or not math.isclose(float(completed[2]),end,rel_tol=5e-6):
        raise ValueError(f'{folder}/run.log: missing or inconsistent completion record')
    return data

def publish_raw(staging,folder):
    """Keep incomplete fixture outputs staged; publish validated snapshots only."""
    records={}
    for filename in [*RAW_SCHEMAS,'run.log']:
        payload=(staging/filename).read_bytes()
        atomic_write(folder/filename,payload)
        records[filename]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    check_file_records(folder,records)
    return records

def validate_movie(path,expected_frames,ffmpeg,log_path=None):
    # Decode every frame; a nonempty container or metadata-only probe is not enough.
    before=file_record(path)
    command=[ffmpeg,'-v','error','-xerror','-err_detect','explode','-i',str(path),
             '-map','0:v:0','-vsync','0','-f','framehash','-hash','sha256','-']
    process=subprocess.run(command,text=True,capture_output=True)
    if log_path is not None:atomic_write(log_path,process.stdout+process.stderr)
    if process.returncode:
        raise RuntimeError(f'Movie decode failed for {path}: {process.stderr.strip()}')
    frames=[line for line in process.stdout.splitlines() if line.strip() and not line.startswith('#')]
    if len(frames)!=expected_frames or any(len(line.split(','))!=6 for line in frames):
        raise RuntimeError(f'Movie frame count mismatch for {path}: decoded {len(frames)}, expected {expected_frames}')
    if file_record(path)!=before:raise RuntimeError(f'Movie changed during validation: {path}')
    return {'decoded_frames':len(frames),'decoder':'ffmpeg full framehash decode','artifact':before}

def seal_artifacts(folder,raw_records):
    check_file_records(folder,raw_records)
    validation=json.loads((folder/'movie-validation.json').read_text())
    check_file_records(folder,{'movie.mp4':validation['artifact']})
    names=[*RAW_SCHEMAS,'run.log','input.txt','comparison.csv','comparison.png','movie.mp4','movie.log','movie-validation.log','movie-validation.json','products.json']
    names += [path.relative_to(folder).as_posix() for path in sorted((folder/'frames').glob('*.png'))]
    records={name:file_record(folder/name) for name in names}
    manifest={'schema_version':1,'algorithm':'sha256','files':records}
    dump(folder/'artifacts.json',manifest)
    check_file_records(folder,records)
    return {'manifest':'artifacts.json','manifest_sha256':file_record(folder/'artifacts.json')['sha256'],
            'status':'passed','file_count':len(records)},records

def verify_artifacts(folder,expected=None):
    """Recheck retained results, optionally against the in-process sealed snapshot."""
    metadata=json.loads((folder/'run.json').read_text())
    integrity=metadata.get('artifact_integrity',{})
    if integrity.get('status')!='passed' or integrity.get('manifest')!='artifacts.json' or integrity.get('manifest_sha256')!=file_record(folder/'artifacts.json')['sha256']:
        raise ValueError(f'{folder}/artifacts.json: manifest does not match run metadata')
    manifest=json.loads((folder/'artifacts.json').read_text())
    if manifest.get('schema_version')!=1 or manifest.get('algorithm')!='sha256' or not manifest.get('files'):
        raise ValueError(f'{folder}/artifacts.json: invalid integrity manifest')
    records=manifest['files']
    for name in records:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Invalid artifact path')
    if expected is not None:check_file_records(folder,expected)
    check_file_records(folder,records)
    return True

def source_identity():
    def git(*args):
        p=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True)
        return p.stdout.strip() if p.returncode==0 else 'unknown'
    files=[ROOT/'src/radiation/rad_grid.cpp',ROOT/'verification_results/runner.py',
           ROOT/'verification_results/radiation/validate_so.py',ROOT/'verification_results/radiation/damping_reference.py']
    files+=list((ROOT/'octotiger').rglob('*.hpp'))+list((ROOT/'test_problems/radiation').glob('*.hpp'))
    files+=list((ROOT/'verification_results/radiation/tests_so').glob('*.inc'))
    files+=list((ROOT/'verification_results/adapters').glob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    commit=git('rev-parse','HEAD');state=git('status','--porcelain')
    if commit=='unknown':
        exported=(ROOT/'verification_results/SOURCE_VERSION').read_text().strip()
        if len(exported)==40 and all(c in '0123456789abcdef' for c in exported):commit=exported
    import scipy,matplotlib
    return {'commit':commit,'dirty':None if state=='unknown' else bool(state),
            'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,
            'files':hashes,'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}

def compile_fixture(out,n,pc,build,cxx,identity):
    generated=production_source(ROOT)
    checks=(ROOT/'verification_results/radiation/tests_so/checks.inc').read_text().split('int main() {')[0]
    if pc:
        needle='using M1=RadiationM1<Real,NDIM>;'
        assert generated.count(needle)==1
        generated=generated.replace(needle,'''using BaseM1=RadiationM1<Real,NDIM>;
struct M1:BaseM1 {
 static std::pair<ConservedState,ConservedState> reconstruct(
   ConservedState const&,ConservedState const& center,ConservedState const&) {
   center.checkState("piecewise constant test center");return {center,center};
 }
};''')
    fixture=(ROOT/'verification_results/radiation/tests_so/suite.inc').read_text()
    marker='    std::cout<<"completed "'
    if fixture.count(marker)!=1:raise RuntimeError('Fixture completion marker changed')
    # Destructors cannot report buffered close failures. Close the verification
    # output streams explicitly, inside the fixture's existing exception guard.
    fixture=fixture.replace(marker,'    for(auto stream:{&slices,&history,&exchange}) {stream->flush();stream->close();}\n'+marker)
    generated+='\n'+checks+'\n'+fixture
    flags={'debug':['-O0','-g'],'release':['-O2','-DNDEBUG'],'relwithdebinfo':['-O2','-g','-DNDEBUG']}[build.lower()]
    version=subprocess.check_output([cxx,'--version'],text=True)
    key=hashlib.sha256((generated+identity['sha256']+version+str(n)+str(flags)).encode()).hexdigest()
    directory=out/'.build'/key;directory.mkdir(parents=True,exist_ok=True)
    src=directory/'suite.cpp';exe=directory/'suite'
    command=[cxx,'-std=c++23',*flags,f'-DTEST_CELLS={n}','-I'+str(ROOT),str(src),'-o',str(exe)]
    if not exe.exists():
        atomic_write(src,generated)
        with (directory/'build.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    return exe,{'compiler':version.splitlines()[0],'build_type':build,'command':command,
                'generated_source_sha256':hashlib.sha256(generated.encode()).hexdigest(),
                'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}

def reference(name,p,t,x,n):
    L=p['length_cm'];dx=L/n;c=p['c_cm_s'];v=c*p['reduced_light_speed_ratio'];chi=p['chi_cm_inverse']
    E0=p['radiation_energy_erg_cm3'];A=p['amplitude'];width=p['width_cm']
    if name=='streaming_front_1d':
        def primitive(y):
            cycles=np.floor((y+L/2)/L);rem=y+L/2-cycles*L
            return cycles*L/2+np.minimum(rem,L/2)
        return 1e-10+(1-1e-10)*(primitive(x-v*t+dx/2)-primitive(x-v*t-dx/2))/dx
    if name=='thin_gaussian':
        E=np.full_like(x,E0)
        for image in range(-4,5):
            E+=A*.5*np.sqrt(np.pi)*width/dx*(erf((x-v*t+dx/2+image*L)/width)-erf((x-v*t-dx/2+image*L)/width))
        return E
    if name in {'thermal_relaxation','moving_scattering','boundary_subcycles'}:
        rho=p['rho_g_cm3'];gas=p['gas_internal_erg_cm3'];velocity=p['velocity_cm_s'];ratio=p['reduced_light_speed_ratio']
        if name=='thermal_relaxation':
            alpha=4*5.670374419e-5/c*((2/3)*1.6735575e-24/(rho*1.380649e-16))**4
            total=gas+E0/ratio
            def rhs(_,u):return [v*chi*(alpha*(total-u[0]/ratio)**4-u[0])]
            sol=solve_ivp(rhs,[0,p['final_time_s']],[E0],rtol=2e-12,atol=2e-14,dense_output=True)
            if not sol.success:raise RuntimeError(sol.message)
            return np.full_like(x,sol.sol(t)[0])
        Q0=.1*E0;mom=rho*velocity+Q0/(c*ratio)
        def rhs(_,u):
            E,Q=u;vel=(mom-Q/(c*ratio))/rho
            return [-ratio*chi*vel*Q,-v*chi*Q+ratio*chi*(4/3)*E*vel]
        sol=solve_ivp(rhs,[0,p['final_time_s']],[E0,Q0],rtol=2e-12,atol=2e-14,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        return np.full_like(x,sol.sol(t)[0])
    amplitude=A*np.sinc(1/n)
    if name in {'static_diffusion','diffusion_thicker'}:
        # Independent 2x2 linear moment system; matrix exponential includes finite
        # radiation inertia. Diffusion asymptote is checked separately.
        from scipy.linalg import expm
        k=2*np.pi/L
        matrix=np.array([[0,-v*k],[v*k/3,-v*chi]])
        factor=expm(matrix*t)[0,0]
        return E0+amplitude*factor*np.cos(k*x)
    value=E0+amplitude*np.cos(2*np.pi*(x-v*t)/L)
    return value*np.exp(-v*chi*t) if name=='damped_wave' else value

def check_exchange_trace(events,h):
    checks={}
    for step in np.unique(events['gas_step']):
        a=events[events['gas_step']==step];bounds=a[a['event']=='radiation'];hydro=a[a['event']=='hydro'];flux=a[a['event']=='flux']
        checks['boundary_counts']=checks.get('boundary_counts',True) and len(hydro)==1 and len(bounds)==len(flux)+1
        checks['independent_epochs']=checks.get('independent_epochs',True) and bool(np.all(np.diff(bounds['rcycle'])==1)) and len(set(bounds['hcycle']))==1
        checks['flux_epoch']=checks.get('flux_epoch',True) and bool(np.array_equal(flux['rcycle'],bounds['rcycle'][:-1]+1))
        checks['flux_interval_time']=checks.get('flux_interval_time',True) and bool(np.array_equal(flux['time'],bounds['time'][:-1]))
        checks['state_and_hydro']=checks.get('state_and_hydro',True) and bool(np.all(bounds['halo_valid']==1) and np.all(bounds['hydro_unchanged']==1))
        index=int(step)
        expected_times=np.linspace(h['t'][index],h['t'][index+1],int(h['subcycles'][index+1])+1)
        valid_times=len(bounds)==len(expected_times) and np.allclose(bounds['time'],expected_times,rtol=2e-14,atol=2e-15)
        checks['subcycle_times']=checks.get('subcycle_times',True) and bool(valid_times)
        if len(bounds)>2:checks['fresh_radiation_state']=checks.get('fresh_radiation_state',True) and bool(np.any(np.diff(bounds['interior_Fx'])!=0))
    checks['radiation_exchanges_exceed_hydro']=int(np.sum(events['event']=='radiation'))>int(np.sum(events['event']=='hydro'))
    # Exact absolute start/end checks against the production interval records.
    b=events[events['event']=='radiation'];checks['final_boundary_time']=abs(float(b['time'][-1])-h['t'][-1])<1e-14*max(h['t'][-1],1)
    return {key:bool(value) for key,value in checks.items()}

def evaluate(name,d,folder,n):
    p=d['parameters'];tol=d['tolerance_policy']
    data=validate_raw(name,p,folder,n);h=data['history.csv'];s=data['samples.csv']
    final=s[s['frame']==s['frame'].max()];ref=reference(name,p,float(final['t'][0]),final['x'],n)
    error=final['E']-ref
    norm=max(p['amplitude'],1e-30) if name not in {'thermal_relaxation','moving_scattering','boundary_subcycles','damped_wave','streaming_front_1d'} else p['radiation_energy_erg_cm3']
    l1=float(np.mean(abs(error)))/norm
    checks={'finite':bool(np.all(np.isfinite(s['E']))),'positivity':bool(np.min(h['min_E'])>=0),
      'reduced_flux':bool(np.max(h['max_reduced_flux'])<=1+tol['realizability_tolerance']),
      'hydro_ghost_preservation':bool(np.all(h['hydro_ghost_unchanged']==1)),
      'accuracy':l1 <= tol['l1_coefficient']/n}
    if name=='streaming_wave_pc':
        # Independent Fourier amplification for piecewise-constant VL:
        # half=u-.5*C*D(u), next=u-C*D(half), D=1-exp(-ik*dx).
        z=1-np.exp(-2j*np.pi/n)
        courant=C*h['gas_dt'][1:]/(p['length_cm']/n)
        amplification=np.prod(1-courant*z+.5*courant**2*z**2)
        discrete=p['radiation_energy_erg_cm3']+p['amplitude']*np.sinc(1/n)*np.real(amplification*np.exp(2j*np.pi*final['x']/p['length_cm']))
        checks['accuracy']=bool(np.max(abs(final['E']-discrete))<=5e-12)
        checks['exact_discrete_pc_update']=checks['accuracy']
    ratio=p['reduced_light_speed_ratio'];total=h['mean_gas']+h['mean_E']/ratio
    # Use radiation scale, not a huge inert gas reservoir, for the numerical budget.
    if name!='damped_wave':
        change=(h['mean_gas']-h['mean_gas'][0])+(h['mean_E']-h['mean_E'][0])/ratio
        scale=max(abs(total[0]),p['radiation_energy_erg_cm3']) if name in {'moving_scattering','boundary_subcycles'} else max(abs(h['mean_E'][0]),1)
        checks['energy_conservation']=bool(np.max(abs(change))/scale<=tol['conservation_relative'])
    else:
        expected=h['mean_E'][0]*np.exp(-C*p['chi_cm_inverse']*h['t'])
        checks['absorption_decay']=bool(np.max(abs(h['mean_E']-expected))<=2/n)
        checks.update(damping_checks(h,p))
    budget=h['mean_momentum']+h['mean_Fx']/(C*C*ratio)
    # Only conserved in periodic source-free transport or when feedback is enabled.
    if name!='thermal_relaxation':
        checks['momentum_conservation']=bool(np.max(abs(budget-budget[0]))/max(abs(budget[0]),p['radiation_energy_erg_cm3']/C)<=2e-11)
    if name.startswith('streaming_') or name in {'thin_gaussian','damped_wave'}:
        phase0=np.sum(s[s['frame']==0]['E']*np.exp(-2j*np.pi*final['x']/p['length_cm']))
        phase1=np.sum(final['E']*np.exp(-2j*np.pi*final['x']/p['length_cm']))
        phase_error=float(abs(np.angle(phase1/phase0*np.exp(2j*np.pi*C*p['final_time_s']/p['length_cm']))))
        checks['propagation_speed']=phase_error<=2*np.pi/n
        checks['symmetric_streaming_limit']=bool(np.max(abs(final['Fx']/C-final['E']))<=2e-12)
    else:phase_error=None
    if name=='thermal_relaxation':
        checks['relaxation_toward_equilibrium']=bool(h['mean_E'][-1]<h['mean_E'][0] and h['mean_gas'][-1]>h['mean_gas'][0])
    if name=='moving_scattering':checks['mechanical_work']=bool(h['mean_E'][-1]<h['mean_E'][0])
    if name=='boundary_subcycles':
        events=data['exchanges.csv']
        checks.update(check_exchange_trace(events,h))
    if name in {'static_diffusion','diffusion_thicker'}:
        k=2*np.pi/p['length_cm'];factor=float(np.dot(final['E']-1,np.cos(k*final['x']))/np.dot(np.cos(k*final['x']),np.cos(k*final['x'])))/(p['amplitude']*np.sinc(1/n))
        expected_diff=math.exp(-C*k*k*p['final_time_s']/(3*p['chi_cm_inverse']))
        checks['diffusion_scaling']=abs(factor-expected_diff)<=2/n+2*(k/p['chi_cm_inverse'])**2
    else:factor=expected_diff=None
    comparison=io.StringIO()
    np.savetxt(comparison,np.column_stack([final['x'],final['E'],ref,error]),delimiter=',',header='x,E,reference,signed_error',comments='')
    atomic_write(folder/'comparison.csv',comparison.getvalue())
    checks={key:bool(value) for key,value in checks.items()}
    return {'status':'passed' if all(checks.values()) else 'failed','cells':n,'L1':l1,
            'L2':float(np.sqrt(np.mean(error**2)))/norm,'Linf':float(np.max(abs(error)))/norm,
            'norm_scale':norm,'checks':checks,'phase_error_radians':phase_error,
            'diffusion_amplitude':factor,'diffusion_limit_amplitude':expected_diff,
            'timesteps':{k:[float(np.min(h[k][1:])),float(np.max(h[k][1:]))] for k in ['gas_dt','rad_dt','subcycles']}}

def visualize(d,folder,n,ffmpeg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read_csv_strict(folder/'samples.csv',RAW_SCHEMAS['samples.csv'])
    comparison_data=read_csv_strict(folder/'comparison.csv',['x','E','reference','signed_error'])
    if len(comparison_data)!=n:raise ValueError('Incomplete comparison cell coverage')
    comparison=np.column_stack([comparison_data[column] for column in comparison_data.dtype.names])
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
    axes[0].plot(comparison[:,0],comparison[:,1],'.-',label='Numerical')
    axes[0].plot(comparison[:,0],comparison[:,2],label='Reference');axes[0].legend();axes[0].set_ylabel('E [erg cm⁻³]')
    axes[1].plot(comparison[:,0],comparison[:,3]);axes[1].axhline(0,color='gray',lw=.5)
    axes[1].set(xlabel='x [cm]',ylabel='Signed error [erg cm⁻³]');fig.suptitle(d['name']+f' | N={n}')
    fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=120);atomic_write(folder/'comparison.png',buffer.getvalue());plt.close(fig)
    frames=folder/'frames';frames.mkdir()
    lo=float(np.min(s['E']));hi=float(np.max(s['E']));pad=max((hi-lo)*.1,1e-12)
    for frame in np.unique(s['frame']).astype(int):
        a=s[s['frame']==frame];ref=reference(d['name'],d['parameters'],float(a['t'][0]),a['x'],n)
        fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(a['x'],a['E'],'.-',label='Numerical');ax.plot(a['x'],ref,label='Reference')
        ax.set(xlabel='x [cm]',ylabel='E [erg cm⁻³]',ylim=(lo-pad,hi+pad),title=f"{d['name']} | t={a['t'][0]:.6g} s");ax.legend();fig.tight_layout()
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=100);atomic_write(frames/f'{frame:04d}.png',buffer.getvalue());plt.close(fig)
    pending=folder/'movie.pending.mp4'
    with (folder/'movie.log').open('w') as log:
        subprocess.run([ffmpeg,'-y','-framerate','4','-i',str(frames/'%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(pending)],stdout=log,stderr=subprocess.STDOUT,check=True)
        log.flush();os.fsync(log.fileno())
    validation=validate_movie(pending,len(np.unique(s['frame'])),ffmpeg,folder/'movie-validation.log')
    payload=pending.read_bytes()
    if {'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}!=validation['artifact']:
        raise RuntimeError('Movie changed after decode validation')
    atomic_write(folder/'movie.mp4',payload)
    dump(folder/'movie-validation.json',validation)
    return ['comparison.png','movie.mp4']

def web(out,results,embedded=False):
    def asset(path,mime):
        return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode() if embedded else path.relative_to(out).as_posix()
    parts=['<!doctype html><meta charset="utf-8"><title>Octo-TIGER radiation verification</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui}img,video{max-width:100%}article{border-top:1px solid #bbb;margin-top:24px}pre{white-space:pre-wrap}.failed{color:#a00}.conditional{color:#865700}</style><h1>Radiation verification</h1><p>Serial production-method checks; HPX/MPI delivery and full application coverage are separate. Reload to see completed runs.</p>']
    if not embedded and any(r['status']=='running' for r in results):
        parts.append('<meta http-equiv="refresh" content="5">')
    provenance=out/'source.json'
    if provenance.exists():
        source=json.loads(provenance.read_text())
        parts.append('<p>Source commit: <code>'+html.escape(source['commit'])+'</code>; dirty: '+str(source['dirty'])+'</p>')
    for r in results:
        parts.append(f'<article><h2>{html.escape(r["id"])}</h2><b class="{r["status"]}">{r["status"]}</b><p>{html.escape(r.get("reason",""))}</p>')
        parts.append('<p>Reference: '+html.escape(json.dumps(r.get('reference',{})))+'</p>')
        if r.get('runs'):
            parts.append('<pre>'+html.escape(json.dumps({k:r.get(k) for k in ['regime','orders','runs']},indent=2))+'</pre>')
            folder=out/r['id']/('l'+str(r['runs'][-1]['level']))
            for filename,mime in [('comparison.png','image/png'),('convergence.png','image/png'),('movie.mp4','video/mp4')]:
                path=folder/filename if filename!='convergence.png' else out/r['id']/filename
                if path.exists():
                    url=asset(path,mime);parts.append(f'<video controls src="{url}"></video>' if mime.startswith('video') else f'<img src="{url}" alt="{filename}">')
        if embedded and r.get('runs'):
            parts.append('<details><summary>All resolutions: raw data, logs, metadata and visual products</summary><ul>')
            for run in r['runs']:
                folder=out/r['id']/('l'+str(run['level']))
                for filename in ['run.json','input.txt','samples.csv','comparison.csv','history.csv','exchanges.csv','run.log','movie.log','movie-validation.log','movie-validation.json','products.json','artifacts.json','comparison.png','movie.mp4']:
                    path=folder/filename
                    if path.exists():
                        mime='application/octet-stream'
                        url=asset(path,mime)
                        label=r['id']+'-l'+str(run['level'])+'-'+filename
                        parts.append('<li><a download="'+html.escape(label)+'" href="'+url+'">'+html.escape(label)+'</a></li>')
            parts.append('</ul></details>')
        parts.append('</article>')
    atomic_write(out/('report.html' if embedded else 'index.html'),''.join(parts))

def execute(selected,arguments,plan=False):
    from verification_results import runner
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('settings',nargs='*');parser.add_argument('--output',type=Path)
    parser.add_argument('--threads',type=int,default=1);parser.add_argument('--cxx',default='g++');parser.add_argument('--ffmpeg',default='ffmpeg')
    opts=parser.parse_args(arguments)
    levels=[int(v) for v in opts.settings if v.isdigit()]
    builds=[v for v in opts.settings if not v.isdigit()];build=builds[-1] if builds else 'Release'
    if build.lower() not in {'debug','release','relwithdebinfo'}:raise ValueError('Invalid build type')
    if opts.threads!=1:raise ValueError('Serial production-method adapter requires --threads 1; use legacy application adapters for HPX threads')
    if len(builds)>1 or any(x<0 or x>5 for x in levels) or levels!=sorted(set(levels)):raise ValueError('Use distinct increasing resolution levels and one build type')
    out=runner.safe_output(opts.output or runner.default_output('suite'))
    manifest={'selected':[k for k,_ in selected],'levels':levels or 'descriptor defaults','build_type':build,'output':str(out),'thread_count':1}
    if plan:print(json.dumps(manifest,indent=2));return 0
    if out.exists() and any(out.iterdir()):raise ValueError('Suite output must be empty; retain prior results and choose a new directory')
    out.mkdir(parents=True,exist_ok=True);identity=source_identity();results=[];audits=[]
    dump(out/'source.json',identity)
    for identifier,d in selected:
        result={'id':identifier,'regime':d['regime'],'status':'running','runs':[],'reference':d['reference_data']}
        results.append(result);dump(out/'summary.json',results);web(out,results)
        if d['adapter']['name']!='native_suite':
            result.update(status='conditional',reason=d['parameters'].get('reason','Full application adapter retained; requires HPX/Silo, CMake, gnuplot and VisIt. Run this identifier through run/live on a supported build.'))
        else:
            for level in levels or d['resolution_levels']:
                folder=out/identifier/f'l{level}';folder.mkdir(parents=True)
                p=d['parameters'];n=8*2**level
                meta={'schema_version':1,'source':identity,'descriptor':d,'descriptor_sha256':hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),
                      'level':level,'cells':n,'thread_count':1,'backend':'serial production methods; fixture communication',
                      'options':p,'status':'running'}
                dump(folder/'run.json',meta)
                sealed=None
                try:
                    exe,build_meta=compile_fixture(out,n,p['reconstruction']=='piecewise_constant',build,opts.cxx,identity)
                    meta['build']=build_meta
                    values=[d['name'],p['length_cm'],p['final_time_s'],p['chi_cm_inverse'],p['amplitude'],p['width_cm'],p['velocity_cm_s'],p['reduced_light_speed_ratio'],p['rho_g_cm3'],p['gas_internal_erg_cm3'],p['radiation_energy_erg_cm3'],p['frames']]
                    atomic_write(folder/'input.txt',' '.join(map(str,values))+'\n')
                    staging=folder/'.raw-pending';staging.mkdir()
                    with (staging/'run.log').open('w') as log:
                        subprocess.run([str(exe),str(folder/'input.txt'),str(staging)],stdout=log,stderr=subprocess.STDOUT,check=True)
                        log.flush();os.fsync(log.fileno())
                    validate_raw(d['name'],p,staging,n)
                    raw_records=publish_raw(staging,folder)
                    status=evaluate(d['name'],d,folder,n);status['level']=level
                    meta.update(status);meta['products']=visualize(d,folder,n,opts.ffmpeg)
                    dump(folder/'products.json',{'run_metadata':'run.json','options':p,'source_commit':identity['commit'],'products':meta['products']})
                    meta['artifact_integrity'],sealed=seal_artifacts(folder,raw_records)
                    status['artifact_integrity']=meta['artifact_integrity']
                except Exception as error:
                    status={'status':'failed','level':level,'cells':n,'error':str(error)};meta.update(status)
                    with (folder/'run.log').open('a') as log:log.write('\nHARNESS FAILURE: '+repr(error)+'\n')
                dump(folder/'run.json',meta)
                if sealed is not None:
                    sealed.update({name:file_record(folder/name) for name in ['run.json','artifacts.json']})
                    audits.append((folder,result,status,meta,sealed))
                result['runs'].append(status);dump(out/'summary.json',results);web(out,results)
            good=[r for r in result['runs'] if 'L1' in r]
            orders=[math.log(a['L1']/b['L1'])/math.log(b['cells']/a['cells']) if a['L1']>0 and b['L1']>0 else None for a,b in zip(good,good[1:])]
            if good and max(r['L1'] for r in good)<1e-12:orders=[]
            result['orders']=orders
            required=d['tolerance_policy']['convergence_required'];threshold=d['tolerance_policy']['min_order']
            result['convergence_status']='not_applicable' if not required else 'conditional' if not orders else 'passed' if orders[-1] is not None and orders[-1]>=threshold else 'failed'
            result['status']='failed' if any(r['status']=='failed' for r in result['runs']) or result['convergence_status']=='failed' else 'conditional' if result['convergence_status']=='conditional' else 'passed'
            if len(good)>1 and orders:
                try:
                    import matplotlib.pyplot as plt
                    fig,ax=plt.subplots();ax.loglog([r['cells'] for r in good],[r['L1'] for r in good],'o-');ax.set(xlabel='Cells per dimension',ylabel='Normalized L1',title=d['name']);fig.tight_layout();buffer=io.BytesIO();fig.savefig(buffer,format='png');atomic_write(out/identifier/'convergence.png',buffer.getvalue());plt.close(fig)
                except Exception as error:result.update(status='failed',reason='Convergence plot failed: '+str(error))
        dump(out/'summary.json',results);web(out,results)
        print(identifier,result['status'],result.get('orders',[]),flush=True)
    def audit_completed():
        changed=False
        for folder,result,status,meta,sealed in audits:
            if status.get('artifact_integrity',{}).get('status')=='failed':continue
            try:verify_artifacts(folder,sealed)
            except Exception as error:
                changed=True;failure='Final artifact audit failed: '+str(error)
                status.update(status='failed',error=failure,artifact_integrity={'status':'failed'})
                meta.update(status);result.update(status='failed',reason=failure)
                dump(folder/'run.json',meta)
                with (folder/'run.log').open('a') as log:log.write('\nHARNESS FAILURE: '+failure+'\n')
        if changed:dump(out/'summary.json',results);web(out,results)
        return changed
    audit_completed()
    web(out,results,embedded=True)
    # Embedding reads the persisted products again. Recheck after that read so
    # a late mutation cannot retain an earlier passing summary/exit status.
    if audit_completed():web(out,results,embedded=True)
    failed=any(r['status']=='failed' for r in results);conditional=any(r['status']=='conditional' for r in results)
    return 1 if failed else 3 if conditional else 0
