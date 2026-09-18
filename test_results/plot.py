#!/usr/bin/env python3
"""Produce PNG/PDF profiles, slice/error maps, convergence plots, CSV and HTML."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from results import (CASES, CONSERVATION_FILE, FIELDS, NORMS,
                     digest, order, read_conservation, read_norms, read_slice, write_json)

LABELS = dict(er='E', fx='Fx', fy='Fy', fz='Fz')
SUPPORTS_CGS = True
FIELD_UNITS = dict(er='erg/cm³', fx='erg/(cm² s)', fy='erg/(cm² s)', fz='erg/(cm² s)')
CONSERVATION_PRODUCTS = ('conservation.png', 'conservation.pdf', 'conservation.csv',
                         'conservation-summary.csv', 'conservation-summary.json',
                         'conservation.units.json')
TITLES = dict(streaming_wave='Streaming wave',streaming_front='Streaming front',
              gaussian_pulse='Gaussian pulse',equilibrium_sphere='Equilibrium sphere')

def is_cgs(meta):
    return meta.get('units', {}).get('system') == 'CGS'

def field_label(meta, field, suffix=''):
    unit = ' (' + FIELD_UNITS[field] + ')' if is_cgs(meta) else ''
    return LABELS[field] + suffix + unit

def save(fig, path):
    fig.savefig(path.with_suffix('.png'),dpi=150,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)

def scientific(ax, which='both'):
    ax.ticklabel_format(axis=which, style='sci', scilimits=(-3,3), useOffset=False)
    ax.grid(alpha=.22)

def write_records(path, records, fieldnames=None):
    """Write portable summaries without lowering the precision of measurements."""
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({key: ('' if value is None else f'{value:.17e}'
                if isinstance(value, float) else value) for key, value in record.items()})

def plot_conservation(folder, meta, target, budget):
    target.mkdir(parents=True, exist_ok=True)
    if budget is None:
        # An old plot must not survive removal of its diagnostic input.
        for name in CONSERVATION_PRODUCTS:
            (target/name).unlink(missing_ok=True)
        return
    data = budget['history']
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), layout='constrained')
    for row, field in enumerate(FIELDS):
        unit = ('erg' if field == 'er' else 'erg cm/s') if is_cgs(meta) else 'code units'
        label = f'∫ {LABELS[field]} dV ({unit})'
        ax = axes[row, 0]
        ax.plot(data['t'], data[field], label='Measured total', lw=1.8)
        ax.plot(data['t'], data[field][0]-data[field+'_boundary']+data[field+'_source'],
                '--', label='Initial − boundary + source', lw=1.4)
        ax.set(ylabel=label, xlabel='Time (s)' if is_cgs(meta) else 'Time')
        scientific(ax); ax.legend(fontsize=10)
        ax = axes[row, 1]
        ax.axhline(0, color='.6', lw=.7)
        ax.plot(data['t'], budget['residuals'][field], color='#b4422e', lw=1.5)
        ax.set(ylabel=f'{LABELS[field]} budget residual ({unit})',
               xlabel='Time (s)' if is_cgs(meta) else 'Time')
        scientific(ax)
    fig.suptitle(f"{TITLES[meta['case']]} • {meta['cells']}³ cells • full-domain radiation conservation\n"
                 'Residual = total − initial + outward boundary transport − source change', fontsize=14)
    save(fig, target/'conservation')
    shutil.copyfile(folder/CONSERVATION_FILE, target/'conservation.csv')
    write_records(target/'conservation-summary.csv', budget['summary'])
    write_json(target/'conservation-summary.json', budget['summary'])
    units = dict(t='s' if is_cgs(meta) else 'code time',
                 volume='cm^3' if is_cgs(meta) else 'code volume')
    for field in FIELDS:
        for suffix in ('', '_boundary', '_source'):
            units[field+suffix] = ('erg' if field == 'er' else 'erg cm/s') if is_cgs(meta) else 'code units'
    write_json(target/'conservation.units.json', units)

def conservation_section(meta, rel, budget):
    if budget is None:
        return '<h3>Radiation conservation</h3><p>Conservation diagnostics unavailable; rerun this test.</p>'
    rows = []
    for record in budget['summary']:
        values = [html.escape(LABELS[record['field']]+' ('+record['integral_units']+')')]
        values += [fmt(record[name]) for name in ('initial', 'final', 'raw_change', 'boundary',
            'source', 'residual', 'normalized_error', 'max_normalized_error')]
        rows.append('<tr>'+''.join('<td>'+value+'</td>' for value in values)+'</tr>')
    return f'''<h3>Radiation conservation</h3>
        <p>Full-domain integrals Q = ∫(E, Fx, Fy, Fz) dV. The budget residual is
        R(t) = Q(t) − Q(0) + B(t) − S(t), where B is cumulative net outward boundary transport
        and S is the cumulative signed change actually applied by radiation source updates.
        Physical sources and boundary transport can change the radiation totals.
        Radiation momentum is ∫F dV / c²; division by c² gives the same relative budget errors.</p>
        <p>Normalized errors are |R(t)| / scale(t). For E, scale = max(|E₀|, |E(t)|, |B_E(t)|, |S_E(t)|).
        For each flux component, scale also includes c times this energy scale and that component's
        |Q₀|, |Q(t)|, |B(t)| and |S(t)|. This remains defined for zero net flux.
        A zero scale with zero residual reports zero; no fixed physical-unit floor is used.
        Maximum errors cover all recorded timesteps.</p>
        <div class="scroll"><table><thead><tr><th>Integral</th><th>Initial</th><th>Final</th>
        <th>Raw change</th><th>Boundary B</th><th>Source S</th><th>Residual R</th>
        <th>Final |R|/scale</th><th>Max |R|/scale</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
        <p><a href="{rel}/conservation.csv">Conservation history CSV</a> ·
        <a href="{rel}/conservation-summary.csv">Budget summary CSV</a> ·
        <a href="{rel}/conservation-summary.json">Budget summary JSON</a> ·
        <a href="{rel}/conservation.units.json">CSV units</a> ·
        <a href="{rel}/conservation.png">History PNG</a> ·
        <a href="{rel}/conservation.pdf">History PDF</a></p>
        <img loading="lazy" src="{rel}/conservation.png" alt="Energy and three flux totals with source and boundary corrected conservation residuals">'''

def plot_run(folder, meta, target):
    data=read_slice(folder,meta); n=meta['cells']; dx=meta['dx']; length=meta['length']
    target.mkdir(parents=True,exist_ok=True)
    excess=meta['case'] in ('gaussian_pulse','equilibrium_sphere')
    suffix=' − background' if excess else ''
    length_unit=' cm' if is_cgs(meta) else ''
    time_unit=' s' if is_cgs(meta) else ''
    x_label='x (cm)' if is_cgs(meta) else 'x'
    y_label='y (cm)' if is_cgs(meta) else 'y'
    origin=meta['origin']
    title=f"{TITLES[meta['case']]} • {n}³ cells • t={meta['time']:g}{time_unit} • {origin}"
    extent=(-length/2,length/2,-length/2,length/2)
    for field in FIELDS:
        numerical=data[field]; reference=data[field+'_ref']; error=numerical-reference
        background=meta['background'] if field=='er' and excess else 0
        shown_label=field_label(meta,field,suffix if field=='er' else '')
        low=min(numerical.min(),reference.min())-background
        high=max(numerical.max(),reference.max())-background
        if high==low: low-=1e-15;high+=1e-15
        limit=max(float(np.max(np.abs(error))),1e-30)
        fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
        for ax,values,label,cmap,vmin,vmax in zip(axes,
            (numerical-background,reference-background,error),
            ('Numerical','Reference','Numerical − reference'),
            ('viridis','viridis','RdBu_r'),(low,low,-limit),(high,high,limit)):
            im=ax.imshow(values.T,origin='lower',extent=extent,interpolation='nearest',
                         cmap=cmap,vmin=vmin,vmax=vmax)
            ax.set(title=label,xlabel=x_label,ylabel=y_label)
            ax.ticklabel_format(axis='both',style='sci',scilimits=(-3,3),useOffset=False)
            formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-2,2))
            fig.colorbar(im,ax=ax,shrink=.8,format=formatter)
        fig.suptitle(title+f'\n{shown_label}, cell layer at z=Δx/2={dx/2:.6e}{length_unit}',fontsize=15)
        save(fig,target/f'slice_{field}')
    fig,axes=plt.subplots(4,2,figsize=(13,14),layout='constrained')
    x=data['x'][:,n//2]
    for row,field in enumerate(FIELDS):
        reference=data[field+'_ref'][:,n//2];numerical=data[field][:,n//2]
        background=meta['background'] if field=='er' and excess else 0
        label=field_label(meta,field,suffix if field=='er' else '')
        ax=axes[row,0]
        ax.plot(x,reference-background,'k--',label='Reference',lw=1.8)
        ax.plot(x,numerical-background,'o-',label='Numerical',ms=3,lw=1)
        ax.set(xlabel=x_label,ylabel=label);scientific(ax);ax.legend(fontsize=10)
        ax=axes[row,1];ax.axhline(0,color='.6',lw=.7)
        ax.plot(x,numerical-reference,color='#b4422e',lw=1.5)
        ax.set(xlabel=x_label,ylabel=field_label(meta,field,' numerical − reference'));scientific(ax)
    fig.suptitle(title+f'\nCell row at y=z=Δx/2={dx/2:.6e}{length_unit}; matched cell-average reference',fontsize=14)
    save(fig,target/'profiles')
    # Preserve paired values in a single portable table, including signed errors.
    names=list(data.dtype.names)+[f+'_error' for f in FIELDS]
    with (target/'slice.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(names)
        for row in data.ravel():
            values=[row[name] for name in data.dtype.names]+[row[field]-row[field+'_ref'] for field in FIELDS]
            writer.writerow([f'{v:.17e}' for v in values])
    if is_cgs(meta):
        units = dict(t='s', dx='cm', x='cm', y='cm', z='cm')
        for field in FIELDS:
            for suffix in ('', '_ref', '_error'):
                units[field+suffix] = 'erg/cm^3' if field=='er' else 'erg/(cm^2 s)'
        write_json(target/'slice.units.json',units)

def convergence(runs, target):
    case=runs[0][1]['case']
    meta=runs[0][1]
    time_unit=' s' if is_cgs(meta) else ''
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,field in zip(axes.ravel(),FIELDS):
        hs=np.array([m['dx'] for _,m in runs])
        any_nonzero=False
        for norm,marker in zip(NORMS,('o','s','^')):
            errors=np.array([m['norms'][field][norm] for _,m in runs])
            valid=errors>0
            if valid.any():
                ax.loglog(hs[valid],errors[valid],marker+'-',label='L∞' if norm=='Linf' else norm)
                any_nonzero=True
        if not any_nonzero: ax.text(.5,.5,'All norms are zero',ha='center',transform=ax.transAxes)
        base=runs[0][1]['norms'][field]['L1']
        if len(runs)>1 and base>0:
            for power,style in ((1,':'),(2,'--')):
                ax.loglog(hs,base*(hs/hs[0])**power,style,color='.55',lw=1,
                          label=f'Δx^{power} guide')
        unit=' ('+FIELD_UNITS[field]+')' if is_cgs(meta) else ''
        ax.set(xlabel='Δx (cm; finer →)' if is_cgs(meta) else 'Δx (finer →)',
               ylabel='Volume-normalized error'+unit,title=LABELS[field])
        ax.invert_xaxis();ax.grid(True,which='both',alpha=.2)
        if any_nonzero: ax.legend(fontsize=10)
    config=runs[0][1].get('config',{})
    cap=float(config.get('hard_dt',0))
    timestep=(f'Δt ≤ min(CFL limit, {cap:.6e}{time_unit})' if cap>0 else 'Δt follows the production CFL step')
    fig.suptitle(f"{TITLES[case]} • full-volume norms • t={runs[0][1]['time']:g}{time_unit}\n"
                f"{runs[0][1]['origin']}; {timestep}",fontsize=15)
    save(fig,target/'convergence')

def fmt(value): return '—' if value is None else f'{value:.6e}'

def render(batch):
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13})
    batch=Path(batch).resolve(); output=batch/'plots';output.mkdir(exist_ok=True)
    groups={};skipped=[]
    for path in sorted(batch.glob('*/l*/run.json')):
        meta=json.loads(path.read_text())
        if meta.get('status')!='complete': skipped.append(str(path.relative_to(batch)));continue
        if meta['case'] not in CASES: raise ValueError(f'Unknown case in {path}')
        meta['norms']=read_norms(path.parent,meta)
        groups.setdefault(meta['case'],[]).append((path.parent,meta))
    if not groups: raise ValueError(f'No complete runs in {batch}')
    records=[];conservation_records=[];sections=[]
    for case,runs in groups.items():
        runs.sort(key=lambda r:-r[1]['dx'])
        first=runs[0][1]; seen=set()
        for folder,meta in runs:
            for key in ('time','length','c','background','comparison_signature','executable_sha256','origin','units'):
                if meta.get(key)!=first.get(key): raise ValueError(f'Cannot mix different {key} in a convergence series')
            if meta['dx'] in seen: raise ValueError('Duplicate resolution in convergence series')
            seen.add(meta['dx'])
        case_dir=output/case;case_dir.mkdir(exist_ok=True)
        convergence_files=tuple((case_dir/'convergence').with_suffix(suffix)
                                for suffix in ('.png','.pdf'))
        if len(runs)>=2:
            convergence(runs,case_dir)
            convergence_html=(f'<p><a href="{case}/convergence.pdf">Convergence PDF</a></p>'
                              f'<img src="{case}/convergence.png" alt="Four-field convergence plot">')
        else:
            # A single resolution contains three norms, but no refinement trend.
            # Remove an older plot if a formerly complete finer run was omitted.
            for path in convergence_files:
                path.unlink(missing_ok=True)
            convergence_html='<p>Convergence becomes available after two resolutions are complete.</p>'
        table=[];details=[]; previous=None
        for folder,meta in runs:
            run_dir=case_dir/f"l{meta['level']}"
            budget=read_conservation(folder,meta)
            signature=dict(meta=meta,renderer=digest(Path(__file__)),
                           reader=digest(Path(__file__).with_name('results.py')),
                           conservation=digest(folder/CONSERVATION_FILE) if budget is not None else None,
                           slices={p.name:digest(p) for p in sorted((folder/'radiation-slices').glob('slice-*.csv'))})
            cached=run_dir/'plot-inputs.json'
            products=[run_dir/(stem+suffix) for stem in ('profiles',*(f'slice_{f}' for f in FIELDS))
                      for suffix in ('.png','.pdf')]+[run_dir/'slice.csv']
            if is_cgs(meta): products.append(run_dir/'slice.units.json')
            if budget is not None: products.extend(run_dir/name for name in CONSERVATION_PRODUCTS)
            try:
                reusable=json.loads(cached.read_text())==signature and all(p.is_file() and p.stat().st_size for p in products)
            except (OSError,ValueError):
                reusable=False
            if not reusable:
                plot_run(folder,meta,run_dir)
                plot_conservation(folder,meta,run_dir,budget)
                write_json(cached,signature)
            elif budget is None:
                # Remove stale products even if a restored cache describes missing input.
                plot_conservation(folder,meta,run_dir,None)
            if budget is not None:
                for record in budget['summary']:
                    conservation_records.append(dict(case=case,level=meta['level'],
                        cells_per_side=meta['cells'],time=meta['time'],origin=meta['origin'],**record))
            energy_orders={}
            for field in FIELDS:
                for norm in NORMS:
                    error=meta['norms'][field][norm]
                    rate=None if previous is None else order(previous['norms'][field][norm],error,previous['dx'],meta['dx'])
                    records.append(dict(case=case,level=meta['level'],cells_per_side=meta['cells'],
                        dx=meta['dx'],time=meta['time'],field=field,norm=norm,error=error,order=rate,
                        origin=meta['origin'], dx_units='cm' if is_cgs(meta) else 'code length',
                        time_units='s' if is_cgs(meta) else 'code time',
                        error_units=FIELD_UNITS[field] if is_cgs(meta) else 'code units'))
                    if field=='er': energy_orders[norm]=rate
            vals=[str(meta['cells']),*(fmt(meta['norms']['er'][norm]) for norm in NORMS),
                  *(fmt(energy_orders[norm]) for norm in NORMS)]
            table.append('<tr>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
            rel=f"{case}/l{meta['level']}"
            details.append(f'''<details {'open' if len(runs)==1 else ''}><summary>{meta['cells']}³ cells — level {meta['level']}</summary>
                <p>Slice at z = {meta['dx']/2:.6e}{' cm' if is_cgs(meta) else ''}. Profiles at y = z = Δx/2.
                <a href="{rel}/slice.csv">Paired values CSV</a> · <a href="{rel}/profiles.pdf">Profiles PDF</a>
                {(' · <a href="'+rel+'/slice.units.json">CSV units</a>') if is_cgs(meta) else ''}</p>
                <img loading="lazy" src="{rel}/slice_er.png" alt="Energy numerical, reference, and signed error">
                <img loading="lazy" src="{rel}/profiles.png" alt="Radiation profiles and signed errors">
                <details><summary>Flux slice maps</summary>{''.join(f'<p><a href="{rel}/slice_{f}.pdf">{LABELS[f]} PDF</a></p><img loading="lazy" src="{rel}/slice_{f}.png" alt="{LABELS[f]} slice">' for f in FIELDS[1:])}</details>
                {conservation_section(meta,rel,budget)}
                </details>''')
            previous=meta
        caveat=('Exact nonlinear M1 streaming reference.' if case.startswith('streaming') else
                'Linearized / diffusion-limit reference; finite-amplitude nonlinear M1 has a model error floor.')
        note='One resolution: observed order requires at least two.' if len(runs)==1 else ''
        sections.append(f'''<section id="{case}"><h2>{TITLES[case]}</h2><p>{caveat} {note}</p>
            {convergence_html}
            <h3>Energy errors{' (erg/cm³)' if is_cgs(first) else ''} and observed orders</h3><div class="scroll"><table><thead><tr>
            <th>N</th><th>L1</th><th>L2</th><th>L∞</th><th>p(L1)</th><th>p(L2)</th><th>p(L∞)</th>
            </tr></thead><tbody>{''.join(table)}</tbody></table></div>{''.join(details)}</section>''')
    with (output/'errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
        for record in records:
            writer.writerow({k:('' if v is None else f'{v:.17e}' if isinstance(v,float) else v) for k,v in record.items()})
    write_json(output/'errors.json',records)
    conservation_keys=('case','level','cells_per_side','time','origin','field','initial','final',
        'raw_change','boundary','source','residual','normalization_scale','normalized_error',
        'max_abs_residual','max_normalized_error','integral_units')
    write_records(output/'conservation.csv',conservation_records,fieldnames=conservation_keys)
    write_json(output/'conservation.json',conservation_records)
    origins=', '.join(sorted({m['origin'] for runs in groups.values() for _,m in runs}))
    unit_note=('CGS: positions and cell sizes in cm, time in s, E and its errors in erg/cm³, '
               'Fx,Fy,Fz and their errors in erg/(cm² s). CSV numbers use these same units.'
               if all(is_cgs(m) for runs in groups.values() for _,m in runs) else
               'Units are shown on each figure and in the errors CSV; runs without unit metadata use code units.')
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Octo-TIGER radiation results</title><style>
    body{{font:18px/1.5 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#172335;background:#fafbfc}}
    h1,h2,h3{{line-height:1.2}}a{{color:#0755a2}}img{{width:100%;height:auto;background:white}}
    section{{margin:50px 0;border-top:2px solid #b9c7d8;padding-top:24px}}details{{margin:18px 0;padding:14px;background:white;border:1px solid #b9c7d8;border-radius:6px}}
    summary{{cursor:pointer;font-size:21px;font-weight:600}}table{{border-collapse:collapse;font:16px ui-monospace,monospace;width:100%}}
    th,td{{padding:10px;text-align:right;border-bottom:1px solid #cbd5e1;white-space:nowrap}}.scroll{{overflow-x:auto}}
    .note{{padding:16px;background:#e8eff6;border-left:4px solid #326494}}
    </style><h1>Octo-TIGER radiation results</h1><p>{html.escape(origins)}</p>
    <p>{' · '.join(f'<a href="#{c}">{TITLES[c]}</a>' for c in groups)} · <a href="errors.csv">All norms/orders CSV</a>
    · <a href="conservation.csv">All conservation budgets CSV</a> · <a href="conservation.json">Conservation JSON</a>
    {(' · <a href="../movies.html">Movie gallery</a>') if (batch/'movies.html').exists() else ''}</p>
    <div class="note">Norms use the entire 3D domain: L1 = Σ|e|ΔV/V, L2 = √(Σe²ΔV/V), L∞ = max|e|.
    Orders are p = log(error coarse / error fine) / log(Δx coarse / Δx fine), measured at the same final time.
    Zero norms have no reported order. {unit_note}
    The central cell layer moves toward z=0 with refinement; each numerical value is compared to its own matched cell average.</div>
    <p>The first-order time update can limit smooth-case convergence. A moving discontinuity can have different orders in different norms.
    These figures show measurements; they do not apply the CTest pass/fail thresholds.</p>
    {('<p>Incomplete runs omitted: '+html.escape(', '.join(skipped))+'</p>') if skipped else ''}
    {''.join(sections)}</html>'''
    (output/'index.html').write_text(page)
    return output/'index.html'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path)
    try: print(render(p.parse_args().batch))
    except (OSError,ValueError) as e: sys.exit(str(e))
