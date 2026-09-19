"""Mechanical adapter for existing Octo-TIGER hydro/gravity scenarios."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, re, subprocess
from pathlib import Path


def execute(selected, arguments, plan=False):
    from verification_results import runner
    output_arg=runner.option_value(arguments,'--output')
    output=Path(output_arg or runner.default_output('run')).expanduser().resolve()
    levels, build=runner.positional_settings(arguments,None,'run')
    exe=runner.option_value(arguments,'--exe')
    root=Path(runner.option_value(arguments,'--root') or runner.SOURCE_ROOT).resolve()
    exe_path=Path(exe).expanduser() if exe else root/'build'/'octotiger'/'octotiger'
    selected=selected
    manifest={'schema_version':runner.SCHEMA_VERSION,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'source':{'commit':runner.git_value('rev-parse','HEAD'),'dirty':bool(runner.git_value('status','--porcelain'))},
      'harness':{'name':'verification_results','adapter':'scenario'},'build':runner.compiler_metadata(build,root),
      'execution':{'mode':'suite','thread_count':int(runner.option_value(arguments,'--threads') or 12),'resolution_levels':levels,'arguments':arguments},
      'tests':[],'artifacts':{'root':str(output),'plots':'plots','movies':'movies','logs':'logs'}}
    for path,value in selected:
        p=value['parameters']; item={'identifier':f"{value['family']}.{value['suite']}.{value['name']}",'family':value['family'],'suite':value['suite'],'name':value['name'],'regime':value['regime'],'descriptor':str(path.relative_to(runner.SOURCE_ROOT)),'parameters':p,'status':'planned'}
        item['descriptor_sha256']=hashlib.sha256(path.read_bytes()).hexdigest(); manifest['tests'].append(item)
    if plan:
        print(json.dumps(manifest,indent=2,sort_keys=True)); return 0
    output.mkdir(parents=True,exist_ok=True); (output/'logs').mkdir(exist_ok=True)
    failures=[]
    for item,(_,value) in zip(manifest['tests'],selected):
        p=value['parameters']; caseout=output/item['identifier']; caseout.mkdir(parents=True,exist_ok=True)
        config=(root/p['config']).resolve(); cmd=[str(exe_path),f'--config_file={config}']+p.get('arguments',[])
        log=caseout/'stdout.log'
        if not exe_path.is_file():
            item['status']='conditional'; item['reason']='octotiger executable unavailable'; continue
        with log.open('w') as stream:
            rc=subprocess.run(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
        item['command']=cmd; item['status']='passed' if rc==0 else 'failed'; item['returncode']=rc
        text=log.read_text(errors='replace')
        checks=[]
        for pattern in p.get('pass_regular_expressions',[]): checks.append({'pattern':pattern,'passed':bool(re.search(pattern,text))})
        item['checks']=checks
        if rc or any(not c['passed'] for c in checks): failures.append(item['identifier'])
    manifest['status']='failed' if failures else ('conditional' if any(x['status']=='conditional' for x in manifest['tests']) else 'complete')
    manifest['failures']=failures
    (output/'verification.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (output/'report.html').write_text('<html><body><h1>Hydro and gravity verification</h1><pre>'+json.dumps(manifest,indent=2)+'</pre></body></html>')
    return 1 if failures else (3 if manifest['status']=='conditional' else 0)
