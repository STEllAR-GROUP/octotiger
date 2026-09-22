"""Read-only post-completion audit of a three-build native verification matrix.

Usage: python -m verification_results.audit_artifacts MATRIX_PATH --decode
Current-source differences remain strict failures, but do not stop the audit of
retained products. This distinguishes source drift from corrupted artifacts.
"""
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

from verification_results import runner
from verification_results.adapters import native_suite as suite

embeddedFiles=['run.json','input.txt','samples.csv','comparison.csv','history.csv','exchanges.csv',
                'run.log','movie.log','movie-validation.log','movie-validation.json','products.json',
                'artifacts.json','comparison.png','movie.mp4']


class Embedded(HTMLParser):
    def __init__(self):
        super().__init__();self.files={}

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='a' and 'download' in attrs and attrs.get('href','').startswith('data:'):
            payload=base64.b64decode(attrs['href'].split(',',1)[1],validate=True)
            self.files[attrs['download']]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}


def audit(root,decode=False):
    root=Path(root)
    descriptors={name:record[1] for name,record in runner.descriptors().items() if record[1]['adapter']['name']=='native_suite'}
    result={'status':'passed','builds':{},'errors':[],'native_runs':0,'decoded_movies':0}
    def require(condition,message):
        if not condition:raise ValueError(message)
    for build in ['Debug','Release','RelWithDebInfo']:
        folder=root/build/'radiation';details={'cases':{},'checks':[]};result['builds'][build]=details
        try:
            summary=json.loads((folder/'summary.json').read_text())
            source=json.loads((folder/'source.json').read_text())
            require(source['sha256']==hashlib.sha256(json.dumps(source['files'],sort_keys=True).encode()).hexdigest(),'Source hash-map digest mismatch')
            details['checks'].append('recorded source hash-map digest is internally consistent')
            details['source_changes_since_run']=[]
            for name,expected in source['files'].items():
                current=suite.root/name
                actual=hashlib.sha256(current.read_bytes()).hexdigest() if current.is_file() else None
                if actual!=expected:
                    result['status']='failed'
                    details['source_changes_since_run'].append({'path':name,'recorded_sha256':expected,'current_sha256':actual})
                    result['errors'].append({'build':build,'kind':'source_mismatch','error':f'Source changed since recorded identity: {name}'})
            if not details['source_changes_since_run']:details['checks'].append('source identity and current source hashes agree')
            embedded=Embedded();embedded.feed((folder/'report.html').read_text())
            selected={entry['id']:entry for entry in summary if entry['id'] in descriptors}
            require(set(selected)==set(descriptors),'Native case coverage is incomplete')
            for identifier,descriptor in descriptors.items():
                case=selected[identifier];runDetails=[];details['cases'][identifier]=runDetails
                require(case['status']=='passed',f'{identifier}: aggregate status is {case["status"]}')
                require([run['level'] for run in case['runs']]==[0,1,2],f'{identifier}: resolution coverage mismatch')
                for run in case['runs']:
                    path=folder/identifier/f'l{run["level"]}'
                    meta=json.loads((path/'run.json').read_text())
                    require(run['status']==meta['status']=='passed',f'{path}: summary/run status mismatch')
                    require(meta['source']==source,f'{path}: run/source identity mismatch')
                    require(meta['descriptor']==descriptor,f'{path}: descriptor mismatch')
                    require(meta['build']['build_type']==build,f'{path}: build type mismatch')
                    command=meta['build']['command']
                    generated=Path(command[-3]);executable=Path(command[-1])
                    require(command[-2]=='-o',f'{path}: unrecognized recorded compiler command')
                    require(hashlib.sha256(generated.read_bytes()).hexdigest()==meta['build']['generated_source_sha256'],f'{path}: generated-source hash mismatch')
                    require(hashlib.sha256(executable.read_bytes()).hexdigest()==meta['build']['executable_sha256'],f'{path}: executable hash mismatch')
                    require(meta['level']==run['level'] and meta['cells']==run['cells'],f'{path}: resolution metadata mismatch')
                    for key,value in run.items():
                        require(meta.get(key)==value,f'{path}: summary/run field mismatch: {key}')
                    suite.verifyArtifacts(path)
                    suite.validateRaw(descriptor['name'],descriptor['parameters'],path,meta['cells'])
                    movie=json.loads((path/'movie-validation.json').read_text())
                    require(movie['decoded_frames']==descriptor['parameters']['frames'],f'{path}: recorded movie frame mismatch')
                    if decode:
                        suite.validateMovie(path/'movie.mp4',descriptor['parameters']['frames'],'ffmpeg')
                        result['decoded_movies']+=1
                    for filename in embeddedFiles:
                        label=f'{identifier}-l{run["level"]}-{filename}'
                        require(embedded.files.get(label)==suite.fileRecord(path/filename),f'{path}: embedded payload mismatch: {filename}')
                    runDetails.append({'level':run['level'],'status':'passed','artifact_files':meta['artifact_integrity']['file_count'],
                                        'movie_frames':movie['decoded_frames'],'L1':run['L1']})
                    result['native_runs']+=1
            details['checks'].extend(['all native summary/run metadata agree','all raw schemas, time/cell coverage and completion records valid',
                                      'all manifests and persisted file hashes agree','generated C++ and executable hashes match recorded builds',
                                      'embedded report payloads equal retained files'])
            aggregate=json.loads((root/build/'summary.json').read_text())
            family=next(entry for entry in aggregate['families'] if entry['family']=='radiation')
            expected='failed' if any(entry['status']=='failed' for entry in summary) else 'conditional' if any(entry['status']=='conditional' for entry in summary) else 'passed'
            require(family['status']==expected,'Unified radiation-family status mismatch')
            require(family['returncode']=={'failed':1,'conditional':3,'passed':0}[expected],'Unified radiation-family return code mismatch')
            details['checks'].append('unified family status/return code agree with case statuses')
        except Exception as error:
            result['status']='failed';result['errors'].append({'build':build,'kind':'artifact_or_metadata','error':str(error)})
    result['expected_native_runs']=len(descriptors)*3*3
    if result['native_runs']!=result['expected_native_runs']:result['status']='failed'
    result['artifact_status']='passed' if result['native_runs']==result['expected_native_runs'] and not any(error['kind']=='artifact_or_metadata' for error in result['errors']) else 'failed'
    result['source_status']='failed' if any(error['kind']=='source_mismatch' for error in result['errors']) else 'passed'
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix',type=Path)
    parser.add_argument('--decode',action='store_true',help='Independently fully decode every retained movie again')
    args=parser.parse_args(argv);result=audit(args.matrix,args.decode)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
