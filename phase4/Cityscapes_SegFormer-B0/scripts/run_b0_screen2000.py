#!/usr/bin/env python3
"""세 H200 묶음의 설치·검증·2k 학습·선별·중단 상태를 관리한다."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]


def terminal_summary(report):
    result={k:v for k,v in report.items() if k not in ('protocol','code','assets','runs')}
    result['protocol_id']=report.get('protocol',{}).get('id')
    result['runs']=[{k:v for k,v in row.items() if k not in ('validation_history','signature','first_25_batch_hashes','best')}
                    for row in report['runs']]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('pack',choices=('pack1','pack2','pack3'))
    args=parser.parse_args();os.chdir(REPO)
    output=Path(os.environ.get('B0_SCREEN_OUTPUT_BASE','/app/output/cityscapes_b0_screen2000_v2'))/args.pack
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    resume_value=os.environ.get('B0_SCREEN_RESUME_FROM');resume=None if not resume_value else Path(resume_value).resolve()
    if any(p.name!='run.log' for p in output.iterdir()) and resume!=output:raise ValueError('Nonempty output requires B0_SCREEN_RESUME_FROM pointing to that pack directory')
    cache=Path(os.environ.get('B0_SCREEN_CACHE','/app/scratch/cityscapes_b0_smoke_v1/assets')).resolve()
    data=Path(os.environ.get('B0_SCREEN_DATA','/app/scratch/cityscapes_b0_smoke_v1/cityscapes')).resolve()
    zip_root=Path(os.environ.get('CITYSCAPES_ZIP_DIR','/app/data/chaoyang')).resolve()
    started=time.time();deadline=started+9*3600-180
    os.environ.update(PYTHONPATH=str(REPO/'src'),PYTHONUNBUFFERED='1',PYTHONHASHSEED='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='9.0')
    report=dict(status='running',pack=args.pack,runs=[],selection={},selected_epoch=None,metrics=None,
                test_used=False,target_steps_per_run=2000,output=str(output),resume_from=None if resume is None else str(resume))
    def save():
        (output/'group_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    def stage(name):
        report['stage']=name;save();print('[B0_SCREEN] '+json.dumps(dict(pack=args.pack,stage=name)),flush=True)
    def run(command,check=True):
        remaining=deadline+90-time.time()
        if remaining<=0:raise TimeoutError('Job time budget exhausted')
        return subprocess.run(command,check=check,timeout=remaining,env=os.environ.copy()).returncode
    try:
        if resume is not None:
            previous=json.loads((resume/'group_summary.json').read_text())
            if previous['pack']!=args.pack:raise ValueError('Resume pack differs')
        stage('install')
        run([sys.executable,'-m','pip','install','--disable-pip-version-check','-e','.',
             'opencv-python-headless==4.13.0.92','gdown==5.2.0','pytest==8.4.2'])
        if args.pack=='pack1':
            run([sys.executable,'-m','pip','install','--disable-pip-version-check','ninja==1.13.0'])
            run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-build-isolation','torchsort==0.1.10'])
        with (output/'pip_freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
        sys.path.insert(0,str(REPO/'src'))
        import numpy as np
        from ibkd_seg.cityscapes.b0.screen import specification,candidates,code_identity
        from ibkd_seg.cityscapes.b0.assets import prepare as prepare_assets
        from ibkd_seg.cityscapes.b0.calibration_data import verify_preparation
        from ibkd_seg.cityscapes.b0.training_data import make_plan,plan_hash,PlannedDataset,batches,calibration_digest_update
        from ibkd_seg.cityscapes.b0.training import rank_candidates
        from ibkd_seg.cityscapes.b0.reproducibility import configure_runtime
        from ibkd_seg.cityscapes.data import save_json,sha256
        config,grid=specification();configure_runtime();inventory=candidates(config,grid,args.pack)
        report.update(expected_runs=inventory,protocol=config,code=code_identity(),
                      git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        if resume is not None:
            if previous.get('protocol')!=config or previous.get('code')!=report['code']:
                raise ValueError('Resume protocol/code differs from the previous pack')
            if resume!=output and (resume/'runs').exists():
                stage('copy_portable_resume_bundles')
                shutil.copytree(resume/'runs',output/'runs')
        stage('unit_checks')
        run([sys.executable,'-m','pytest','-q','tests/test_cityscapes_b0_training.py','tests/test_cityscapes_b0_calibration.py'])
        if args.pack=='pack1':run([sys.executable,'-m','pytest','-q','tests/test_cityscapes_b0.py'])
        stage('data')
        metadata=[(data/name).exists() for name in ('manifest.json','preparation.json')]
        if any(metadata) and not all(metadata):raise ValueError('Incomplete data cache provenance')
        if not all(metadata):
            run([sys.executable,str(REPO/'phase4/phase4_cityscapes/scripts/check_cityscapes_upload.py'),
                 '--search-root',str(zip_root),'--output',str(output/'zip_audit.json')])
            links=output/'zip_links';links.mkdir(exist_ok=True)
            for record in json.loads((output/'zip_audit.json').read_text())['files']:
                source=Path(record['path']).resolve();target=links/source.name
                if target.is_symlink() or target.exists():
                    if target.resolve()!=source:raise ValueError('ZIP link differs')
                else:target.symlink_to(source)
            run([sys.executable,'-m','ibkd_seg.cityscapes.prepare','--zip-dir',str(links),'--data-dir',str(data)])
        report['data_verification']=verify_preparation(data,config['expected_archives'])
        if report['data_verification']['manifest_sha256']!=grid['manifest_sha256']:raise ValueError('Data differs from beta calibration')
        stage('assets')
        report['assets']=prepare_assets(cache,weight_names=('cirkd_teacher.pth','nvidia_mit_b0.bin'),manifest_name='screen_asset_manifest.json')
        stage('input_preflight')
        plan=make_plan(1280000);plan_path=output/'augmentation_plan.npy'
        with plan_path.with_suffix('.tmp').open('wb') as f:np.save(f,plan,allow_pickle=False)
        plan_path.with_suffix('.tmp').replace(plan_path)
        dataset=PlannedDataset(data,cache/'cirkd/dataset/list/cityscapes/train.lst',plan)
        digest=hashlib.sha256();checked_digest=hashlib.sha256();ignored=[];valid_counts=[]
        checked_batches=config['input']['preflight_batches']
        for index,(x,y,names) in enumerate(batches(dataset,0,checked_batches*16,16,workers=4),1):
            if index<=25:calibration_digest_update(digest,x,y,names)
            calibration_digest_update(checked_digest,x,y,names)
            valid=y!=-1;valid_counts.append(int(valid.sum()))
            if not valid.any():raise ValueError(f'No valid labels in preflight batch {index}: {names}')
            for sample in (~valid.flatten(1).any(1)).nonzero().flatten().tolist():
                ignored.append(dict(batch=index,sample_in_batch=sample+1,name=names[sample]))
            if index%5==0 or index==checked_batches:print(f'[B0_INPUT_CHECK] {index}/{checked_batches}',flush=True)
        if digest.hexdigest()!=grid['calibration_tensor_sha256']:raise ValueError('Training transforms differ from measured calibration inputs')
        preflight=dict(status='passed',plan_sha256=plan_hash(plan),calibration_tensor_sha256=digest.hexdigest(),
                       manifest_sha256=grid['manifest_sha256'],samples=checked_batches*16,optimizer_updates=0,
                       calibration_samples=400,checked_batches=checked_batches,checked_tensor_sha256=checked_digest.hexdigest(),
                       valid_pixels_per_batch=valid_counts,ignore_only_samples=ignored)
        save_json(output/'input_preflight.json',preflight);report['input_preflight']=preflight
        print('[B0_INPUT_PREFLIGHT] '+json.dumps(preflight),flush=True)
        del plan,dataset,x,y
        for candidate in inventory:
            run_id=candidate['run_id'];target=output/'runs'/run_id
            pointer=None if resume is None else target/'resume.json'
            if pointer is not None and not pointer.exists():pointer=None
            if time.time()>deadline-900:
                prior=json.loads((target/'summary.json').read_text()) if (target/'summary.json').exists() else dict(**candidate,completed_steps=0,metrics=None,selected_epoch=None)
                prior.update(status='pending',reason='Insufficient remaining job time; resume pack',resume_available=pointer is not None)
                report['runs'].append(prior)
                continue
            stage(run_id)
            command=[sys.executable,'-m','ibkd_seg.cityscapes.b0.screen','--cache',str(cache),'--data',str(data),
                     '--output',str(target),'--plan',str(plan_path),'--preflight',str(output/'input_preflight.json'),
                     '--run-id',run_id,'--deadline',str(deadline)]
            if pointer is not None:command.extend(['--resume',str(pointer)])
            code=run(command,check=False)
            record=json.loads((target/'summary.json').read_text()) if (target/'summary.json').exists() else dict(status='failed',**candidate,error='Worker exited without summary')
            if code:record.update(status='failed',exit_code=code)
            report['runs'].append(record);save()
        stage('selection')
        for method in config['groups'][args.pack]:
            rows=[r for r in report['runs'] if r['method']==method]
            if method in ('vanilla','fskd'):report['selection'][method]=dict(status='fixed_baseline',selected_run_ids=[],continue_without_beta_selection=True)
            else:report['selection'][method]=rank_candidates(rows,[r['run_id'] for r in inventory if r['method']==method])
        completed=[r for r in report['runs'] if r['status']=='completed']
        if completed:
            if len({json.dumps(r['first_25_batch_hashes']) for r in completed})!=1:raise ValueError('Candidates used different initial training inputs')
            if len({json.dumps(r['signature']['environment'],sort_keys=True) for r in completed})!=1:raise ValueError('Candidate runtime environments differ')
        report['status']='failed' if any(r['status']=='failed' for r in report['runs']) else 'completed' if len(completed)==len(inventory) else 'paused'
        report['completed_runs']=len(completed)
        save_json(output/'selection.json',report['selection'])
    except Exception as error:
        import traceback
        traceback.print_exc();report.update(status='failed',error=repr(error),failed_stage=report.get('stage'))
    report['elapsed_seconds']=time.time()-started;save()
    print(json.dumps(terminal_summary(report),ensure_ascii=False),flush=True)
    raise SystemExit(1 if report['status']=='failed' else 0)


if __name__=='__main__':main()
