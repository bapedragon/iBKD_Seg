"""One authorized Tiny iBKD lambda0.25 pack: eight fixed 2000-step full-val endpoints."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

from .tiny_val_timing import CONFIG_DIR, save_json
from .tiny_screen2000_report import final_line

CONFIG=CONFIG_DIR/'beta_grid2000_ibkd_l025_v1.json'


def load_config(path):
    config=json.loads(path.read_text())
    if path.name != CONFIG.name or config != json.loads(CONFIG.read_text()):
        raise ValueError('Use the committed lambda0.25 2000-step pack config')
    old=json.loads((CONFIG_DIR/'beta_grid500_shared_lg_8betas_v6.json').read_text())
    changed={'protocol_id','run_kind','steps','runs','lg_alg_shared_screen','diagnostic_validation_samples'}
    for key,value in old.items():
        if key not in changed and config.get(key) != value:
            raise ValueError(f'Common Tiny protocol drift: {key}')
    expected=[p for p in old['runs'] if p['method']=='ibkd' and p['lambda']==.25]
    if config['runs'] != expected or config['steps'] != 2000 or not config['full_validation_at_endpoint']:
        raise ValueError('Unexpected candidates or endpoint')
    if (config['validation_samples'] != 500 or config['job_budget_seconds'] != 35100 or
            config['save_reserve_seconds'] != 180 or config['evaluation_cpu_threads'] != 4):
        raise ValueError('Unexpected validation or runtime budget')
    return config


def identity_checks(rows):
    """Only compare a family present in this pack, including partial input prefixes."""
    observed=[r for r in rows if r.get('student_initial_state_sha256')]
    checks={'initial_identity_present_for_passed':all(
        all(r.get(k) for k in ('student_initial_state_sha256','teacher_state_sha256','guidance_initial_state_sha256'))
        for r in rows if r.get('status')=='passed')}
    if not observed:
        return checks, [k for k,v in checks.items() if not v]
    for key in ('student_initial_state_sha256','teacher_state_sha256','guidance_initial_state_sha256'):
        checks[key]=all(r.get(key) for r in observed) and len({r.get(key) for r in observed})==1
    reference=max((r.get('input_hashes',[]) for r in observed),key=len)
    checks['same_observed_input_prefixes']=all(
        r.get('input_hashes',[])==reference[:len(r.get('input_hashes',[]))] for r in observed)
    checks['complete_passed_endpoints']=all(
        r.get('completed_steps')==2000 and r.get('selected_step')==2000 and r.get('full_validation') is True
        and r.get('validation_samples')==500 and (r.get('diagnostic_metrics') or {}).get('evaluated_classes')==19
        and len(r.get('input_hashes',[]))==2000 and r.get('teacher_frozen_verified') is True
        and (r.get('checkpoint') or {}).get('strict_state_roundtrip')=='passed'
        for r in rows if r.get('status')=='passed')
    return checks,[k for k,v in checks.items() if not v]


def collect_child(destination, completed, plan):
    summary=destination/'summary.json'
    if summary.is_file():
        row=json.loads(summary.read_text())
    else:
        progress=destination/'progress.json'
        row=json.loads(progress.read_text()) if progress.is_file() else {}
        row.update(status='runtime_failure',error=f'child_exit={completed.returncode}; final summary missing',
                   run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],
                   initial_beta=plan['beta'],**{'lambda':plan['lambda']})
    for key,value in dict(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],
                          initial_beta=plan['beta'],**{'lambda':plan['lambda']}).items():
        row.setdefault(key,value)
    if completed.returncode != 0 and row.get('status') in ('passed','paused'):
        row.update(status='runtime_failure',error=f'child_exit={completed.returncode}')
    return row


def launch_child(command, should_stop):
    with subprocess.Popen(command) as child:
        sent_stop=False
        while True:
            if should_stop() and not sent_stop:
                child.send_signal(signal.SIGTERM)
                sent_stop=True
            try:
                return subprocess.CompletedProcess(command,child.wait(timeout=1))
            except subprocess.TimeoutExpired:
                pass


def execute_pack(args, config, plans, output, report, should_stop):
    """Sequential subprocess isolation; pause stops scheduling without inventing missing scores."""
    rows=[]
    for plan in plans:
        if should_stop():
            break
        destination=output/plan['id']
        command=[sys.executable,'-u','-m','ibkd_seg.cityscapes.tiny_screen2000']
        for flag in ('cache-root','data-dir','manifest','config'):
            command += ['--'+flag,str(getattr(args,flag.replace('-','_')).resolve())]
        command += ['--output-dir',str(destination),'--run-id',plan['id'],
                    '--deadline',str(args.deadline),'--start-candidate',str(args.start_candidate)]
        if args.resume and plan == plans[0]:
            command += ['--resume',str(args.resume.resolve())]
        print(f'[TI16_GRID2000_START] run={plan["id"]} beta={plan["beta"]} lambda=0.25 remaining_seconds={args.deadline-time.time():.1f}',flush=True)
        completed=launch_child(command,should_stop)
        row=collect_child(destination,completed,plan)
        rows.append(row)
        report['runs']=rows
        save_json(output/'grid_summary.json',report)
        if row['status']=='paused':
            break
    checks,issues=identity_checks(rows)
    checks['run_inventory']=[r.get('run_id') for r in rows]==[p['id'] for p in plans[:len(rows)]]
    checks['planned_method_beta_lambda']=all(
        (r.get('method'),r.get('initial_beta'),r.get('lambda'))==(p['method'],p['beta'],p['lambda'])
        for r,p in zip(rows,plans))
    issues=[k for k,v in checks.items() if not v]
    failed=any(r.get('status') not in ('passed','paused') for r in rows)
    complete=len(rows)==len(plans) and all(r['status']=='passed' for r in rows)
    report.update(identity_checks=checks,review_items=issues,
                  status='needs_review' if failed or issues else 'passed' if complete else 'paused',
                  beta_ranking_performed=False,automatic_next_stage=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    for flag in ('cache-root','data-dir','output-dir'):
        parser.add_argument('--'+flag,required=True,type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--zip-dir',type=Path,default=Path('/app/data/chaoyang'))
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--prepare-data-only',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--start-candidate',type=int,default=1,choices=range(1,9))
    parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not (args.preflight_only or args.prepare_data_only) and output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory; resume goes into a fresh destination')
    output.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    started=float(os.environ.get('CITYSCAPES_TI16_JOB_STARTED',time.time()))
    report=dict(status='running',protocol_id='cityscapes_ti16_crop512_ibkd_l025_grid2000_v1',runs=[],
                start_candidate=args.start_candidate,test_used=False,automatic_next_stage=False,
                beta_ranking_performed=False)
    stopped={'requested':False}
    def request_stop(signum, frame):
        stopped['requested']=True
        print(f'[TI16_GRID2000_STOP_REQUEST] signal={signum}',flush=True)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,request_stop)
    failed=False; plans=[]
    try:
        config=load_config(args.config)
        plans=[p for p in config['runs'] if (p['id']==args.run_id if args.run_id else p['candidate']>=args.start_candidate)]
        if not plans:
            raise ValueError('Unknown candidate')
        if args.run_id:
            plan=plans[0]
            report.update(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],
                          initial_beta=plan['beta'],initial_target_ratio=plan['initial_target_ratio'],
                          **{'lambda':plan['lambda']},completed_steps=0,selected_step=None,
                          selected_epoch=None,diagnostic_metrics=None,full_validation=False)
        if args.resume and not args.resume.is_file():
            raise FileNotFoundError('Restore the same 2000-step run folder and point --resume to its resume.json')
        args.deadline=min(args.deadline or started+config['job_budget_seconds'], started+config['job_budget_seconds'])
        args.should_stop=lambda: stopped['requested'] or time.time()>=args.deadline-config['save_reserve_seconds']
        if args.preflight_only or args.prepare_data_only:
            from .tiny_val_initial import preflight_initial, prepare_initial_data
            setup_config=dict(dataset_config={k:config[k] for k in ('run_kind','val_samples','image_size','crop_size','seed')})
            source=preflight_initial(args.data_dir.resolve(),args.zip_dir.resolve(),setup_config)
            if args.prepare_data_only:
                source=prepare_initial_data(source,setup_config,output)
            save_json(output/'data_setup.json',{k:v for k,v in source.items() if k!='manifest'})
            print(f'[TI16_GRID2000_PREFLIGHT] data_ready={not source["needs_prepare"]} training_updates=0',flush=True)
            return
        if args.manifest is None:
            raise ValueError('--manifest is required for execution')
        from .official_api import bootstrap
        from .official_assets import verify
        from .tiny_grid import run
        from .tiny_repeat import warning_summary
        report['phase']='runtime_setup'
        bootstrap(args.cache_root)
        provenance=verify(args.cache_root,student='tiny')
        report['assets']=provenance['weights']
        save_json(output/'config.json',config); save_json(output/'provenance.json',provenance)
        if args.run_id:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter('always')
                try:
                    run(args,config,plan,output,report)
                finally:
                    report['warning_summary']=warning_summary(records)
                    report['deterministic_warning_count']=sum('deterministic' in str(w.message) for w in records)
                    save_json(output/'warnings.json',report['warning_summary'])
        else:
            from .full_data import prepare_labels
            report['phase']='data_labels_and_audit'
            prepare_labels(args.data_dir,args.manifest,{'train':2975,'val':500},output)
            execute_pack(args,config,plans,output,report,args.should_stop)
        failed=report['status'] not in ('passed','paused')
    except Exception as error:
        failed=True
        report.update(status='numerical_failure' if isinstance(error,FloatingPointError) else 'runtime_failure',
                      error=repr(error),failure_stage=report.get('phase','setup'))
        progress=output/'progress.json'
        if args.run_id and progress.is_file():
            partial=json.loads(progress.read_text())
            for key in ('losses','input_hashes'):
                report.setdefault(key,partial.get(key,[]))
        (output/'traceback.txt').write_text(traceback.format_exc()); traceback.print_exc()
    finally:
        if failed or not (args.preflight_only or args.prepare_data_only):
            path=output/('summary.json' if args.run_id else 'grid_summary.json')
            report.update(summary_path=str(path),invocation_seconds=time.monotonic()-began,
                          total_job_seconds=time.time()-started)
            save_json(path,report)
            terminal=dict(report,runs=[report]) if args.run_id else report
            line=final_line(terminal,plans)
            (output/'terminal_summary.log').write_text(line+'\n',encoding='ascii')
            print(line,flush=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
