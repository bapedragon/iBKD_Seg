"""Authorized Tiny endpoint packs, including mixed 2k/10k, under one job deadline."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

from .tiny_val_timing import CONFIG_DIR, save_json
from .tiny_screen2000_report import final_line, select_plans

CONFIG=CONFIG_DIR/'beta_grid2000_ibkd_l025_v2.json'
LEGACY_CONFIG=CONFIG_DIR/'beta_grid2000_ibkd_l025_v1.json'
LG_ALG_CONFIG=CONFIG_DIR/'beta_grid2000_lg_alg_v1.json'
IBKD050_CONFIG=CONFIG_DIR/'beta_grid2000_ibkd_l050_4betas_v1.json'
FIXED_CONFIG=CONFIG_DIR/'baseline_grid2000_fskd_c2vkd_v1.json'
FOLLOWUP_CONFIG=CONFIG_DIR/'followup10k_alg_ibkd025_top1_vanilla2k_v1.json'


def load_config(path):
    config=json.loads(path.read_text())
    locked=next((p for p in (CONFIG,LEGACY_CONFIG,LG_ALG_CONFIG,IBKD050_CONFIG,FIXED_CONFIG,FOLLOWUP_CONFIG)
                 if p.name==path.name),None)
    if locked is None or config != json.loads(locked.read_text()):
        raise ValueError('Use a committed Tiny endpoint pack config')
    if locked==FOLLOWUP_CONFIG:
        from .tiny_followup import validate_config
        return validate_config(config)
    if locked==FIXED_CONFIG:
        original=load_config(CONFIG)
        changed={'protocol_id','pack','run_kind','runs','ibkd_lambdas','beta_multipliers',
                 'beta_initial_ce_ratio','beta_status','selection_metric','secondary_metric',
                 'start_policy','continuation_policy'}
        if (set(config)!=set(original)|{'fskd','c2vkd'} or
                any(config[k]!=v for k,v in original.items() if k not in changed)):
            raise ValueError('Fixed baseline common training or evaluation protocol drift')
        smoke=json.loads((CONFIG_DIR/'smoke25_fskd_c2vkd_v3.json').read_text())
        expected=[dict(p,run_index=i+1,candidate=None,beta=None,initial_target_ratio=None)
                  for i,p in enumerate(p for p in smoke['runs'] if p['method'] in ('fskd','c2vkd'))]
        expected[0]['comparison_group']='primary_common_pretraining'
        if (config['runs']!=expected or config['pack']!='fskd_c2vkd' or
                any(config[k]!=smoke[k] for k in ('fskd','c2vkd')) or
                config['beta_multipliers'] or config['beta_initial_ce_ratio'] is not None or
                config['selection_metric']!='miou'):
            raise ValueError('FSKD*/C2VKD* must preserve the smoke recipes, without beta tuning')
        return config
    if locked==IBKD050_CONFIG:
        original=load_config(CONFIG)
        changed={'protocol_id','pack','runs','ibkd_lambdas','beta_multipliers','continuation_policy'}
        if set(config)!=set(original) or any(config[k]!=v for k,v in original.items() if k not in changed):
            raise ValueError('iBKD lambda0.5 common training or evaluation protocol drift')
        old=json.loads((CONFIG_DIR/'beta_grid500_shared_lg_8betas_v6.json').read_text())
        expected=[p for p in old['runs'] if p.get('lambda')==.5 and p['candidate'] in (1,3,6,8)]
        if (config['runs']!=expected or config['ibkd_lambdas']!=[.5]
                or config['beta_multipliers']!=[.5,1.5,4,8] or config['pack']!='ibkd_l050_4betas'):
            raise ValueError('iBKD lambda0.5 pack must contain the fixed original candidates 1,3,6,8')
        return config
    if locked==LG_ALG_CONFIG:
        original=load_config(CONFIG)
        changed={'protocol_id','pack','runs','ibkd_lambdas','start_policy','continuation_policy'}
        if any(config.get(k)!=v for k,v in original.items() if k not in changed):
            raise ValueError('LG/ALG common training or evaluation protocol drift')
        old=json.loads((CONFIG_DIR/'beta_grid500_shared_lg_8betas_v6.json').read_text())
        expected=[]
        for plan in (p for p in old['runs'] if p['method']=='lg'):
            for method in ('lg','alg'):
                expected.append(dict(plan,id=f'{method}_b{plan["candidate"]}',method=method,
                                     run_index=len(expected)+1))
        if config['runs']!=expected or config['ibkd_lambdas']!=[] or config['pack']!='lg_alg':
            raise ValueError('LG/ALG pack must contain the eight fixed beta pairs in interleaved order')
        return config
    old=json.loads((CONFIG_DIR/'beta_grid500_shared_lg_8betas_v6.json').read_text())
    changed={'protocol_id','run_kind','steps','runs','lg_alg_shared_screen','diagnostic_validation_samples'}
    for key,value in old.items():
        if key not in changed and config.get(key) != value:
            raise ValueError(f'Common Tiny protocol drift: {key}')
    expected=[p for p in old['runs'] if p['method']=='ibkd' and p['lambda']==.25]
    if config['runs'] != expected or config['steps'] != 2000 or not config['full_validation_at_endpoint']:
        raise ValueError('Unexpected candidates or endpoint')
    budget,reserve=(35100,180) if locked==LEGACY_CONFIG else (36000,120)
    if (config['validation_samples'] != 500 or config['job_budget_seconds'] != budget or
            config['save_reserve_seconds'] != reserve or config['evaluation_cpu_threads'] != 4):
        raise ValueError('Unexpected validation or runtime budget')
    return config


def stopping_deadlines(started, config, deadline=None):
    """One job-wide limit, including setup; reserve the final two minutes in v2."""
    if not math.isfinite(started) or (deadline is not None and not math.isfinite(deadline)):
        raise ValueError('Job start/deadline must be finite Unix timestamps')
    hard=started+config['job_budget_seconds']
    if deadline is not None:
        hard=min(hard,deadline)
    return hard,hard-config['save_reserve_seconds']


def identity_checks(rows, *, fixed_recipes=False):
    """Only compare a family present in this pack, including partial input prefixes."""
    observed=[r for r in rows if r.get('student_initial_state_sha256')]
    checks={'initial_identity_present_for_passed':all(
        all(r.get(k) for k in ('student_initial_state_sha256','teacher_state_sha256','guidance_initial_state_sha256'))
        for r in rows if r.get('status')=='passed')}
    if not observed:
        return checks, [k for k,v in checks.items() if not v]
    keys=['student_initial_state_sha256','teacher_state_sha256']
    if not fixed_recipes:
        keys.append('guidance_initial_state_sha256')
    for key in keys:
        checks[key]=all(r.get(key) for r in observed) and len({r.get(key) for r in observed})==1
    if fixed_recipes:
        # Different losses have different adapters; compare only within the same method.
        checks['same_adapter_within_method']=all(
            len({r.get('guidance_initial_state_sha256') for r in observed if r.get('method')==method})==1
            for method in {r.get('method') for r in observed})
        checks['fixed_recipe_no_controller']=all(r.get('controller') is None for r in observed)
        checks['fixed_assets_verified']=all(
            (r.get('soft_rank_execution') or {}).get('forward_backward')=='passed' if r.get('method')=='fskd'
            else (r.get('clip_pool') or {}).get('frozen_no_grad_unchanged') is True
            for r in rows if r.get('status')=='passed')
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
                   initial_beta=plan['beta'],**{'lambda':plan.get('lambda')})
    for key,value in dict(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],
                          initial_beta=plan['beta'],**{'lambda':plan.get('lambda')}).items():
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


def compare_active_lg_alg(rows, config):
    """Compare observed scalar prefixes only while both controllers still use guidance."""
    by_id={r['run_id']:r for r in rows}
    pairs=[]
    for candidate in range(1,9):
        lg,alg=by_id.get(f'lg_b{candidate}'),by_id.get(f'alg_b{candidate}')
        if lg is None or alg is None:
            continue
        compared=0;first=None
        for a,b in zip(lg.get('losses',[]),alg.get('losses',[])):
            if a['beta']<=0 or b['beta']<=0:
                break
            compared+=1
            for key in ('step','beta','loss','ce','guidance','grad_norm_unclipped'):
                if not math.isclose(a[key],b[key],rel_tol=config['resume_rtol'],abs_tol=config['resume_atol']):
                    first=dict(step=b['step'],field=key,lg=a[key],alg=b[key])
                    break
            if first:
                break
        pairs.append(dict(candidate=candidate,compared_active_steps=compared,
                          first_mismatch=first,matches=None if compared==0 else first is None,
                          alg_stop_step=alg.get('guidance_stop_step')))
    return pairs


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
        if getattr(args,'start_run',None) is not None:
            command += ['--start-run',str(args.start_run)]
        if args.resume and plan == plans[0]:
            command += ['--resume',str(args.resume.resolve())]
        print(f'[TI16_GRID2000_START] run={plan["id"]} method={plan["method"]} beta={plan["beta"]} lambda={plan.get("lambda")} remaining_seconds={args.deadline-time.time():.1f}',flush=True)
        completed=launch_child(command,should_stop)
        row=collect_child(destination,completed,plan)
        rows.append(row)
        report['runs']=rows
        save_json(output/'grid_summary.json',report)
        if row['status']=='paused':
            break
    if config.get('pack')=='alg_ibkd025_top1_10k_vanilla2k':
        from .tiny_followup import identity_checks as mixed_identity_checks
        checks,issues=mixed_identity_checks(rows,plans)
    else:
        checks,issues=identity_checks(rows,fixed_recipes=config.get('pack')=='fskd_c2vkd')
    if config.get('pack')=='fskd_c2vkd':
        checks['fixed_recipe_provenance']=all(
            r.get('method_provenance')==config[r['method']] and
            r.get('fixed_loss_coefficients')==config[r['method']]['coefficients']
            for r in rows if r.get('status')=='passed')
    checks['run_inventory']=[r.get('run_id') for r in rows]==[p['id'] for p in plans[:len(rows)]]
    checks['planned_method_beta_lambda']=all(
        (r.get('method'),r.get('initial_beta'),r.get('lambda'))==(p['method'],p['beta'],p.get('lambda'))
        for r,p in zip(rows,plans))
    issues=[k for k,v in checks.items() if not v]
    if config.get('pack')=='lg_alg':
        report['lg_alg_active_prefix_pairs']=compare_active_lg_alg(rows,config)
        issues += [f'lg_alg_b{p["candidate"]}_active_prefix' for p in report['lg_alg_active_prefix_pairs']
                   if p['matches'] is False]
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
    parser.add_argument('--start-run',type=int,choices=range(1,17),
                        help='Execution order: LG/ALG1..16, fixed baselines1..2, mixed2k/10k1..3')
    parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if not (args.preflight_only or args.prepare_data_only) and output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory; resume goes into a fresh destination')
    output.mkdir(parents=True,exist_ok=True)
    began=time.monotonic()
    started=float(os.environ.get('CITYSCAPES_TI16_JOB_STARTED',time.time()))
    report=dict(status='running',protocol_id='cityscapes_ti16_crop512_ibkd_l025_grid2000_v2',runs=[],
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
        report.update(protocol_id=config['protocol_id'],job_budget_seconds=config['job_budget_seconds'],
                      save_reserve_seconds=config['save_reserve_seconds'],pack=config['pack'],
                      start_run=args.start_run)
        plans=select_plans(config,start_candidate=args.start_candidate,start_run=args.start_run,run_id=args.run_id)
        if not plans:
            raise ValueError('Unknown candidate')
        if args.run_id:
            plan=plans[0]
            report.update(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],
                          initial_beta=plan['beta'],initial_target_ratio=plan['initial_target_ratio'],
                          **{'lambda':plan.get('lambda')},completed_steps=0,selected_step=None,
                          selected_epoch=None,diagnostic_metrics=None,full_validation=False)
            if config['pack']=='fskd_c2vkd':
                report.update(comparison_group=plan['comparison_group'],display_name=plan['display_name'])
        if args.resume and not args.resume.is_file():
            raise FileNotFoundError('Restore the same config/endpoint run folder and point --resume to its resume.json')
        args.deadline,stop_at=stopping_deadlines(started,config,args.deadline)
        report.update(hard_deadline_unix=args.deadline,stop_at_unix=stop_at,
                      training_stop_after_seconds=stop_at-started)
        args.should_stop=lambda: stopped['requested'] or time.time()>=stop_at
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
                    if config['pack']=='alg_ibkd025_top1_10k_vanilla2k':
                        from .tiny_followup import effective_config
                        run_config=effective_config(config,plan)
                        save_json(output/'effective_config.json',run_config)
                    else:
                        run_config=config
                    run(args,run_config,plan,output,report)
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
