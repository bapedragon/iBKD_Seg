"""ALG pilot with L/16 first-batch loss ratios, separate from the old Tiny grid."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .tiny_grid_report import CLASS_NAMES,MAX_FINAL_BYTES,bounded_text,number,terminal_row

PACK='alg_l16_ratio_grid2000'
PROTOCOL='cityscapes_ti16_alg_l16_ratio_grid2000_v1'
MARKER='[CITYSCAPES_TI16_L16_RATIO2000_FINAL] '
ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'phase4/Cityscapes_Segmenter-Ti16/experiments/l16_ratio_match_v1/alg_reference.json'


def validate_config(config):
    from .tiny_screen2000 import FOLLOWUP_CONFIG,load_config
    original=load_config(FOLLOWUP_CONFIG)
    original.pop('selection_reference')
    ref=json.loads(REFERENCE.read_text())
    calibration=ref['tiny_calibration']
    if hashlib.sha256((ROOT/calibration['path']).read_bytes()).hexdigest()!=calibration['sha256']:
        raise ValueError('Historical Tiny first-step source changed')
    plans=[]
    for index,item in enumerate(ref['l16_candidates'],1):
        ratio=item['l16_beta']*item['guidance']/item['ce']
        if item['candidate']!=index or ratio!=item['target_ratio']:
            raise ValueError('L/16 first-step ratio reference is inconsistent')
        plans.append(dict(id=f'alg_l16r{index}',method='alg',candidate=index,
            beta=ratio*calibration['ce']/calibration['guidance'],initial_target_ratio=ratio,
            run_index=index,target_steps=2000))
    if len(plans)!=4 or ref['runs']!=plans or ref['l16_miou_rank_order']!=[2,1,4,3]:
        raise ValueError('Expected all four original L/16 ALG candidates')
    expected=dict(original,protocol_id=PROTOCOL,pack=PACK,
        run_kind='l16_first_step_ratio_matched_2000_endpoint_alg_screen',steps=2000,
        beta_initial_ce_ratio=None,beta_status='fixed_l16_first_step_targets_new_tiny_betas_no_online_recalibration',
        runs=plans,ibkd_lambdas=[],selection_rule='fixed_2000_endpoint_not_best_checkpoint',
        validation_log_tag='TI16_L16_RATIO_VAL',first_step_ratio_check=dict(rtol=1e-4,atol=1e-8),
        ratio_reference=dict(path=str(REFERENCE.relative_to(ROOT)),sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()))
    if config!=expected:
        raise ValueError('L/16-matched ALG grid changed the locked beta, source or shared Tiny protocol')
    return config


def check_first_step(row,plan,tolerance):
    ce,guidance=row.get('ce'),row.get('guidance')
    actual=None if ce is None or guidance is None or ce<=0 else plan['beta']*guidance/ce
    if actual is None or not math.isfinite(actual) or not math.isclose(
            actual,plan['initial_target_ratio'],rel_tol=tolerance['rtol'],abs_tol=tolerance['atol']):
        raise ValueError(f'First-step CE ratio mismatch: actual={actual}, target={plan["initial_target_ratio"]}; beta was not changed')
    return actual


def ratio_matches(row,plan):
    first=next((v for v in row.get('losses',[]) if v.get('step')==1),None)
    if first is None:
        return False
    try:
        check_first_step(first,plan,dict(rtol=1e-4,atol=1e-8))
    except ValueError:
        return False
    return True


def final_line(report,plans):
    ref=json.loads(REFERENCE.read_text())
    references={r['candidate']:r for r in ref['l16_candidates']}
    by_id={r['run_id']:r for r in report.get('runs',[])}
    rows=[];scores={}
    for plan in plans:
        raw=by_id.get(plan['id'],{});row=terminal_row(raw,plan)
        first=next((v for v in raw.get('losses',[]) if v.get('step')==1),{})
        observation=raw.get('first_step_ratio_observation') or {}
        if not first and observation:
            first=observation
        ce,guidance=first.get('ce'),first.get('guidance')
        actual=None if ce is None or guidance is None or ce<=0 else plan['beta']*guidance/ce
        controller=raw.get('controller') or {};reference=references[plan['candidate']]
        row.update(run_index=plan['run_index'],target_steps=2000,
            initial_ratio_basis='l16_first_training_batch_target',initial_ratio_target_percent=100*plan['initial_target_ratio'],
            initial_ratio_first_step_percent=number(None if actual is None else 100*actual),
            first_step_ce=number(ce),first_step_guidance=number(guidance),
            first_step_ratio_matches_target=None if actual is None else math.isclose(actual,plan['initial_target_ratio'],rel_tol=1e-4,abs_tol=1e-8),
            l16_beta=reference['l16_beta'],l16_miou_pct=number(reference['miou_pct']),
            l16_stop_epoch=reference['stop_epoch'],full_validation=raw.get('full_validation',False),
            validation_seconds=number(raw.get('validation_seconds')),
            student_unchanged_during_validation=raw.get('student_unchanged_during_validation'),
            controller_warmup_epochs=controller.get('warmup_epochs'),
            controller_epoch_guidance_means=[number(v) for v in controller.get('loss_history',[])[:6]],
            controller_smoothed_derivatives=[number(v) for v in controller.get('smoothed_derivative_history',[])[:6]],
            pause_reason=bounded_text(raw.get('pause_reason'),256),
            checkpoint_pointer=bounded_text((raw.get('checkpoint') or {}).get('pointer'),1024))
        rows.append(row)
        metrics=raw.get('diagnostic_metrics_percent')
        if metrics and metrics.get('miou') is not None:
            score=metrics['miou']
        else:
            score=(raw.get('diagnostic_metrics') or {}).get('miou')
            if score is not None:
                score*=100
        if row['status']=='passed' and row['full_validation'] and row['first_step_ratio_matches_target'] is True and number(score) is not None:
            scores[plan['candidate']]=score
    complete=(report.get('status')=='passed' and set(scores)=={1,2,3,4})
    ties=len(set(scores.values()))!=len(scores)
    ranked=sorted(scores,key=lambda c:-scores[c]) if complete and not ties else None
    baseline=ref['l16_miou_rank_order']
    result=dict(status=report.get('status'),protocol_id=PROTOCOL,pack=PACK,configured_pack_runs=4,
        expected_runs=len(plans),runs=rows,target_steps=2000,seed=1,schedule_total_steps=80000,
        primary_metric='miou',secondary_metric='pixel_accuracy_at_same_checkpoint',
        selection_rule='fixed_2000_endpoint_not_best_checkpoint',planned_validation_samples=500,
        completed_runs=[r['run_id'] for r in rows if r['status']=='passed'],
        paused_runs=[r['run_id'] for r in rows if r['status']=='paused'],
        not_run_runs=[r['run_id'] for r in rows if r['status']=='not_run'],
        failed_runs=[r['run_id'] for r in rows if r['status'] not in ('passed','paused','not_run')],
        ranking=dict(complete=complete,exact_score_tie=ties,reference_l16=baseline,tiny=ranked,
            top1_matches=None if ranked is None else ranked[0]==baseline[0],
            top2_set_matches=None if ranked is None else set(ranked[:2])==set(baseline[:2]),
            full_order_matches=None if ranked is None else ranked==baseline,
            spearman=None if ranked is None else number(1-6*sum((ranked.index(c)-baseline.index(c))**2 for c in baseline)/(4*(4**2-1))),
            suggested_top2_candidates=None if ranked is None else ranked[:2]),
        alg_warmup_epochs=0,alg_earliest_possible_off_step=745,first_step_ratio_tolerance=ref['runtime_ratio_tolerance'],
        class_iou_order=CLASS_NAMES,trajectory_order=['first25_median','last25_median','maximum','minimum'],
        identity_checks=report.get('identity_checks',{}),review_items=report.get('review_items',[]),
        job_budget_seconds=36000,save_reserve_seconds=120,training_stop_after_seconds=report.get('training_stop_after_seconds',35880),
        total_job_seconds=number(report.get('total_job_seconds')),invocation_seconds=number(report.get('invocation_seconds')),
        error=bounded_text(report.get('error'),1024),pipeline_exit_code=report.get('pipeline_exit_code'),
        summary_path=bounded_text(report.get('summary_path'),1024),scientific_result=False,test_used=False,
        automatic_next_stage=False,automatic_hyperparameter_changes=False,
        score_scope='new ratio-matched seed1 pilot; compare 2000-step full-val rank within each model; not final80k')
    line=MARKER+json.dumps(result,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if len(line)+1>MAX_FINAL_BYTES:
        raise ValueError('Ratio-matched final report exceeded 50,000 bytes')
    return line
