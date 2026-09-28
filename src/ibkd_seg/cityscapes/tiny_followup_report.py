"""Tiny followup endpoints in one bounded H200 log tail."""
from __future__ import annotations

import json
from .tiny_grid_report import CLASS_NAMES,MAX_FINAL_BYTES,bounded_text,number,terminal_row
from .tiny_followup import (PACK,PROTOCOL,VANILLA_PACK,VANILLA_PROTOCOL,
                            IBKD025_B7_PACK,IBKD025_B7_PROTOCOL)
from .tiny_ratio10k import PACK as RATIO10K_PACK,PROTOCOL as RATIO10K_PROTOCOL,ratio_fields

MARKER='[CITYSCAPES_TI16_FOLLOWUP_FINAL] '


def final_line(report,plans):
    ratio_pair=report.get('pack')==RATIO10K_PACK or report.get('protocol_id')==RATIO10K_PROTOCOL
    vanilla_only=report.get('pack')==VANILLA_PACK or report.get('protocol_id')==VANILLA_PROTOCOL
    ibkd_only=report.get('pack')==IBKD025_B7_PACK or report.get('protocol_id')==IBKD025_B7_PROTOCOL
    by_id={r['run_id']:r for r in report.get('runs',[])}
    rows=[]
    for plan in plans:
        raw=by_id.get(plan['id'],{})
        row=terminal_row(raw,plan)
        controller=raw.get('controller') or {}
        row.update(run_index=plan['run_index'],target_steps=plan['target_steps'],
                   selection_rule=f"fixed_{plan['target_steps']}_endpoint_not_best_checkpoint",
                   full_validation=raw.get('full_validation',False),
                   validation_seconds=number(raw.get('validation_seconds')),
                   student_unchanged_during_validation=raw.get('student_unchanged_during_validation'),
                   teacher_loaded=raw.get('teacher_loaded'),guidance_loaded=raw.get('guidance_loaded'),
                   controller_applicable=plan['method']!='vanilla',
                   guidance_active=False if plan['method']=='vanilla' else controller.get('active'),
                   controller_warmup_epochs=controller.get('warmup_epochs'),
                   controller_threshold=number(controller.get('threshold')),
                   controller_window=controller.get('smoothing_window'),
                   controller_epoch_guidance_means=[number(v) for v in controller.get('loss_history',[])[:28]],
                   controller_smoothed_derivatives=[number(v) for v in controller.get('smoothed_derivative_history',[])[:28]],
                   pause_reason=bounded_text(raw.get('pause_reason'),256),
                   checkpoint_pointer=bounded_text((raw.get('checkpoint') or {}).get('pointer'),1024))
        if ibkd_only:
            # Preserve the historical calibration target; report the observed first batch separately.
            first=next((v for v in raw.get('losses',[]) if v.get('step')==1),{})
            ratio=first.get('weighted_guidance_to_seg_loss')
            row.update(initial_ratio_basis='historical_25batch_median_target',
                       initial_ratio_first_step_percent=number(None if ratio is None else 100*ratio),
                       first_step_ce=number(first.get('ce')),first_step_guidance=number(first.get('guidance')),
                       first_step_ratio_basis='100*beta*guidance/ce_before_first_optimizer_update')
        if ratio_pair:
            row.update(ratio_fields(raw,plan))
        rows.append(row)
    result=dict(status=report.get('status'),protocol_id=VANILLA_PROTOCOL if vanilla_only else
                IBKD025_B7_PROTOCOL if ibkd_only else PROTOCOL,
                pack=VANILLA_PACK if vanilla_only else IBKD025_B7_PACK if ibkd_only else PACK,
                expected_runs=len(plans),configured_pack_runs=1 if vanilla_only or ibkd_only else 3,
                invocation_start_run=report.get('start_run') or 1,
                runs=rows,primary_metric='miou',secondary_metric='pixel_accuracy_at_same_checkpoint',
                planned_steps={'vanilla_10k':10000} if vanilla_only else
                              {'ibkd_l025_b7':10000} if ibkd_only else
                              {'vanilla_2k':2000,'alg_b7':10000,'ibkd_l025_b1':10000},
                schedule_total_steps=80000,seed=1,planned_validation_samples=500,
                selection_rule='fixed_10000_endpoint_not_best_checkpoint' if vanilla_only or ibkd_only else
                               'fixed_per_run_endpoint_not_best_checkpoint',
                score_scope='Vanilla fixed10000 full val500; same training budget as earlier ALG/iBKD10k; not final80k'
                            if vanilla_only else
                            'iBKD lambda0.25 candidate7 fixed10000 full val500; same training budget as candidate1; not final80k'
                            if ibkd_only else
                            'mixed endpoints: compare Vanilla2k with earlier2k, not directly with KD10k; not final80k',
                automatic_next_stage=False,automatic_hyperparameter_changes=False,
                beta_ranking_performed=False,scientific_result=False,test_used=False,
                completed_runs=[r['run_id'] for r in rows if r['status']=='passed'],
                paused_runs=[r['run_id'] for r in rows if r['status']=='paused'],
                not_run_runs=[r['run_id'] for r in rows if r['status']=='not_run'],
                failed_runs=[r['run_id'] for r in rows if r['status'] not in ('passed','paused','not_run')],
                alg_warmup_epochs=None if vanilla_only or ibkd_only else 0,ibkd_warmup_epochs=None if vanilla_only else 20,
                alg_earliest_possible_off_step=None if vanilla_only or ibkd_only else 745,
                ibkd_earliest_possible_off_step=None if vanilla_only else 7441,
                class_iou_order=CLASS_NAMES,trajectory_order=['first25_median','last25_median','maximum','minimum'],
                identity_checks=report.get('identity_checks',{}),review_items=report.get('review_items',[]),
                job_budget_seconds=36000,save_reserve_seconds=120,
                training_stop_after_seconds=report.get('training_stop_after_seconds',35880),
                hard_deadline_unix=report.get('hard_deadline_unix'),stop_at_unix=report.get('stop_at_unix'),
                total_job_seconds=number(report.get('total_job_seconds')),
                invocation_seconds=number(report.get('invocation_seconds')),
                error=bounded_text(report.get('error'),1024),pipeline_exit_code=report.get('pipeline_exit_code'),
                summary_path=bounded_text(report.get('summary_path'),1024),
                display_significant_digits=6,beta_full_precision=True)
    if ratio_pair:
        result.update(protocol_id=RATIO10K_PROTOCOL,pack=RATIO10K_PACK,configured_pack_runs=2,
                      planned_steps={'alg_l16r3_10k':10000,'ibkd_l025_l16b05_10k':10000},
                      selection_rule='fixed_10000_endpoint_not_best_checkpoint',
                      first_step_ratio_tolerance=dict(rtol=1e-4,atol=1e-8),
                      score_scope='new first-step-ratio track: ALG new2k rank1 and user-selected iBKD lambda0.25 L16 beta0.5 ratio, fixed10k full val500; not final80k')
    line=MARKER+json.dumps(result,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if len(line)+1>MAX_FINAL_BYTES:
        raise ValueError('Followup terminal report exceeded 50,000 bytes')
    return line
