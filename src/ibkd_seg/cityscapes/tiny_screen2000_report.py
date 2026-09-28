"""Bounded full-val endpoint results and setup-failure reporting, with stdlib only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tiny_grid_report import CLASS_NAMES, MAX_FINAL_BYTES, bounded_text, number, terminal_row

MARKER = '[CITYSCAPES_TI16_GRID2000_FINAL] '


def select_plans(config, *, start_candidate=1, start_run=None, run_id=None):
    from .tiny_followup import PACKS
    if config.get('pack') in ('lg_alg','fskd_c2vkd')+PACKS:
        first=1 if start_run is None else start_run
        if start_candidate!=1 or not 1<=first<=len(config['runs']):
            raise ValueError('This pack uses --start-run in its fixed execution order, not --start-candidate')
        plans=config['runs'][first-1:]
    else:
        if start_run is not None or not 1<=start_candidate<=8:
            raise ValueError('iBKD pack uses --start-candidate 1..8')
        if config.get('pack')=='ibkd_l050_4betas' and start_candidate not in (1,3,6,8):
            raise ValueError('iBKD lambda0.5 uses original candidate IDs 1,3,6,8')
        plans=[p for p in config['runs'] if p['candidate']>=start_candidate]
    if run_id:
        plans=[p for p in plans if p['id']==run_id]
    if not plans:
        raise ValueError('Unknown run ID or empty execution range')
    return plans


def final_line(report, plans):
    from .tiny_followup import PACKS,PROTOCOLS
    if report.get('pack') in PACKS or report.get('protocol_id') in PROTOCOLS:
        from .tiny_followup_report import final_line as followup_final_line
        return followup_final_line(report,plans)
    legacy=report.get('protocol_id')=='cityscapes_ti16_crop512_ibkd_l025_grid2000_v1'
    budget=report.get('job_budget_seconds',35100 if legacy else 36000)
    reserve=report.get('save_reserve_seconds',180 if legacy else 120)
    lg_alg=(report.get('pack')=='lg_alg' or
            report.get('protocol_id')=='cityscapes_ti16_crop512_lg_alg_grid2000_v1')
    ibkd050=(report.get('pack')=='ibkd_l050_4betas' or
             report.get('protocol_id')=='cityscapes_ti16_crop512_ibkd_l050_grid2000_4betas_v1')
    fixed=(report.get('pack')=='fskd_c2vkd' or
           report.get('protocol_id')=='cityscapes_ti16_crop512_fskd_c2vkd_grid2000_v1')
    pack='fskd_c2vkd' if fixed else 'lg_alg' if lg_alg else 'ibkd_l050_4betas' if ibkd050 else 'ibkd_l025'
    by_id = {r['run_id']:r for r in report.get('runs', [])}
    rows = []
    for plan in plans:
        raw = by_id.get(plan['id'], {})
        row = terminal_row(raw, plan)
        if lg_alg or fixed:
            row['run_index']=plan['run_index']
        if fixed:
            last=raw.get('last') or {}
            provenance=raw.get('method_provenance') or {}
            pool=raw.get('clip_pool') or {}
            row.update(display_name=plan['display_name'],comparison_group=plan['comparison_group'],
                       author_exact_reproduction=False,controller_applicable=False,
                       guidance_active=None if not raw.get('completed_steps') else True,
                       beta_search=False,standalone_ce_coefficient=number(last.get('standalone_ce_coefficient')),
                       guidance_multiplier=number(last.get('guidance_multiplier')),
                       fixed_loss_coefficients=raw.get('fixed_loss_coefficients'),
                       loss_components={k:number(v) for k,v in (last.get('components') or {}).items()},
                       weighted_loss_components={k:number(v) for k,v in (last.get('weighted_components') or {}).items()},
                       method_id=provenance.get('method_id'),source_commit=provenance.get('source_commit'),
                       recipe_scope=bounded_text(provenance.get('interpretation'),512),
                       soft_rank_execution=raw.get('soft_rank_execution'),
                       clip_pool_frozen=pool.get('frozen_no_grad_unchanged'),
                       clip_pool_sha256=(pool.get('asset') or {}).get('sha256'))
        row.update(full_validation=raw.get('full_validation', False),
                   validation_seconds=number(raw.get('validation_seconds')),
                   student_unchanged_during_validation=raw.get('student_unchanged_during_validation'),
                   pause_reason=bounded_text(raw.get('pause_reason'), 256),
                   checkpoint_pointer=bounded_text((raw.get('checkpoint') or {}).get('pointer'), 1024))
        rows.append(row)
    result = dict(status=report.get('status'), protocol_id=report.get('protocol_id'),
                  pack=pack, expected_runs=len(plans), configured_pack_runs=2 if fixed else 16 if lg_alg else 4 if ibkd050 else 8,
                  invocation_start_candidate=report.get('start_candidate', 1), runs=rows,
                  target_steps=2000, schedule_total_steps=80000, seed=1,
                  selection_rule='fixed_2000_endpoint_not_best_checkpoint',
                  planned_validation_samples=500, test_used=False, automatic_next_stage=False,
                  scientific_result=False, beta_ranking_performed=False,
                  score_scope=('full-val fixed 2000-step baseline screen; not final 80000-step performance' if fixed
                               else 'full-val fixed 2000-step beta screen; not final 80000-step performance'),
                  primary_metric='miou' if fixed else 'pixel_accuracy',
                  secondary_metric='pixel_accuracy_at_same_checkpoint' if fixed else 'miou_at_same_checkpoint',
                  completed_candidates=[r['run_id'] for r in rows if r['status']=='passed'],
                  paused_candidates=[r['run_id'] for r in rows if r['status']=='paused'],
                  not_run_candidates=[r['run_id'] for r in rows if r['status']=='not_run'],
                  failed_candidates=[r['run_id'] for r in rows if r['status'] not in ('passed','paused','not_run')],
                  class_iou_order=CLASS_NAMES,
                  trajectory_order=['first25_median','last25_median','maximum','minimum'],
                  identity_checks=report.get('identity_checks', {}),
                  review_items=report.get('review_items', []),
                  job_budget_seconds=budget, save_reserve_seconds=reserve,
                  training_stop_after_seconds=report.get('training_stop_after_seconds',budget-reserve),
                  hard_deadline_unix=report.get('hard_deadline_unix'),stop_at_unix=report.get('stop_at_unix'),
                  total_job_seconds=number(report.get('total_job_seconds')),
                  invocation_seconds=number(report.get('invocation_seconds')),
                  error=bounded_text(report.get('error'), 1024), pipeline_exit_code=report.get('pipeline_exit_code'),
                  summary_path=bounded_text(report.get('summary_path'), 1024),
                  display_significant_digits=6, beta_full_precision=True)
    if lg_alg:
        result.update(invocation_start_run=report.get('start_run') or 1,
                      lg_alg_active_prefix_pairs=report.get('lg_alg_active_prefix_pairs',[]),
                      alg_warmup_epochs=0,alg_earliest_possible_off_step=745,
                      prefix_comparison_scope='observed steps with guidance on in both runs; post-shutdown divergence is expected')
    if ibkd050:
        result.update(configured_candidates=[1,3,6,8],ibkd_warmup_epochs=20,
                      guidance_scope='warmup20 keeps guidance on throughout this 2000-step screen')
    if fixed:
        result.update(invocation_start_run=report.get('start_run') or 1,beta_search=False,
                      recipe_scope='smoke25 fixed transfers; FSKD* primary, C2VKD* supplementary CLIP-pool',
                      next_stage='review_both_2000_step_full_val_results_before_separate_10000_step_job')
    line = MARKER + json.dumps(result, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    if len(line)+1 > MAX_FINAL_BYTES:
        raise ValueError('2000-step terminal report exceeded 50,000 bytes')
    return line


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--exit-code',type=int,required=True)
    parser.add_argument('--start-candidate',type=int,default=1,choices=range(1,9))
    parser.add_argument('--start-run',type=int,choices=range(1,17))
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    path=args.output_root/'artifacts/grid_summary.json'
    try:
        report=json.loads(path.read_text())
    except (OSError, ValueError):
        report=dict(status='runtime_failure',runs=[],error='Pipeline setup failed; see run.log')
    if report.get('status') in ('passed','running'):
        report['status']='runtime_failure'
    report.update(protocol_id=config['protocol_id'],pipeline_exit_code=args.exit_code,
                  job_budget_seconds=config['job_budget_seconds'],save_reserve_seconds=config['save_reserve_seconds'],
                  start_candidate=args.start_candidate,start_run=args.start_run,pack=config['pack'],summary_path=str(path))
    plans=select_plans(config,start_candidate=args.start_candidate,start_run=args.start_run)
    line=final_line(report,plans)
    (args.output_root/'terminal_summary.log').write_text(line+'\n',encoding='ascii')
    print(line,flush=True)


if __name__=='__main__':
    main()
