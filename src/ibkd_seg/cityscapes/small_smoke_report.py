"""Small smoke terminal summary, including setup failures; standard library only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tiny_grid_report import CLASS_NAMES, bounded_text, count, number

MARKER = '[CITYSCAPES_S16_SMOKE_FINAL] '
MAX_FINAL_BYTES = 50_000


def terminal_row(row, plan):
    last = row.get('last') or {}
    scores = row.get('diagnostic_metrics_percent') or {}
    calibration = row.get('calibration') or {}
    checkpoint = row.get('checkpoint') or {}
    return dict(
        run_id=plan['id'], method=plan['method'], display_name=plan.get('display_name', plan['method']),
        **{'lambda': plan.get('lambda')}, status=bounded_text(row.get('status', 'not_run'), 48),
        completed_steps=count(row.get('completed_steps', 0)), selected_step=count(row.get('selected_step')),
        selected_epoch=count(row.get('selected_epoch')), loss=number(last.get('loss')),
        ce=number(last.get('ce')), guidance=number(last.get('guidance')),
        weighted_guidance=number(last.get('weighted_guidance')),
        weighted_guidance_to_seg_loss=number(last.get('weighted_guidance_to_seg_loss')),
        alignment=number(last.get('alignment')), fusion=number(last.get('fusion')),
        grad_norm=number(last.get('grad_norm_unclipped')),
        pilot_beta=calibration.get('pilot_beta'), beta_candidates=calibration.get('beta_candidates', []),
        calibration_ce_median=number(calibration.get('ce_median')),
        calibration_guidance_median=number(calibration.get('guidance_median')),
        measured_ratio_medians=[number(x.get('weighted_guidance_to_seg_loss_median'))
                                for x in calibration.get('measured_ratios', [])],
        calibration_state_restored=row.get('calibration_state_restored'),
        calibration_beta_source=bounded_text(calibration.get('beta_source', 'own_initial_training_batches')),
        loss_coefficients=row.get('fixed_loss_coefficients'),
        raw_components={k:number(v) for k,v in (last.get('components') or {}).items()},
        weighted_components={k:number(v) for k,v in (last.get('weighted_components') or {}).items()},
        pixel_accuracy_pct=number(scores.get('pixel_accuracy')), miou_pct=number(scores.get('miou')),
        class_iou_pct=[number((scores.get('class_iou') or {}).get(k)) for k in CLASS_NAMES],
        evaluated_classes=count(scores.get('evaluated_classes')), valid_pixels=count(scores.get('valid_pixels')),
        validation_samples=count(row.get('validation_samples')), full_validation=False,
        teacher_frozen=row.get('teacher_frozen_verified'),
        checkpoint_reload=checkpoint.get('strict_reload'), replayed_update=checkpoint.get('replayed_update'),
        natural_guidance_off_tested=False, guidance_on=row.get('guidance_on'),
        median_step_seconds=number(row.get('median_step_seconds_excluding_first')),
        train_peak_allocated_bytes=count(row.get('train_peak_allocated_bytes')),
        deterministic_warning_count=count(row.get('deterministic_warning_count')),
        error=bounded_text(row.get('error'), 500), summary_path=bounded_text(row.get('summary_path'), 500),
    )


def final_line(report, config, *, only_run=None):
    by_id = {row['run_id']:row for row in report.get('runs', [])}
    plans = [p for p in config['runs'] if only_run is None or p['id'] == only_run]
    result = dict(
        status=report.get('status', 'failed'), protocol_id=config['protocol_id'],
        student=config['student_backbone'], teacher=config['teacher'],
        scientific_result=False, test_used=False, automatic_next_stage=False,
        score_scope='25-step endpoint, first 2 val images; not full-val or beta ranking',
        beta_scope='initial training loss-scale proposals; not best beta or constant training ratio',
        resume_scope='one replayed update only; not long-run resume validation',
        source_and_asset_verification=report.get('source_and_asset_verification'),
        assets=report.get('assets'), cross_method_checks=report.get('cross_method_checks'),
        class_iou_order=list(CLASS_NAMES), expected_runs=len(plans),
        runs=[terminal_row(by_id.get(p['id'], {}), p) for p in plans],
        pipeline_exit_code=report.get('pipeline_exit_code'),
        error=bounded_text(report.get('error'), 800), summary_path=bounded_text(report.get('summary_path'), 500),
    )
    line = MARKER + json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    if len((line + '\n').encode()) > MAX_FINAL_BYTES:
        raise RuntimeError('Small summary exceeds the final log budget; preserve full report on disk')
    return line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    path = args.output_root / 'artifacts/smoke_summary.json'
    report = json.loads(path.read_text()) if path.exists() else dict(
        status='failed', runs=[], error='Pipeline setup failed; see run.log')
    report.update(pipeline_exit_code=args.exit_code, summary_path=str(path))
    if args.exit_code:
        report['status'] = 'failed'
    line = final_line(report, config)
    (args.output_root / 'terminal_summary.json').write_text(line[len(MARKER):] + '\n')
    print(line, flush=True)


if __name__ == '__main__':
    main()
