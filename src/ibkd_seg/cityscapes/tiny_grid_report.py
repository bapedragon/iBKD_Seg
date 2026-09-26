"""Bounded final results for a 65,000-character H200 log tail (stdlib only)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MAX_FINAL_BYTES = 50_000  # Includes marker/newline; leaves 15,000 for the host wrapper.
CLASS_NAMES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
    'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle',
)
TRAJECTORY_KEYS = ('loss', 'ce', 'guidance', 'weighted_guidance_to_seg_loss', 'grad_norm_unclipped')


def bounded_text(value, budget=160):
    """Bound escaped JSON length too, including non-ASCII messages and newlines."""
    if value is None:
        return None
    text = str(value)
    if len(json.dumps(text, ensure_ascii=True)) <= budget:
        return text
    text = text[:budget]
    while len(json.dumps(text + '...', ensure_ascii=True)) > budget:
        text = text[:-1]
    return text + '...'


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
        if not math.isfinite(x):
            return None
        rounded = float(format(x, '.6g'))
        return rounded if math.isfinite(rounded) else x
    except (TypeError, ValueError, OverflowError):
        return None


def count(value):
    x = number(value)
    return None if x is None else int(max(-10**12, min(10**12, float(value))))


def terminal_row(row, plan):
    last = row.get('last') or {}
    scores = row.get('diagnostic_metrics_percent')
    if scores is None and row.get('diagnostic_metrics') is not None:
        raw = row['diagnostic_metrics']
        scores = dict(raw, pixel_accuracy=100 * raw['pixel_accuracy'], miou=100 * raw['miou'],
                      class_iou={k: None if v is None else 100 * v for k, v in raw['class_iou'].items()})
    scores = scores or {}
    checkpoint = row.get('checkpoint') or {}
    controller = row.get('controller') or {}
    trajectory = row.get('trajectory') or {}
    # beta/lambda remain full-precision; other displayed floats use six significant digits.
    result = dict(
        run_id=plan['id'], method=plan['method'], candidate=plan['candidate'],
        beta=plan['beta'], **{'lambda': plan.get('lambda')}, initial_ratio=plan['initial_target_ratio'],
        status=bounded_text(row.get('status', 'not_run'), 48),
        completed_steps=count(row.get('completed_steps', 0)), attempted_step=count(row.get('attempted_step')),
        selected_step=count(row.get('selected_step')), selected_epoch=count(row.get('selected_epoch')),
        last_epoch=count(last.get('epoch')), last_beta=number(last.get('beta')),
        loss=number(last.get('loss')), ce=number(last.get('ce')), guidance=number(last.get('guidance')),
        alignment=number(last.get('alignment')), fusion=number(last.get('fusion')),
        weighted_guidance=number(last.get('weighted_guidance')),
        weighted_guidance_to_ce=number(last.get('weighted_guidance_to_seg_loss')),
        gradient_norm=number(last.get('grad_norm_unclipped')), lr=number(last.get('lr')),
        pixel_accuracy_pct=number(scores.get('pixel_accuracy')), miou_pct=number(scores.get('miou')),
        class_iou_pct=[number((scores.get('class_iou') or {}).get(k)) for k in CLASS_NAMES],
        evaluated_classes=count(scores.get('evaluated_classes')), valid_pixels=count(scores.get('valid_pixels')),
        validation_samples=count(row.get('validation_samples')),
        guidance_active=controller.get('active'), stop_epoch=count(row.get('guidance_stop_epoch')),
        stop_step=count(row.get('guidance_stop_step')), teacher_frozen=row.get('teacher_frozen_verified'),
        checkpoint_step=count(checkpoint.get('saved_step')),
        checkpoint_roundtrip=bounded_text(checkpoint.get('strict_state_roundtrip'), 32),
        warning_count=count(sum(int(w.get('count', 0)) for w in row.get('warning_summary', []))),
        deterministic_warning_count=count(row.get('deterministic_warning_count', 0)),
        failure_stage=bounded_text(row.get('failure_stage'), 64), error_brief=bounded_text(row.get('error')),
        median_step_seconds=number(row.get('median_step_seconds')),
        train_wall_seconds=number(row.get('train_wall_seconds')),
        train_peak_allocated_bytes=count(row.get('train_peak_allocated_bytes')),
        trajectory={k: [number((trajectory.get(k) or {}).get(stat)) for stat in
                        ('first25_median', 'last25_median', 'maximum', 'minimum')] for k in TRAJECTORY_KEYS},
    )
    if row.get('failure_observation'):
        observation = row['failure_observation']
        result['failure_observation'] = {
            key: bounded_text(observation[key], 96) if isinstance(observation[key], str)
            else observation[key] if isinstance(observation[key], bool) else number(observation[key])
            for key in ('step', 'phase', 'loss', 'ce', 'guidance', 'alignment', 'fusion',
                        'weighted_guidance', 'gradient_norm', 'parameters_optimizer_finite', 'input_sha256')
            if key in observation}
    return result


def final_line(report, plans, *, child=False):
    """All planned candidates survive truncation of verbose diagnostic fields."""
    if len(plans) > 24:
        raise ValueError('This terminal schema supports at most 24 candidates')
    by_id = {r['run_id']: r for r in report.get('runs', []) if r.get('run_id')}
    rows = [terminal_row(by_id.get(p['id'], {}), p) for p in plans]
    checks = report.get('cross_checks') or {}
    steps = report.get('expected_steps', 500)
    result = dict(
        status=bounded_text(report.get('status'), 48), protocol_id=bounded_text(report.get('protocol_id'), 128),
        expected_runs=len(plans), reported_runs=len(rows), expected_steps=steps, runs=rows,
        finite_candidates=[r['run_id'] for r in rows if r['status'] == 'passed'],
        failed_candidates=[r['run_id'] for r in rows if r['status'] not in ('passed', 'not_run')],
        not_run_candidates=[r['run_id'] for r in rows if r['status'] == 'not_run'],
        lg_alg_shared_screen=report.get('lg_alg_shared_screen', False),
        full_validation=False, diagnostic_validation_samples=2,
        selection_rule=f'fixed_{steps}_endpoint_not_best_checkpoint',
        scientific_result=False, beta_ranking_performed=False, automatic_next_stage=False, test_used=False,
        class_iou_order=CLASS_NAMES,
        trajectory_order=['first25_median', 'last25_median', 'maximum', 'minimum'],
        display_significant_digits=6, beta_full_precision=True,
        identity_checks={bounded_text(k, 64): bool(v) for k, v in list(checks.get('identity_checks', {}).items())[:8]},
        cross_check_review_count=len(checks.get('review_items', [])),
        lg_alg_scope=bounded_text(checks.get('lg_alg_scope'), 96),
        error_brief=bounded_text(report.get('error')), pipeline_exit_code=count(report.get('pipeline_exit_code')),
        artifact_root=bounded_text(report.get('summary_path') or report.get('output_root'), 512),
        details='Full precision, warnings, traces and checkpoint hashes: grid_summary.json and each run/summary.json',
        terminal_budget_bytes=MAX_FINAL_BYTES,
    )
    prefix = ('[TI16_GRID_RUN_FINAL] ' if child else '[CITYSCAPES_TI16_HIGH_BETA100_FINAL] '
              if steps == 100 else '[CITYSCAPES_TI16_GRID500_FINAL] ')
    encode = lambda obj: prefix + json.dumps(obj, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    line = encode(result)
    if len(line) + 1 > MAX_FINAL_BYTES:
        # Retain every field/metric; factor repeated keys out in pathological cases.
        result['run_columns'] = list(rows[0])
        result['runs'] = [[r[k] for k in result['run_columns']] for r in rows]
        line = encode(result)
    if len(line) + 1 > MAX_FINAL_BYTES:
        raise ValueError('Terminal schema exceeded its tested bound')
    return line


def main():
    """Shell EXIT fallback works even when pip/torch installation failed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    path = args.output_root / 'artifacts/grid_summary.json'
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        report = dict(status='runtime_failure', error='Pipeline setup failed; see run.log', runs=[])
    if report.get('status') in ('passed', 'running'):
        report['status'] = 'runtime_failure'
    report.update(pipeline_exit_code=args.exit_code, output_root=str(args.output_root),
                  protocol_id=config['protocol_id'], expected_steps=config['steps'],
                  lg_alg_shared_screen=config.get('lg_alg_shared_screen', False))
    line = final_line(report, config['runs'])
    (args.output_root / 'terminal_summary.log').write_text(line + '\n', encoding='ascii')
    print(line, flush=True)


if __name__ == '__main__':
    main()
