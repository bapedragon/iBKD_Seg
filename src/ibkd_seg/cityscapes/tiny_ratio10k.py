"""Provenance and initial-ratio checks for the authorized two-run Tiny 10k pack."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .tiny_grid_report import number

PACK='alg_top1_ibkd025_l16ratio_10k'
PROTOCOL='cityscapes_ti16_alg_top1_ibkd025_l16ratio_10k_v1'
ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'phase4/Cityscapes_Segmenter-Ti16/experiments/l16_ratio_match_v1/followup10k_reference.json'


def validate_config(config):
    from .tiny_screen2000 import FOLLOWUP_CONFIG,load_config
    original=load_config(FOLLOWUP_CONFIG)
    original.pop('selection_reference')
    ref=json.loads(REFERENCE.read_text())
    for source in (ref['alg_selection'],ref['tiny_calibration']):
        if hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Tiny 10k selection/calibration source changed')
    results=json.loads((ROOT/ref['alg_selection']['path']).read_text())['terminal_report']
    rows=results['runs']
    if (results['status']!='passed' or len(rows)!=4 or
            any(r['status']!='passed' or r['selected_step']!=2000 or
                r['validation_samples']!=500 or not r['full_validation'] for r in rows)):
        raise ValueError('Expected four complete full-val ALG selection results')
    ranked=sorted(rows,key=lambda r:-r['miou_pct'])
    alg=ranked[0];selected=ref['alg_selection']
    if (alg['miou_pct']==ranked[1]['miou_pct'] or alg['candidate']!=3 or
            any(alg[k]!=selected[k] for k in ('run_id','candidate','beta','miou_pct')) or
            alg['initial_ratio']!=selected['target_ratio']):
        raise ValueError('ALG must use the new ratio-matched 2k mIoU rank1 beta')
    ibkd=ref['ibkd_reference'];cal=ref['tiny_calibration']
    ratio=ibkd['l16_beta']*ibkd['guidance']/ibkd['ce']
    if (ibkd['l16_beta']!=.5 or ibkd['lambda_value']!=.25 or ratio!=ibkd['target_ratio']):
        raise ValueError('iBKD must match the requested L16 beta0.5 lambda0.25 first-step ratio')
    plans=[dict(id='alg_l16r3_10k',method='alg',candidate=3,beta=alg['beta'],
                initial_target_ratio=alg['initial_ratio'],run_index=1,target_steps=10000),
           dict(id='ibkd_l025_l16b05_10k',method='ibkd',candidate=3,
                beta=ratio*cal['ce']/cal['guidance'],initial_target_ratio=ratio,
                **{'lambda':.25},run_index=2,target_steps=10000)]
    expected=dict(original,protocol_id=PROTOCOL,pack=PACK,
        run_kind='two_fixed10000_endpoints_alg_new_rank1_ibkd_l16_beta0p5_ratio',
        beta_initial_ce_ratio=None,
        beta_status='fixed_first_step_l16_targets_alg_new2k_rank1_ibkd_user_selected_no_recalibration',
        selection_rule='fixed_10000_endpoint_not_best_checkpoint',validation_log_tag='TI16_RATIO10K_VAL',
        first_step_ratio_check=dict(rtol=1e-4,atol=1e-8),
        ratio_reference=dict(path=str(REFERENCE.relative_to(ROOT)),sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()),
        runs=plans)
    if config!=expected:
        raise ValueError('Ratio10k changed the locked beta, lambda, endpoints or common Tiny protocol')
    return config


def ratio_fields(raw,plan):
    first=next((r for r in raw.get('losses',[]) if r.get('step')==1),None)
    if first is None:
        first=raw.get('first_step_ratio_observation') or {}
    ce,guidance=first.get('ce'),first.get('guidance')
    actual=None if ce is None or guidance is None or ce<=0 else plan['beta']*guidance/ce
    return dict(initial_ratio_basis='l16_first_training_batch_target',
        initial_ratio_target_percent=100*plan['initial_target_ratio'],
        initial_ratio_first_step_percent=number(None if actual is None else 100*actual),
        first_step_ce=number(ce),first_step_guidance=number(guidance),
        first_step_ratio_matches_target=None if actual is None else math.isclose(
            actual,plan['initial_target_ratio'],rel_tol=1e-4,abs_tol=1e-8),
        first_step_ratio_basis='100*beta*guidance/ce_before_first_optimizer_update',
        selection_basis='tiny_new_ratio2000_full_val_miou_rank1' if plan['method']=='alg' else
                        'user_requested_l16_beta0p5_ratio_direct10k_without_new_tiny2k',
        l16_reference_beta=.1 if plan['method']=='alg' else .5,
        same_beta_tiny2000_completed=plan['method']=='alg')
