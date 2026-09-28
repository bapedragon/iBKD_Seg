"""Locked Tiny followup contracts, independent of GPU dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .tiny_ratio2000 import PACK as RATIO_PACK,PROTOCOL as RATIO_PROTOCOL
from .tiny_ratio10k import PACK as RATIO10K_PACK,PROTOCOL as RATIO10K_PROTOCOL

PACK='alg_ibkd025_top1_10k_vanilla2k'
PROTOCOL='cityscapes_ti16_alg_ibkd025_top1_10k_vanilla2k_v1'
VANILLA_PACK='vanilla_10k'
VANILLA_PROTOCOL='cityscapes_ti16_vanilla_10k_v1'
IBKD025_B7_PACK='ibkd_l025_b7_10k'
IBKD025_B7_PROTOCOL='cityscapes_ti16_ibkd_l025_b7_10k_v1'
PACKS=(PACK,VANILLA_PACK,IBKD025_B7_PACK,RATIO_PACK,RATIO10K_PACK)
PROTOCOLS=(PROTOCOL,VANILLA_PROTOCOL,IBKD025_B7_PROTOCOL,RATIO_PROTOCOL,RATIO10K_PROTOCOL)


def validate_config(config):
    from .tiny_screen2000 import CONFIG,CONFIG_DIR,LG_ALG_CONFIG,load_config
    original=load_config(CONFIG)
    changed={'protocol_id','run_kind','pack','steps','runs','ibkd_lambdas','beta_multipliers',
             'beta_status','selection_metric','secondary_metric','selection_rule','start_policy','continuation_policy'}
    extra={'selection_reference','validation_log_tag','record_failed_step_details'}
    if (set(config)!=(set(original)|extra) or
            any(config[k]!=v for k,v in original.items() if k not in changed)):
        raise ValueError('Mixed followup changed the common Tiny training/evaluation protocol')
    ref=CONFIG_DIR.parent/'reports/beta_screen/ti16_grid2000_top2_miou_v1.json'
    manifest=json.loads(ref.read_text())
    selected={g['group']:g['selected'][0] for g in manifest['groups'] if g['group'] in ('alg','ibkd_l025')}
    expected_ref=dict(path=str(ref.relative_to(Path(__file__).resolve().parents[3])),
                      sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),selected=selected)
    alg=next(p for p in load_config(LG_ALG_CONFIG)['runs'] if p['id']==selected['alg']['run_id'])
    ibkd=next(p for p in original['runs'] if p['id']==selected['ibkd_l025']['run_id'])
    expected=[dict(id='vanilla_2k',method='vanilla',candidate=None,beta=0.,initial_target_ratio=None,
                   run_index=1,target_steps=2000),
              dict(alg,run_index=2,target_steps=10000),dict(ibkd,run_index=3,target_steps=10000)]
    if (config['runs']!=expected or config['selection_reference']!=expected_ref or
            config['pack']!=PACK or config['protocol_id']!=PROTOCOL or config['steps']!=10000 or
            config['selection_metric']!='miou' or config['ibkd_lambdas']!=[.25] or config['beta_multipliers']):
        raise ValueError('Expected Vanilla2k + ALG rank1 10k + iBKD lambda0.25 rank1 10k')
    return config


def validate_vanilla_config(config):
    from .tiny_screen2000 import FOLLOWUP_CONFIG,load_config
    original=load_config(FOLLOWUP_CONFIG)
    changed={'protocol_id','run_kind','pack','runs','ibkd_lambdas','beta_initial_ce_ratio',
             'beta_status','selection_rule'}
    expected=[dict(original['runs'][0],id='vanilla_10k',target_steps=10000)]
    if (set(config)!=set(original)-{'selection_reference'} or
            any(config[k]!=v for k,v in original.items() if k not in changed|{'selection_reference'}) or
            config['runs']!=expected or config['protocol_id']!=VANILLA_PROTOCOL or
            config['pack']!=VANILLA_PACK or config['ibkd_lambdas'] or
            config['beta_initial_ce_ratio'] is not None or
            config['selection_rule']!='fixed_10000_endpoint_not_best_checkpoint'):
        raise ValueError('Vanilla10k must preserve the shared protocol and contain only CE-only Vanilla')
    return config


def validate_ibkd025_b7_config(config):
    from .tiny_screen2000 import CONFIG,CONFIG_DIR,FOLLOWUP_CONFIG,load_config
    original=load_config(FOLLOWUP_CONFIG)
    ref=CONFIG_DIR.parent/'reports/beta_screen/ti16_grid2000_top2_miou_v1.json'
    group=next(g for g in json.loads(ref.read_text())['groups'] if g['group']=='ibkd_l025')
    selected=next(p for p in group['selected'] if p['rank']==2)
    source=next(p for p in load_config(CONFIG)['runs'] if p['id']==selected['run_id'])
    expected=dict(original,protocol_id=IBKD025_B7_PROTOCOL,pack=IBKD025_B7_PACK,
                  run_kind='fixed_endpoint_10k_ibkd_lambda025_rank2_full_val',
                  beta_status='fixed_full_val500_miou_rank2_from_grid2000_no_reestimation',
                  selection_rule='fixed_10000_endpoint_not_best_checkpoint',
                  runs=[dict(source,run_index=1,target_steps=10000)],
                  selection_reference=dict(path=str(ref.relative_to(Path(__file__).resolve().parents[3])),
                      sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),selected={'ibkd_l025':selected}))
    if (config!=expected or selected['candidate']!=7 or selected['lambda_value']!=.25 or
            selected['beta']!=source['beta']):
        raise ValueError('iBKD lambda0.25 candidate7 10k must preserve the shared protocol and selected rank2 beta')
    return config


def effective_config(config,plan):
    """Only the endpoint varies; the LR schedule stays at 80k for every run."""
    return dict(config,steps=plan['target_steps'],
                selection_rule=f"fixed_{plan['target_steps']}_endpoint_not_best_checkpoint")


def identity_checks(rows,plans):
    expected={p['id']:p for p in plans}
    passed=[r for r in rows if r.get('status')=='passed']
    observed=[r for r in rows if r.get('student_initial_state_sha256')]
    guided=[r for r in observed if r.get('method')!='vanilla']
    checks=dict(initial_student_identity_present=all(r.get('student_initial_state_sha256') for r in passed))
    if observed:
        checks['same_initial_student']=len({r['student_initial_state_sha256'] for r in observed})==1
        reference=max((r.get('input_hashes',[]) for r in observed),key=len)
        checks['same_observed_input_prefixes']=all(
            r.get('input_hashes',[])==reference[:len(r.get('input_hashes',[]))] for r in observed)
    if guided:
        checks['same_guided_teacher']=all(r.get('teacher_state_sha256') for r in guided) and len(
            {r['teacher_state_sha256'] for r in guided})==1
        checks['same_adapter_within_method']=all(
            all(r.get('guidance_initial_state_sha256') for r in guided if r['method']==method) and len(
                {r['guidance_initial_state_sha256'] for r in guided if r['method']==method})==1
            for method in {r['method'] for r in guided})
    checks['method_specific_teacher_and_guidance']=all(
        (r.get('teacher_loaded') is False and r.get('guidance_loaded') is False and r.get('controller') is None
         and r.get('teacher_state_sha256') is None and r.get('guidance_initial_state_sha256') is None)
        if r.get('method')=='vanilla' else
        (r.get('teacher_loaded') is True and r.get('guidance_loaded') is True and
         r.get('teacher_frozen_verified') is True and bool(r.get('controller')))
        for r in passed)
    checks['complete_planned_endpoints']=all(
        r.get('completed_steps')==r.get('selected_step')==expected[r['run_id']]['target_steps'] and
        r.get('target_steps')==expected[r['run_id']]['target_steps'] and r.get('full_validation') is True and
        r.get('validation_samples')==500 and (r.get('diagnostic_metrics') or {}).get('evaluated_classes')==19 and
        len(r.get('input_hashes',[]))==expected[r['run_id']]['target_steps'] and
        r.get('student_unchanged_during_validation') is True and
        (r.get('checkpoint') or {}).get('strict_state_roundtrip')=='passed'
        for r in passed)
    return checks,[k for k,v in checks.items() if not v]
