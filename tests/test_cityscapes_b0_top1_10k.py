"""Selected-method routing and full-state resume contracts for the 10k job."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ibkd_seg.cityscapes.b0 import screen as base
from ibkd_seg.cityscapes.b0_warmup20 import screen as warm

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'b0_top1_job', ROOT / 'phase4/Cityscapes_SegFormer-B0/scripts/run_b0_top1_10k.py')
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def inputs():
    profiles = {name: (module, *module.specification()) for name, module in
                [('b0', base), ('b0_warmup20', warm)]}
    plan = json.loads(job.PLAN.read_text())
    return plan, profiles, job.inventory_for(plan, profiles)


def test_selected_candidates_routing_and_shared_budget():
    plan, profiles, rows = inputs()
    assert [r['candidate']['run_id'] for r in rows] == ['alg_beta_r003', 'ibkd_lambda025_beta_r003']
    assert [r['candidate']['beta'] for r in rows] == [.197479, .387021]
    assert [r['guidance_warmup_epochs'] for r in rows] == [0, 20]
    assert [r['candidates'][1]['beta'] for r in plan['retained'].values()] == [.460784, 1.9351]
    assert not plan['selection']['current_job_performs_final_selection']
    assert job.MAX_JOB_SECONDS == 34800
    for row in rows:
        command = job.worker_command(row, 'cache', 'data', 'output', 'plan', 'preflight', 12345., 'resume.json')
        assert command[2] == 'ibkd_seg.cityscapes.' + row['profile'] + '.screen'
        assert command[command.index('--target-steps') + 1] == '10000'
        assert command[command.index('--deadline') + 1] == '12345.0'
        assert command[-2:] == ['--resume', 'resume.json']
    bad = copy.deepcopy(plan)
    bad['retained']['ibkd_lambda025']['candidates'][0]['beta'] = .903048
    with pytest.raises(ValueError, match='frozen grid'):
        job.inventory_for(bad, profiles)


@pytest.mark.parametrize('index', [0, 1])
def test_resume_accepts_only_matching_full_validation_candidate(tmp_path, index):
    _, profiles, rows = inputs(); row = rows[index]
    module, config, _ = profiles[row['profile']]
    signature = dict(candidate=row['candidate'], protocol_id=config['id'], source=module.code_identity(),
                     protocol_sha256=job.hashlib.sha256(module.CONFIG.read_bytes()).hexdigest())
    summary = dict(**row['candidate'], signature=signature, completed_steps=2000,
                   controller=dict(controller=dict(warmup_epochs=row['guidance_warmup_epochs'])))
    (tmp_path / 'resume.json').write_text('{}')
    def write(value):
        (tmp_path / 'summary.json').write_text(json.dumps(value))
    write(summary)
    assert job.validate_resume(tmp_path, row, module, config)['completed_steps'] == 2000
    for mutation, error in [
        (lambda s: s['signature'].update(diagnostic_scope={}), 'Diagnostic'),
        (lambda s: s['controller']['controller'].update(warmup_epochs=99), 'warm-up'),
        (lambda s: s['signature'].update(source={}), 'signature'),
        (lambda s: s.update(completed_steps=10400), 'exceeds'),
        (lambda s: s.update(beta=99.), 'candidate'),
    ]:
        bad = copy.deepcopy(summary); mutation(bad); write(bad)
        with pytest.raises(ValueError, match=error):
            job.validate_resume(tmp_path, row, module, config)
    (tmp_path / 'resume.json').unlink()
    with pytest.raises(ValueError, match='Missing full-state'):
        job.validate_resume(tmp_path, row, module, config)


def test_invalid_resume_does_not_overwrite_existing_output(tmp_path, monkeypatch, capsys):
    original = '{"original": true}\n'
    (tmp_path / 'group_summary.json').write_text(original)
    monkeypatch.setenv('B0_10K_OUTPUT', str(tmp_path))
    for name in ('B0_10K_RESUME_FROM', 'B0_10K_ALG_RESUME_FROM', 'B0_10K_IBKD_RESUME_FROM'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(job.sys, 'argv', ['run_b0_top1_10k.py'])
    with pytest.raises(SystemExit) as error:
        job.main()
    assert error.value.code == 1
    assert (tmp_path / 'group_summary.json').read_text() == original
    final = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert final['status'] == 'failed' and 'Nonempty output' in final['error']


@pytest.mark.parametrize('budget_exhausted', [False, True])
def test_job_dispatch_and_time_pause_without_gpu(tmp_path, monkeypatch, capsys, budget_exhausted):
    import numpy as np
    import torch
    from ibkd_seg.cityscapes.b0 import assets, calibration_data, training_data, reproducibility
    _, profiles, rows = inputs()
    for name, (module, config, grid) in profiles.items():
        grid = {**grid, 'calibration_tensor_sha256': job.hashlib.sha256().hexdigest()}
        monkeypatch.setattr(module, 'specification', lambda c=config, g=grid: (c, g))
    grid = base.specification()[1]
    data = tmp_path / 'data'; data.mkdir()
    for name in ('manifest.json', 'preparation.json'):
        (data / name).write_text('{}')
    monkeypatch.setenv('B0_10K_OUTPUT', str(tmp_path / 'output'))
    monkeypatch.setenv('B0_10K_DATA', str(data))
    for name in ('B0_10K_RESUME_FROM', 'B0_10K_ALG_RESUME_FROM', 'B0_10K_IBKD_RESUME_FROM'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(job.sys, 'argv', ['run_b0_top1_10k.py'])
    monkeypatch.setattr(reproducibility, 'configure_runtime', lambda: None)
    monkeypatch.setattr(assets, 'prepare', lambda *a, **k: {})
    monkeypatch.setattr(calibration_data, 'verify_preparation', lambda *a: dict(manifest_sha256=grid['manifest_sha256']))
    monkeypatch.setattr(training_data, 'make_plan', lambda *a: np.zeros((2, 5), dtype=np.int32))
    monkeypatch.setattr(training_data, 'PlannedDataset', lambda *a: None)
    monkeypatch.setattr(training_data, 'calibration_digest_update', lambda *a: None)
    monkeypatch.setattr(training_data, 'batches', lambda *a, **k: iter([
        (torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long), ['sample'])] * 32))
    monkeypatch.setattr(job.subprocess, 'check_output', lambda *a, **k: 'test-commit')
    clock = [0.]; commands = []
    monkeypatch.setattr(job.time, 'time', lambda: clock[0])
    def run(command, **kwargs):
        if len(command) > 2 and command[2].endswith('.screen'):
            commands.append(command)
            candidate = next(r['candidate'] for r in rows if r['candidate']['run_id'] == command[command.index('--run-id') + 1])
            target = Path(command[command.index('--output') + 1]); target.mkdir(parents=True)
            record = dict(**candidate, status='paused' if budget_exhausted else 'completed',
                          completed_steps=1000 if budget_exhausted else 10000,
                          last_loss={'ce': 1.}, selected_epoch=None, selected_step=10000,
                          metrics={'miou': .5}, first_25_batch_hashes=['same'], signature={'environment': {'gpu': 'mock'}})
            (target / 'summary.json').write_text(json.dumps(record))
            if budget_exhausted:
                clock[0] = 34600.
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(job.subprocess, 'run', run)
    with pytest.raises(SystemExit) as error:
        job.main()
    assert error.value.code == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result['job_runtime']['maximum_job_seconds'] == 34800
    assert all(r['status'] == 'pending' for r in result['selection'].values())
    assert result['status'] == ('paused' if budget_exhausted else 'completed')
    assert len(commands) == (1 if budget_exhausted else 2)
    assert {c[c.index('--deadline') + 1] for c in commands} == {'34620.0'}
    if budget_exhausted:
        assert result['runs'][-1]['status'] == 'pending' and result['runs'][-1]['completed_steps'] == 0
    else:
        assert result['completed_runs'] == 2 and all(r['last_loss'] and r['metrics'] for r in result['runs'])
