import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ibkd_seg.cityscapes import tiny_grid as grid
from ibkd_seg.cityscapes.tiny_grid_report import final_line, main as report_main


class HighBetaTests(unittest.TestCase):
    def config(self):
        return grid.load_config(grid.CONFIG_DIR / grid.HIGH_BETA100)

    def test_authorized_betas_and_original_training_protocol(self):
        c = self.config()
        self.assertEqual([p['beta'] for p in c['runs']],
                         [2.5, 13.588629717453085, 2.5, 35.715049953558584, 2.5, 53.428435805136004])
        self.assertEqual([p.get('lambda') for p in c['runs']], [None, None, .25, .25, .5, .5])
        self.assertEqual(c['steps'], 100)
        self.assertEqual(c['classification_reference']['request'], 841)
        self.assertEqual(c['total_steps'], 80000)
        self.assertFalse(c['gradient_clipping'])
        self.assertEqual(c['lr_warmup_steps'], 0)
        self.assertEqual(c['diagnostic_validation_samples'], 2)
        self.assertFalse(c['finite_spike_auto_stop'])
        self.assertFalse(c['automatic_next_stage'])
        for plan in c['runs'][1::2]:
            self.assertAlmostEqual(plan['initial_target_ratio'],
                c['classification_reference']['ratios'][plan['id'].removesuffix('_ratio')])

    def test_protocol_drift_and_wrong_beta_are_rejected(self):
        for field, value in [('learning_rate', .001), ('steps', 500), ('gradient_clipping', True)]:
            c = self.config(); c[field] = value
            with self.assertRaises(ValueError): grid.validate_high_beta(c)
        c = self.config(); c['runs'][0]['beta'] = 2.6
        with self.assertRaises(ValueError): grid.validate_high_beta(c)

    def test_failure_scalars_preserve_nonfinite_kind_in_valid_json(self):
        values = [grid.diagnostic_scalar(v) for v in [math.inf, -math.inf, math.nan, 1e30]]
        self.assertEqual(values[:3], ['+Inf', '-Inf', 'NaN'])
        self.assertEqual(json.loads(json.dumps(values, allow_nan=False)), values)

    def fixture(self, plan, failed=False):
        n = 3 if failed else 100
        scalar = dict(loss=3., ce=1., guidance=.8, weighted_guidance=2.,
                      weighted_guidance_to_seg_loss=2., grad_norm_unclipped=4., epoch=1, step=n,
                      beta=plan['beta'], lr=.01)
        row = dict(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],
                   initial_beta=plan['beta'],status='numerical_failure' if failed else 'passed',
                   completed_steps=n,attempted_step=n+1 if failed else n,
                   selected_step=None if failed else n,selected_epoch=None if failed else 1,
                   student_initial_state_sha256='same-student',teacher_state_sha256='same-teacher',
                   guidance_initial_state_sha256=plan['method'],teacher_frozen_verified=True,
                   input_hashes=[str(i) for i in range(n)],losses=[scalar]*n,last=scalar)
        if failed:
            row.update(error='FloatingPointError: nonfinite gradient',failure_stage='training',
                       failure_observation=dict(step=4,phase='gradient_check',loss=3.,ce=1.,
                                                guidance=.8,gradient_norm='+Inf',input_sha256='failed-input'))
        return row

    def test_parent_continues_all_six_after_numerical_failure(self):
        c = self.config(); by_id = {p['id']:p for p in c['runs']}
        def child(command, should_stop):
            self.assertFalse(should_stop())
            run_id = command[command.index('--run-id')+1]
            destination = Path(command[command.index('--output-dir')+1]); destination.mkdir()
            failed = run_id == c['runs'][0]['id']
            (destination/'summary.json').write_text(json.dumps(self.fixture(by_id[run_id], failed)))
            return SimpleNamespace(returncode=1 if failed else 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); capture = io.StringIO()
            argv = ['probe','--cache-root',str(root/'cache'),'--data-dir',str(root/'data'),
                    '--manifest',str(root/'manifest.json'),'--output-dir',str(root/'out'),
                    '--config',str(grid.CONFIG_DIR/grid.HIGH_BETA100)]
            with patch('sys.argv',argv), patch('ibkd_seg.cityscapes.official_api.bootstrap'), \
                 patch('ibkd_seg.cityscapes.official_assets.verify',return_value={'weights':{}}), \
                 patch('ibkd_seg.cityscapes.full_data.prepare_labels'), \
                 patch('ibkd_seg.cityscapes.tiny_screen2000.launch_child',side_effect=child) as launch, \
                 contextlib.redirect_stdout(capture):
                with self.assertRaises(SystemExit) as caught: grid.main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(launch.call_count, 6)
            last = capture.getvalue().splitlines()[-1]
            self.assertTrue(last.startswith('[CITYSCAPES_TI16_HIGH_BETA100_FINAL] '))
            result = json.loads(last.split('] ',1)[1])
            self.assertEqual(result['expected_steps'],100)
            self.assertEqual(result['selection_rule'],'fixed_100_endpoint_not_best_checkpoint')
            self.assertEqual(len(result['runs']),6)
            self.assertEqual(len(result['finite_candidates']),5)
            self.assertEqual(result['runs'][0]['failure_observation']['gradient_norm'],'+Inf')
            self.assertEqual(result['runs'][0]['completed_steps'],3)
            self.assertEqual(result['runs'][0]['attempted_step'],4)
            self.assertEqual(result['cross_check_review_count'],0)
            self.assertFalse(result['automatic_next_stage'])

    def test_all_failures_keep_values_and_fit_log_tail(self):
        c = self.config()
        report = dict(status='needs_review',expected_steps=100,protocol_id=c['protocol_id'],
                      runs=[self.fixture(p,True) for p in c['runs']])
        for r in report['runs']: r['error'] = '상세오류'*10000
        line = final_line(report,c['runs'])
        self.assertLess(len(line),50000)
        tail = ('old logs\n'*10000 + line + '\n[3] 완료\n')[-65000:]
        result = json.loads(next(s for s in tail.splitlines() if s.startswith('[CITYSCAPES_TI16_HIGH_BETA100_FINAL]')).split('] ',1)[1])
        self.assertEqual(len(result['failed_candidates']),6)
        self.assertEqual([r['beta'] for r in result['runs']], [p['beta'] for p in c['runs']])
        print('100-step six-run all-failure terminal bytes:',len(line))

    def test_fallback_includes_six_not_run_rows_on_setup_failure(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as capture:
            with patch('sys.argv',['report','--output-root',tmp,'--config',str(grid.CONFIG_DIR/grid.HIGH_BETA100),'--exit-code','73']):
                report_main()
            result = json.loads(capture.getvalue().split('] ',1)[1])
            self.assertEqual(result['expected_steps'],100)
            self.assertEqual(len(result['not_run_candidates']),6)
            self.assertEqual(result['pipeline_exit_code'],73)

    def test_shell_install_failure_never_starts_training_and_keeps_final_json(self):
        repo = grid.CONFIG_DIR.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bindir=root/'bin'; bindir.mkdir()
            fake=bindir/'python'
            fake.write_text('#!/bin/bash\nif [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 73; fi\nexec '+sys.executable+' "$@"\n')
            timer=bindir/'timeout'; timer.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
            for p in (fake,timer): p.chmod(0o755)
            env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'))
            p=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_high_beta100.sh')],
                             cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,73,p.stderr)
            result=json.loads(p.stdout.strip().splitlines()[-1].split('] ',1)[1])
            self.assertEqual(result['expected_steps'],100)
            self.assertEqual(len(result['not_run_candidates']),6)
            self.assertFalse((root/'out/manifest.json').exists())


if __name__=='__main__': unittest.main()
