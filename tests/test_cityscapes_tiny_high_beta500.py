import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ibkd_seg.cityscapes import tiny_grid as grid
from ibkd_seg.cityscapes.tiny_grid_report import main as report_main


class HighBeta500Tests(unittest.TestCase):
    def config(self):
        return grid.load_config(grid.CONFIG_DIR / grid.HIGH_BETA500)

    def fixture(self, plan, failed=False):
        n = 300 if failed else 500
        losses = [dict(step=s, epoch=1 if s <= 372 else 2, beta=2.5, loss=2.5-s/1000,
                       ce=2.-s/1000, guidance=.2, weighted_guidance=.5,
                       weighted_guidance_to_seg_loss=.5/(2.-s/1000), grad_norm_unclipped=3., lr=.01)
                  for s in range(1, n+1)]
        return dict(protocol_id=self.config()['protocol_id'], run_id=plan['id'], method=plan['method'],
                    candidate=1, initial_beta=2.5, **{'lambda': plan.get('lambda')},
                    status='numerical_failure' if failed else 'passed', completed_steps=n,
                    attempted_step=n+1 if failed else n, selected_step=None if failed else n,
                    selected_epoch=None if failed else 2, input_hashes=[str(i) for i in range(n)],
                    student_initial_state_sha256='student', teacher_state_sha256='teacher',
                    guidance_initial_state_sha256=plan['method'], teacher_frozen_verified=True,
                    losses=losses, last=losses[-1], trajectory=grid.trajectory_summary(losses),
                    failure_stage='training' if failed else None,
                    error='nonfinite gradient' if failed else None)

    def test_protocol_matches_100_probe_and_only_two_authorized_conditions(self):
        c = self.config()
        self.assertEqual([(p['method'], p.get('lambda'), p['beta']) for p in c['runs']],
                         [('lg', None, 2.5), ('ibkd', .25, 2.5)])
        self.assertEqual(c['steps'], 500)
        self.assertEqual(c['total_steps'], 80000)
        self.assertEqual(c['diagnostic_validation_samples'], 2)
        self.assertEqual(c['ibkd_warmup_epochs'], 20)
        self.assertEqual(c['alg_warmup_epochs'], 0)
        self.assertEqual(c['screening_reference']['request'], 842)
        self.assertFalse(c['automatic_next_stage'])
        self.assertFalse(c['gradient_clipping'])
        self.assertEqual((c['job_budget_seconds'], c['save_reserve_seconds']), (36000, 120))
        self.assertTrue(c['check_state_each_step'])

    def test_changed_training_protocol_or_extra_run_is_rejected(self):
        for key, value in [('learning_rate', .001), ('steps', 2000), ('seed', 2),
                           ('ibkd_lambdas', [.25, .5]), ('check_state_each_step', False)]:
            c = self.config(); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): grid.validate_high_beta500(c)
        c = self.config(); c['runs'].append(copy.deepcopy(c['runs'][1]))
        with self.assertRaises(ValueError): grid.validate_high_beta500(c)

    def test_parent_runs_both_and_preserves_milestones_after_one_failure(self):
        c = self.config(); by_id = {p['id']: p for p in c['runs']}
        for fail_first in (False, True):
            def child(command, should_stop):
                self.assertFalse(should_stop())
                self.assertNotIn('--resume', command)
                run_id = command[command.index('--run-id')+1]
                destination = Path(command[command.index('--output-dir')+1]); destination.mkdir()
                failed = fail_first and run_id == 'lg_b2p5'
                (destination/'summary.json').write_text(json.dumps(self.fixture(by_id[run_id], failed)))
                return SimpleNamespace(returncode=1 if failed else 0)
            with self.subTest(fail_first=fail_first), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); capture = io.StringIO()
                argv = ['probe', '--cache-root', str(root/'cache'), '--data-dir', str(root/'data'),
                        '--manifest', str(root/'manifest.json'), '--output-dir', str(root/'out'),
                        '--config', str(grid.CONFIG_DIR/grid.HIGH_BETA500)]
                with patch('sys.argv', argv), patch('ibkd_seg.cityscapes.official_api.bootstrap'), \
                     patch('ibkd_seg.cityscapes.official_assets.verify', return_value={'weights': {}}), \
                     patch('ibkd_seg.cityscapes.full_data.prepare_labels'), \
                     patch('ibkd_seg.cityscapes.tiny_screen2000.launch_child', side_effect=child) as launch, \
                     contextlib.redirect_stdout(capture):
                    if fail_first:
                        with self.assertRaises(SystemExit) as caught: grid.main()
                        self.assertEqual(caught.exception.code, 1)
                    else:
                        grid.main()
                self.assertEqual(launch.call_count, 2)
                line = capture.getvalue().splitlines()[-1]
                self.assertTrue(line.startswith('[CITYSCAPES_TI16_HIGH_BETA500_FINAL] '))
                d = json.loads(line.split('] ', 1)[1])
                self.assertEqual(d['expected_steps'], 500)
                self.assertEqual(d['selection_rule'], 'fixed_500_endpoint_not_best_checkpoint')
                self.assertEqual(d['cross_check_review_count'], 0)
                self.assertEqual(len(d['runs']), 2)
                self.assertEqual(d['runs'][1]['status'], 'passed')
                self.assertEqual(d['runs'][0]['milestones'][0]['ce'], 1.9)
                self.assertEqual(d['runs'][0]['milestones'][-1]['recorded'], not fail_first)
                self.assertEqual(d['runs'][0]['milestones'][-1]['loss'], None if fail_first else 2.)
                self.assertLess(len(line)+1, 50000)
                tail = ('old logs\n'*10000 + line + '\n[3] 완료\n')[-65000:]
                self.assertIn(line, tail)

    def test_setup_failure_final_reports_both_as_not_run(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as capture:
            with patch('sys.argv', ['report', '--output-root', tmp, '--config',
                       str(grid.CONFIG_DIR/grid.HIGH_BETA500), '--exit-code', '73']):
                report_main()
            line = capture.getvalue().strip()
            self.assertTrue(line.startswith('[CITYSCAPES_TI16_HIGH_BETA500_FINAL] '))
            d = json.loads(line.split('] ', 1)[1])
            self.assertEqual(d['not_run_candidates'], ['lg_b2p5', 'ibkd_l025_b2p5'])
            self.assertEqual(d['expected_steps'], 500)
            self.assertEqual(d['pipeline_exit_code'], 73)

    def test_shell_install_failure_never_starts_training_and_keeps_final_json(self):
        repo = grid.CONFIG_DIR.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bindir = root/'bin'; bindir.mkdir()
            fake = bindir/'python'
            fake.write_text('#!/bin/bash\nif [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 73; fi\nexec '+sys.executable+' "$@"\n')
            timer = bindir/'timeout'; timer.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
            for p in (fake, timer): p.chmod(0o755)
            env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                       CITYSCAPES_TI16_OUTPUT=str(root/'out'))
            result = subprocess.run(['bash', str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_high_beta500.sh')],
                                    cwd=repo, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 73, result.stderr)
            d = json.loads(result.stdout.strip().splitlines()[-1].split('] ', 1)[1])
            self.assertEqual(d['expected_steps'], 500)
            self.assertEqual(len(d['not_run_candidates']), 2)
            self.assertFalse((root/'out/manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
