import copy
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ibkd_seg.cityscapes.official_api import student_spec, guidance
from ibkd_seg.cityscapes.official_assets import SMALL_WEIGHT, WEIGHTS, weight_manifest
from ibkd_seg.cityscapes.tiny_smoke import (SMALL_CONFIG, C2VKD_CONFIG, load_config,
                                         beta_candidates, shared_lg_calibration)
from ibkd_seg.cityscapes.small_smoke_report import final_line, MARKER, MAX_FINAL_BYTES
from ibkd_seg.cityscapes.tiny_fskd import TinyFSKD, weighted_components
from ibkd_seg.cityscapes.tiny_c2vkd import TinyC2VKD, FinalFeatureCapture
from ibkd_seg.cityscapes import tiny_smoke


class SmallSmokeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(SMALL_CONFIG)

    def test_common_training_recipe_is_preserved_and_only_small_assets_are_selected(self):
        tiny = load_config(C2VKD_CONFIG)
        changed = {'protocol_id', 'student_backbone', 'encoder_channels', 'beta_multipliers',
                   'fskd', 'c2vkd', 'runs'}
        for key, value in tiny.items():
            if key not in changed:
                self.assertEqual(self.config[key], value, key)
        self.assertEqual([x['method'] for x in self.config['runs']],
                         ['vanilla', 'lg', 'alg', 'ibkd', 'ibkd', 'fskd', 'c2vkd'])
        self.assertEqual([r['lambda'] for r in self.config['runs'] if r['method']=='ibkd'], [.25, .5])
        self.assertEqual(student_spec(self.config['student_backbone']),
                         dict(blocks=12, channels=384, asset_set='small', weight='vit_small_384.npz'))
        assets = weight_manifest('small')
        self.assertEqual(assets['deeplabv3_r101.pth'], WEIGHTS['deeplabv3_r101.pth'])
        self.assertEqual(weight_manifest('small', include_teacher=False), {'vit_small_384.npz':SMALL_WEIGHT})
        for method in ('fskd', 'c2vkd'):
            self.assertEqual(self.config[method]['coefficients'], tiny[method]['coefficients'])
            self.assertFalse(self.config[method]['beta_search'])

    def test_eight_candidate_ratios_keep_three_percent_pilot(self):
        result = beta_candidates([{'ce':4., 'guidance':2.}]*25,
                                 multipliers=self.config['beta_multipliers'], pilot_multiplier=1)
        self.assertEqual(len(result['beta_candidates']), 8)
        self.assertEqual(result['pilot_beta'], .06)
        self.assertEqual(result['beta_candidates'][0], .03)
        for row, ratio in zip(result['measured_ratios'], [.015,.03,.045,.06,.09,.12,.18,.24]):
            self.assertAlmostEqual(row['weighted_guidance_to_seg_loss_median'], ratio)

    def test_shared_lg_beta_is_exact_but_alg_keeps_own_measurement_and_rejects_wrong_identity(self):
        rows = [{'ce':4., 'guidance':2.}]
        own = beta_candidates(rows, multipliers=self.config['beta_multipliers'], pilot_multiplier=1)
        reference = copy.deepcopy(own)
        reference['pilot_beta'] += 1e-9
        reference['beta_candidates'][1] = reference['pilot_beta']
        identity = dict(student='S', adapter='A', teacher='T', inputs=['i'], config_sha256='C', source_sha256='code')
        reference.update(run_id='lg', calibration_identity=identity)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'calibration.json'
            path.write_text(json.dumps(reference))
            result = shared_lg_calibration(path, own, rows, identity)
            self.assertEqual(result['pilot_beta'], reference['pilot_beta'])
            self.assertEqual(result['ce_median'], own['ce_median'])
            for key in identity:
                wrong = dict(identity, **{key:'different'})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs'):
                    shared_lg_calibration(path, own, rows, wrong)

    def test_small_fskd_alignment_and_six_head_attention_backpropagate(self):
        torch.manual_seed(4)
        features = [torch.randn(2,384,2,2,requires_grad=True) for _ in range(12)]
        teacher = [torch.randn(2,c,size,size,requires_grad=True)
                   for c,size in ((256,8),(512,4),(1024,4),(2048,4))]
        queries = torch.randn(2,6,5,requires_grad=True)
        attention = queries.softmax(-1)[:,:,1:]
        logits = torch.randn(2,19,32,32,requires_grad=True)
        tlogits = torch.randn_like(logits,requires_grad=True)
        labels = torch.randint(19,(2,32,32))
        guide = TinyFSKD(crop=32,student_channels=384)
        loss = sum(weighted_components(guide(features,attention,teacher,logits,tlogits,labels)).values())
        loss.backward()
        for index in (0,3):
            self.assertGreater(float(features[index].grad.abs().max()), 0)
        self.assertGreater(float(queries.grad.abs().max()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in guide.parameters()))
        self.assertTrue(all(x.grad is None for x in teacher))

    def test_small_lg_and_ibkd_factory_connect_all_required_blocks(self):
        from ibkd_seg.cityscapes.ibkd_deterministic import apply_deterministic_candidate
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for method, coefficient in [('lg',None),('ibkd',.25),('ibkd',.5)]:
                torch.manual_seed(1)
                guide = guidance(method, self.config)
                features = [torch.randn(1,384,2,2,requires_grad=True) for _ in range(12)]
                teacher = [torch.randn(1,c,2,2) for c in (512,1024,2048)]
                self.assertTrue(all(p.in_channels==384 for p in guide.projections))
                if method == 'ibkd':
                    self.assertTrue(apply_deterministic_candidate(guide)['applied'])
                    alignment, fusion = guide(features,teacher)
                    loss = (1-coefficient)*alignment + coefficient*fusion
                else:
                    loss = guide(features,teacher)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                active = [i for i,f in enumerate(features) if f.grad is not None and f.grad.abs().max()>0]
                self.assertEqual(active,[0,6,11] if method=='lg' else list(range(12)))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in guide.parameters()))
        finally:
            torch.set_num_threads(threads)

    def test_small_c2_capture_and_adapters_accept_width384(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.norm = nn.LayerNorm(384)
            def forward(self, images):
                return self.encoder.norm(images)
        # Hook geometry uses the image spatial shape; five tokens represent 32x32.
        model = Model()
        capture = FinalFeatureCapture(model, student_channels=384)
        tokens = torch.randn(1,5,384,requires_grad=True)
        with patch.object(model, 'forward', side_effect=lambda _:model.encoder.norm(tokens)):
            _, feature = capture.forward(model, torch.zeros(1,3,32,32))
        self.assertEqual(feature.shape, (1,384,2,2))
        with patch('ibkd_seg.cityscapes.tiny_c2vkd.verify_pool_asset', return_value={}), \
             patch('ibkd_seg.cityscapes.tiny_c2vkd.AttentionPool', return_value=nn.Identity()):
            guide = TinyC2VKD('/unused', student_channels=384)
        loss = guide.visual(feature).square().mean() + guide.linguistic(feature.mean((2,3),keepdim=True)).square().mean()
        loss.backward()
        self.assertGreater(float(tokens.grad.abs().max()), 0)
        self.assertEqual(guide.visual.in_channels,384)
        self.assertEqual(guide.linguistic.in_channels,384)
        capture.close()

    def test_terminal_summary_keeps_every_method_beta_and_metric_within_budget(self):
        candidates = beta_candidates([{'ce':4.,'guidance':2.}], multipliers=self.config['beta_multipliers'])
        rows = [dict(run_id=p['id'], status='passed', selected_step=25, selected_epoch=1,
                     last=dict(loss=3.,ce=2.,guidance=1.), calibration=candidates,
                     diagnostic_metrics_percent=dict(pixel_accuracy=10.,miou=2.,
                         class_iou={name:3. for name in ('road','sky','car')})) for p in self.config['runs']]
        line = final_line(dict(status='passed',runs=rows,error='오류'*10000),self.config)
        self.assertLessEqual(len(line.encode())+1,MAX_FINAL_BYTES)
        result = json.loads(line[len(MARKER):])
        self.assertEqual(len(result['runs']),7)
        for row in result['runs']:
            self.assertEqual(row['beta_candidates'],candidates['beta_candidates'])
            self.assertEqual(len(row['class_iou_pct']),19)
            self.assertEqual(row['loss'],3.)
        failure = json.loads(final_line(dict(status='failed',runs=[]),self.config)[len(MARKER):])
        self.assertTrue(all(r['status']=='not_run' and r['miou_pct'] is None for r in failure['runs']))

    def test_parent_selects_small_assets_shares_lg_beta_path_and_keeps_all_children_after_failure(self):
        for fail_alg in (False, True):
            with self.subTest(fail_alg=fail_alg), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                commands = []
                def child(command, check):
                    commands.append(command)
                    run_id = command[command.index('--run-id')+1]
                    if fail_alg and run_id == 'alg':
                        return subprocess.CompletedProcess(command, 1)
                    plan = next(p for p in self.config['runs'] if p['id']==run_id)
                    row = dict(status='passed', run_id=run_id, method=plan['method'],
                               student_initial_state_sha256='S', input_sha256='I', teacher_state_sha256='T',
                               guidance_initial_state_sha256='G',
                               calibration=dict(ce_median=2.,guidance_median=1.,pilot_beta=.06),
                               losses=[dict(step=1,loss=2.06,ce=2.,guidance=1.,grad_norm_unclipped=1.)])
                    destination = Path(command[command.index('--output-dir')+1])
                    destination.mkdir()
                    (destination/'summary.json').write_text(json.dumps(row))
                    return subprocess.CompletedProcess(command, 0)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys,'argv',['smoke','--config',str(SMALL_CONFIG),
                        '--cache-root',str(root/'cache'),'--data-dir',str(root/'data'),
                        '--manifest',str(root/'manifest.json'),'--output-dir',str(root/'out')]))
                    stack.enter_context(patch('ibkd_seg.cityscapes.official_api.bootstrap'))
                    verify = stack.enter_context(patch('ibkd_seg.cityscapes.official_assets.verify',
                                                       return_value={'weights':weight_manifest('small')}))
                    stack.enter_context(patch('ibkd_seg.cityscapes.tiny_c2vkd.verify_pool_asset',return_value={}))
                    stack.enter_context(patch('ibkd_seg.cityscapes.full_data.prepare_labels'))
                    stack.enter_context(patch.object(tiny_smoke.subprocess,'run',side_effect=child))
                    stdout = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    if fail_alg:
                        with self.assertRaises(SystemExit):
                            tiny_smoke.main()
                    else:
                        tiny_smoke.main()
                verify.assert_called_once_with(root/'cache',student='small')
                self.assertEqual(len(commands),7)
                alg = commands[2]
                self.assertEqual(alg[alg.index('--shared-calibration')+1],str(root/'out/lg/calibration.json'))
                self.assertTrue(all('--shared-calibration' not in c for i,c in enumerate(commands) if i!=2))
                report = json.loads(stdout.getvalue().strip().splitlines()[-1][len(MARKER):])
                self.assertEqual(report['status'],'failed' if fail_alg else 'passed')
                self.assertEqual(len(report['runs']),7)
                self.assertEqual(report['runs'][-1]['status'],'passed')

    def test_real_launcher_reports_install_failure_and_does_not_start_training(self):
        repo = SMALL_CONFIG.parents[3]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root/'python'
            python.write_text('#!/bin/bash\nif [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 79; fi\n'
                              f'exec {shlex.quote(sys.executable)} "$@"\n')
            python.chmod(0o755)
            env = dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CITYSCAPES_S16_OUTPUT=str(root/'out'))
            done = subprocess.run(['bash',str(SMALL_CONFIG.parent.parent/'scripts/run_smoke25.sh')],
                                  cwd=repo,env=env,text=True,capture_output=True)
            self.assertEqual(done.returncode,79,done.stdout+done.stderr)
            last = done.stdout.strip().splitlines()[-1]
            self.assertTrue(last.startswith(MARKER))
            result = json.loads(last[len(MARKER):])
            self.assertEqual(result['pipeline_exit_code'],79)
            self.assertEqual(len(result['runs']),7)
            self.assertFalse((root/'out/artifacts').exists())


if __name__ == '__main__':
    unittest.main()
