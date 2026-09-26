import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import torch

from ibkd_seg.cityscapes import tiny_val_timing as timing
from ibkd_seg.cityscapes.tiny_val_initial import preflight_initial, prepare_initial_data, initial_student
from ibkd_seg.cityscapes.official_assets import weight_manifest, TINY_WEIGHT
from ibkd_seg.cityscapes.runtime import seed_all, state_hash


class TinyInitialTimingTests(unittest.TestCase):
    def setUp(self):
        self.config = timing.load_config(timing.INITIAL_CONFIG)

    def data_fixture(self, root):
        data = root / 'cityscapes'
        for component in ('leftImg8bit', 'gtFine'):
            (data / component / 'val').mkdir(parents=True)
        (data / 'manifest.json').write_text(json.dumps({'splits':{'train':[{}]*2975, 'val':[{}]*500}}))
        return data

    def test_initial_config_matches_existing_evaluation_geometry(self):
        old = timing.load_config(timing.CONFIG)
        for key in ('validation_samples', 'original_target_hw', 'window_size', 'window_stride',
                    'inference_window_batch_size', 'precision', 'cpu_threads', 'data_workers',
                    'evaluation_passes', 'warmup_images'):
            self.assertEqual(self.config[key], old[key])
        training = json.loads((timing.CONFIG_DIR / old['training_config']).read_text())
        for key in ('val_samples', 'image_size', 'crop_size', 'seed'):
            self.assertEqual(self.config['dataset_config'][key], training[key])
        self.assertEqual(weight_manifest('tiny', include_teacher=False), {'vit_tiny_384.npz':TINY_WEIGHT})
        self.assertIn('deeplabv3_r101.pth', weight_manifest('tiny'))

    def test_existing_data_works_without_archives_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.data_fixture(root)
            source=preflight_initial(data, root/'absent_zips', self.config)
            self.assertFalse(source['needs_prepare'])
            with patch('ibkd_seg.cityscapes.tiny_val_initial.subprocess.run') as run:
                ready=prepare_initial_data(source, self.config, root)
            run.assert_not_called()
            self.assertEqual(ready['manifest_sha256'], timing.sha(data/'manifest.json'))
            self.assertNotIn('checkpoint', ready)

    def test_fresh_data_requires_verified_zips_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); zips=root/'zips'; zips.mkdir()
            archive=zips/'sample.zip'; archive.write_bytes(b'small fixture')
            expected={'sample.zip':dict(bytes=archive.stat().st_size,sha256=timing.sha(archive))}
            with patch('ibkd_seg.cityscapes.tiny_val_initial.ARCHIVES', expected):
                source=preflight_initial(root/'cityscapes', zips, self.config)
                self.assertTrue(source['needs_prepare'])
                archive.write_bytes(b'wrong fixture')  # Same byte count, different content.
                with patch('ibkd_seg.cityscapes.tiny_val_initial.subprocess.run') as run:
                    with self.assertRaisesRegex(ValueError, 'SHA-256'):
                        prepare_initial_data(source, self.config, root)
                    run.assert_not_called()
                archive.write_bytes(b'small fixture')
                with patch('ibkd_seg.cityscapes.tiny_val_initial.subprocess.run',
                           side_effect=lambda *a,**k:self.data_fixture(root)) as run:
                    ready=prepare_initial_data(source, self.config, root)
                self.assertFalse(ready['needs_prepare'])
                self.assertIn('ibkd_seg.cityscapes.prepare', run.call_args.args[0])
            with self.assertRaisesRegex(FileNotFoundError, 'ZIP missing'):
                preflight_initial(root/'other'/'cityscapes', root/'missing', self.config)

    def test_initial_student_never_loads_checkpoint_and_freezes_seeded_model(self):
        seen=[]
        def factory(root, **kwargs):
            seen.append(kwargs)
            return torch.nn.Linear(3, 19)
        with patch('torch.load', side_effect=AssertionError('Must not load a trained checkpoint')):
            first, first_hash=initial_student(Path('/unused'), self.config, factory, seed_all, state_hash)
            second, second_hash=initial_student(Path('/unused'), self.config, factory, seed_all, state_hash)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(seen[0], dict(backbone='vit_tiny_patch16_384', image_size=512, decoder_layers=1, recompute=True))
        self.assertFalse(first.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in first.parameters()))

    def test_v2_main_runs_evaluation_without_checkpoint_teacher_or_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.data_fixture(root)
            ptu=types.ModuleType('segm.utils.torch')
            modules={name:types.ModuleType(name) for name in ('segm','segm.utils','segm.model','segm.model.utils')}
            modules['segm.utils.torch']=ptu
            modules['segm'].utils=modules['segm.utils']
            modules['segm.utils'].torch=ptu
            modules['segm.model.utils'].inference=Mock()
            scores=dict(pixel_accuracy=.1, miou=.01, class_iou={str(i):.01 for i in range(19)},
                        evaluated_classes=19, valid_pixels=123)
            result=dict(metrics=scores, validation_samples=500, validation_seconds=123.4,
                        per_image_timings=[], validation_ids=[])
            args=['timing','--config',str(timing.INITIAL_CONFIG),'--data-dir',str(data),
                  '--cache-root',str(root/'cache'),'--output-dir',str(root/'out')]
            cpu=torch.device('cpu')
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules,modules))
                stack.enter_context(patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
                stack.enter_context(patch.object(sys,'argv',args))
                stack.enter_context(patch('ibkd_seg.cityscapes.official_api.bootstrap'))
                verify=stack.enter_context(patch('ibkd_seg.cityscapes.official_assets.verify',
                                  return_value={'weights':weight_manifest('tiny',include_teacher=False)}))
                stack.enter_context(patch('ibkd_seg.cityscapes.official_api.student',side_effect=lambda *a,**k:torch.nn.Linear(3,19)))
                stack.enter_context(patch('ibkd_seg.cityscapes.official_api.teacher',side_effect=AssertionError('No teacher')))
                stack.enter_context(patch('ibkd_seg.cityscapes.official_api.optimizer_scheduler',side_effect=AssertionError('No optimizer')))
                stack.enter_context(patch('torch.load',side_effect=AssertionError('No checkpoint')))
                for name,value in [('is_available',True),('device_count',1),('get_device_name','H200'),
                                   ('max_memory_allocated',123),('reset_peak_memory_stats',None),('synchronize',None)]:
                    stack.enter_context(patch('torch.cuda.'+name,return_value=value))
                stack.enter_context(patch('torch.device',return_value=cpu))
                stack.enter_context(patch('ibkd_seg.cityscapes.real_smoke.validate_manifest'))
                stack.enter_context(patch('ibkd_seg.cityscapes.data.verify_manifest'))
                stack.enter_context(patch('ibkd_seg.cityscapes.full_data.FullDataset',return_value=[]))
                evaluate=stack.enter_context(patch.object(timing,'evaluate',return_value=result))
                stdout=stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                timing.main()
            verify.assert_called_once_with(root/'cache',student='tiny',include_teacher=False)
            evaluate.assert_called_once()
            report=json.loads(stdout.getvalue().strip().splitlines()[-1][len(timing.MARKER):])
            self.assertEqual(report['status'],'passed')
            self.assertEqual(report['validation_samples'],500)
            self.assertEqual(report['validation_seconds'],123.4)
            self.assertEqual(report['optimizer_updates'],0)
            self.assertEqual(report['selected_step'],0)
            self.assertTrue(report['student_state_unchanged'])
            self.assertFalse(report['trained_checkpoint_loaded'])
            self.assertIsNone(report['checkpoint'])
            self.assertIsNone(report['last_training_loss'])
            self.assertEqual(len(report['metrics_percent']['class_iou']),19)
            self.assertIn('untrained decoder diagnostic',report['score_scope'])

    def test_launcher_passes_preflight_without_resume_and_reports_install_failure(self):
        repo=timing.CONFIG_DIR.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=self.data_fixture(root)
            python=root/'python'
            python.write_text('#!/bin/bash\nif [[ "$*" == *"-m pip"* ]]; then exit 7; fi\nexec /usr/bin/python3 "$@"\n')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],
                     CITYSCAPES_TI16_VAL_OUTPUT=str(root/'out'),CITYSCAPES_CROP512_DATA_DIR=str(data),
                     CITYSCAPES_TI16_EVAL_RESUME=str(root/'absent/resume.json'))
            run=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_val500_timing_initial.sh')],
                               cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,7,run.stderr)
            self.assertIn('checkpoint=none',run.stdout)
            line=run.stdout.strip().splitlines()[-1]
            self.assertTrue(line.startswith(timing.MARKER))
            self.assertLess(len(line),65000)
            report=json.loads(line[len(timing.MARKER):])
            self.assertEqual(report['protocol_id'],self.config['protocol_id'])
            self.assertEqual(report['pipeline_exit_code'],7)
            self.assertEqual(report['optimizer_updates'],0)
            self.assertFalse(report['trained_checkpoint_loaded'])
            self.assertIsNone(report['metrics_percent'])
            self.assertIn('untrained decoder diagnostic',report['score_scope'])


if __name__ == '__main__':
    unittest.main()
