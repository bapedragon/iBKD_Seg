import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from ibkd_seg.cityscapes.tiny_val_timing import (
    CONFIG, CONFIG_DIR, MARKER, evaluate, json_hash, load_config, preflight, sha, student_state, terminal_report,
)


class TinyValTimingTests(unittest.TestCase):
    def fixture(self, root):
        config=load_config(CONFIG)
        training=json.loads((CONFIG_DIR/config['training_config']).read_text())
        data=root/'cityscapes'; data.mkdir()
        for component in ('leftImg8bit','gtFine'): (data/component/'val').mkdir(parents=True)
        manifest=data/'manifest.json'
        manifest.write_text(json.dumps({'splits':{'train':[{}]*2975,'val':[{}]*500}}))
        identity=dict(config_sha256=json_hash(training),plan=training['runs'][0],
                      source_sha256=config['training_source_sha256'],manifest_sha256=sha(manifest))
        last=dict(step=500,loss=1.1,ce=1.,guidance=.1)
        payload=dict(signature=identity,model={'weight':torch.ones(1)},progress=dict(global_step=500,epoch=2,
                     next_batch=128,rows=[{}]*499+[last]))
        run=root/'lg_b1'; (run/'checkpoints').mkdir(parents=True)
        ckpt=run/'checkpoints/resume_000000500.pt'; torch.save(payload,ckpt)
        record=dict(file='checkpoints/'+ckpt.name,bytes=ckpt.stat().st_size,sha256=sha(ckpt),global_step=500,best=None)
        summary=dict(status='passed',run_id='lg_b1',completed_steps=500,selected_step=500,selected_epoch=2,
                     final_student_sha256='a'*64,checkpoint={'current':record},last=last)
        (run/'identity.json').write_text(json.dumps(identity))
        (run/'summary.json').write_text(json.dumps(summary))
        pointer=run/'resume.json'; pointer.write_text(json.dumps(dict(format=1,current=record,previous=None)))
        return config,data,pointer,payload

    def test_verified_endpoint_accepts_model_and_rejects_other_beta_or_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,data,pointer,payload=self.fixture(Path(tmp))
            source=preflight(pointer,data,config)
            torch.testing.assert_close(student_state(payload,source)['weight'],torch.ones(1))
            wrong=copy.deepcopy(payload); wrong['progress']['global_step']=372
            with self.assertRaisesRegex(ValueError,'endpoint'): student_state(wrong,source)
            identity=json.loads((pointer.parent/'identity.json').read_text())
            identity['plan']['beta']*=2
            (pointer.parent/'identity.json').write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError,'candidate'): preflight(pointer,data,config)

    def test_checkpoint_hash_and_path_are_checked_before_loading_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,data,pointer,_=self.fixture(Path(tmp))
            saved=json.loads(pointer.read_text())
            ckpt=pointer.parent/saved['current']['file']
            with ckpt.open('ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA-256'): preflight(pointer,data,config)
            saved['current']['file']='../outside.pt'
            pointer.write_text(json.dumps(saved))
            summary=json.loads((pointer.parent/'summary.json').read_text()); summary['checkpoint']['current']=saved['current']
            (pointer.parent/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'Unsafe'): preflight(pointer,data,config)

    def test_dataset_manifest_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,data,pointer,_=self.fixture(Path(tmp))
            with (data/'manifest.json').open('a') as f: f.write(' ')
            with self.assertRaisesRegex(ValueError,'manifest'): preflight(pointer,data,config)

    def evaluation_fixture(self):
        model=torch.nn.Linear(19,19,bias=False).eval()
        with torch.no_grad(): model.weight.copy_(torch.eye(19))
        target=torch.arange(19)[None,:]
        dataset=[([torch.eye(19)],{},target.clone(),f'val_{i}') for i in range(3)]
        dataset[0][2][0,-1]=255
        config=dict(validation_samples=3,original_target_hw=[1,19],window_size=512,window_stride=512,
                    inference_window_batch_size=1,expected_evaluated_classes=19)
        return model,dataset,config

    def test_eval_only_preserves_weights_and_computes_global_metrics_with_voids(self):
        model,dataset,config=self.evaluation_fixture(); before=model.weight.detach().clone()
        seen=[]
        def inference(m,ims,metas,shape,window,stride,batch_size):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(m.training)
            self.assertEqual((window,stride,batch_size),(512,512,1))
            return m(ims[0]).T[:,None,:]
        result=evaluate(model,dataset,inference,config,lambda:None,lambda n,t:seen.append(n))
        self.assertEqual(result['metrics']['pixel_accuracy'],1.)
        self.assertEqual(result['metrics']['miou'],1.)
        self.assertEqual(result['metrics']['valid_pixels'],56)
        self.assertEqual(result['metrics']['evaluated_classes'],19)
        self.assertEqual(result['validation_samples'],3)
        self.assertEqual(len(result['per_image_timings']),3)
        self.assertGreater(result['validation_seconds'],0)
        self.assertEqual(seen,[1,3])
        torch.testing.assert_close(model.weight,before,rtol=0,atol=0)
        self.assertIsNone(model.weight.grad)

    def test_nonfinite_output_or_duplicate_images_cannot_pass(self):
        model,dataset,config=self.evaluation_fixture()
        def invalid(*args,**kwargs): return torch.full((19,1,19),float('nan'))
        with self.assertRaises(FloatingPointError): evaluate(model,dataset,invalid,config,lambda:None,lambda *_:None)
        dataset[-1]=dataset[0]
        def valid(m,ims,*args,**kwargs): return m(ims[0]).T[:,None,:]
        with self.assertRaisesRegex(ValueError,'Duplicate'): evaluate(model,dataset,valid,config,lambda:None,lambda *_:None)

    def test_final_log_contains_timing_losses_and_all_metrics(self):
        model,dataset,config=self.evaluation_fixture()
        result=evaluate(model,dataset,lambda m,ims,*a,**k:m(ims[0]).T[:,None,:],config,lambda:None,lambda *_:None)
        report=dict(result,status='passed',optimizer_updates=0,teacher_loaded=False,guidance_loaded=False,
                    last_training_loss=1.1,last_training_ce=1.,last_training_guidance=.1,selected_step=500,
                    selected_epoch=2,total_job_seconds=600.,full_validation=True)
        line=terminal_report(report)
        decoded=json.loads(line[len(MARKER):])
        self.assertLess(len(line),65000)
        self.assertEqual(decoded['selected_step'],500)
        self.assertEqual(decoded['selected_epoch'],2)
        self.assertEqual(decoded['last_training_loss'],1.1)
        self.assertEqual(decoded['metrics_percent']['miou'],100.)
        self.assertEqual(len(decoded['metrics_percent']['class_iou']),19)
        self.assertEqual(decoded['total_job_seconds'],600.)
        self.assertGreater(decoded['validation_seconds'],0)

    def test_missing_checkpoint_aborts_shell_before_install_or_training(self):
        repo=CONFIG_DIR.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bindir=root/'bin'; bindir.mkdir()
            fake=bindir/'python'
            fake.write_text('#!/bin/bash\nif [[ "$1" == "-m" && "$2" == "pip" ]]; then touch "'+str(root/'pip_started')+'"; exit 88; fi\nexec /usr/bin/python3 "$@"\n')
            fake.chmod(0o755)
            env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                     CITYSCAPES_TI16_VAL_OUTPUT=str(root/'out'),CITYSCAPES_TI16_EVAL_RESUME=str(root/'missing/resume.json'))
            result=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_val500_timing.sh')],
                                  cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertFalse((root/'pip_started').exists())
            last=result.stdout.strip().splitlines()[-1]
            self.assertTrue(last.startswith(MARKER))
            report=json.loads(last[len(MARKER):])
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['optimizer_updates'],0)
            self.assertIsNone(report['metrics_percent'])
            self.assertIn('checkpoint pointer missing',report['error'])


if __name__=='__main__': unittest.main()
