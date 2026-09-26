import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch

from ibkd_seg.cityscapes import tiny_screen2000 as screen
from ibkd_seg.cityscapes import tiny_grid
from ibkd_seg.cityscapes.tiny_screen2000_report import final_line, MARKER
from ibkd_seg.cityscapes.tiny_val_timing import evaluate, ValidationInterrupted
from ibkd_seg.phase1.controllers import GuidanceController
from test_cityscapes_tiny_grid import append_step


class TinyScreen2000Tests(unittest.TestCase):
    def setUp(self):
        self.config=screen.load_config(screen.CONFIG)

    def test_pack_preserves_all_eight_025_betas_and_training_protocol(self):
        old=tiny_grid.load_config(tiny_grid.CONFIG_DIR/tiny_grid.GRID_V6)
        self.assertEqual(self.config['runs'],[p for p in old['runs'] if p.get('lambda')==.25])
        self.assertEqual(len(self.config['runs']),8)
        self.assertEqual(self.config['steps'],2000)
        self.assertEqual(self.config['total_steps'],80000)
        self.assertEqual(self.config['ibkd_lambdas'],[.25,.5])
        self.assertEqual(self.config['diagnostic_validation_samples'],500)
        self.assertEqual(self.config['job_budget_seconds'],9*3600+45*60)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/screen.CONFIG.name
            bad=copy.deepcopy(self.config); bad['runs'][0]['beta']*=2
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): screen.load_config(p)

    def test_2000_endpoint_is_epoch6_partial_and_ibkd_warmup_is_preserved(self):
        progress=tiny_grid.initial_progress()
        ctrl=GuidanceController(kind='ibkd',beta=.024,warmup_epochs=20)
        for i in range(1,2001): append_step(progress,ctrl,i)
        self.assertEqual((progress['epoch'],progress['next_batch'],progress['samples']),(6,140,1120))
        self.assertEqual(len(progress['completed_epochs']),5)
        self.assertEqual(sum(r['batch_samples'] for r in progress['rows']),15995)
        self.assertIsNone(ctrl.stop_epoch)
        self.assertTrue(all(r['beta']==.024 for r in progress['rows']))

    def fixture_rows(self):
        scores=dict(pixel_accuracy=.8,miou=.4,class_iou={k:.4 for k in
                    ('road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation',
                     'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')},
                    evaluated_classes=19,valid_pixels=123)
        return [dict(run_id=p['id'],method='ibkd',candidate=p['candidate'],initial_beta=p['beta'],
                     **{'lambda':.25},status='passed',completed_steps=2000,selected_step=2000,selected_epoch=6,
                     full_validation=True,validation_samples=500,diagnostic_metrics=scores,
                     student_initial_state_sha256='student',teacher_state_sha256='teacher',
                     guidance_initial_state_sha256='guide',input_hashes=[str(i) for i in range(2000)],
                     teacher_frozen_verified=True,checkpoint={'strict_state_roundtrip':'passed','saved_step':2000},
                     validation_seconds=108.,last=dict(loss=1.,ce=.9,guidance=.1,epoch=6))
                for p in self.config['runs']]

    def pack(self, root, mode, start=1, resume=None):
        rows={r['run_id']:r for r in self.fixture_rows()}
        def child(command, should_stop):
            ident=command[command.index('--run-id')+1]
            output=Path(command[command.index('--output-dir')+1]); output.mkdir()
            row=copy.deepcopy(rows[ident])
            if ident=='ibkd_l025_b2' and mode!='passed':
                row.update(status=mode,completed_steps=777,selected_step=None,selected_epoch=None,
                           full_validation=False,diagnostic_metrics=None,validation_samples=0,
                           input_hashes=row['input_hashes'][:777])
            (output/'summary.json').write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if row['status']=='numerical_failure' else 0)
        args=SimpleNamespace(cache_root=root/'cache',data_dir=root/'data',manifest=root/'manifest.json',
                             config=screen.CONFIG,deadline=time.time()+35100,start_candidate=start,resume=resume)
        plans=self.config['runs'][start-1:]
        report=dict(status='running',protocol_id=self.config['protocol_id'],runs=[],start_candidate=start)
        with patch.object(screen,'launch_child',side_effect=child) as launch,contextlib.redirect_stdout(io.StringIO()):
            screen.execute_pack(args,self.config,plans,root,report,lambda:False)
        return report,plans,launch

    def test_pack_continues_numerical_failure_and_final_log_keeps_all_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.pack(Path(tmp),'numerical_failure')
        self.assertEqual(launch.call_count,8)
        self.assertEqual(report['status'],'needs_review')
        line=final_line(report,plans)
        self.assertLess(len(line),50000)
        decoded=json.loads(line[len(MARKER):])
        self.assertEqual(len(decoded['runs']),8)
        self.assertEqual(decoded['failed_candidates'],['ibkd_l025_b2'])
        self.assertEqual(decoded['runs'][0]['miou_pct'],40.)
        self.assertEqual(len(decoded['runs'][0]['class_iou_pct']),19)
        self.assertIsNone(decoded['runs'][1]['miou_pct'])

    def test_pause_stops_scheduling_and_resume_pointer_only_targets_first_remaining_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.pack(Path(tmp),'paused')
        self.assertEqual(launch.call_count,2)
        self.assertEqual(report['status'],'paused')
        decoded=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(decoded['paused_candidates'],['ibkd_l025_b2'])
        self.assertEqual(len(decoded['not_run_candidates']),6)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pointer=root/'restored/resume.json'
            report,plans,launch=self.pack(root,'passed',start=7,resume=pointer)
        self.assertEqual(launch.call_count,2)
        self.assertIn('--resume',launch.call_args_list[0].args[0])
        self.assertNotIn('--resume',launch.call_args_list[1].args[0])
        self.assertEqual(report['status'],'passed')
        self.assertEqual(len(plans),2)

    def test_expired_budget_starts_no_candidates_and_missing_identity_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=dict(runs=[])
            with patch.object(screen,'launch_child') as launch:
                screen.execute_pack(None,self.config,self.config['runs'],Path(tmp),report,lambda:True)
            launch.assert_not_called()
        self.assertEqual(report['status'],'paused')
        rows=self.fixture_rows()
        rows[2]['input_hashes'][123]='wrong'
        self.assertIn('same_observed_input_prefixes',screen.identity_checks(rows)[1])
        rows=self.fixture_rows(); rows[1].pop('student_initial_state_sha256')
        self.assertIn('initial_identity_present_for_passed',screen.identity_checks(rows)[1])

    def test_parent_stop_is_forwarded_once_to_active_child(self):
        process=MagicMock()
        process.__enter__.return_value=process
        process.wait.side_effect=[subprocess.TimeoutExpired(['test'],1),0]
        with patch.object(screen.subprocess,'Popen',return_value=process):
            result=screen.launch_child(['test'],lambda:True)
        process.send_signal.assert_called_once_with(screen.signal.SIGTERM)
        self.assertEqual(result.returncode,0)

    def test_child_setup_failure_without_metadata_still_has_candidate_in_final_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'summary.json').write_text(json.dumps({'status':'runtime_failure','error':'bootstrap failed'}))
            row=screen.collect_child(root,SimpleNamespace(returncode=1),self.config['runs'][0])
        report=dict(status='needs_review',runs=[row])
        final=json.loads(final_line(report,self.config['runs'])[len(MARKER):])
        self.assertEqual(final['failed_candidates'],['ibkd_l025_b1'])
        self.assertEqual(len(final['not_run_candidates']),7)

    def test_interrupted_validation_never_returns_partial_scores(self):
        target=torch.arange(19)[None,:]
        dataset=[([],{},target,'a'),([],{},target,'b')]
        config=dict(validation_samples=2,original_target_hw=[1,19],window_size=512,window_stride=512,
                    inference_window_batch_size=1,expected_evaluated_classes=19)
        calls=iter([False,True])
        with self.assertRaises(ValidationInterrupted):
            evaluate(None,dataset,lambda *a,**k:torch.eye(19)[:,None,:],config,
                     lambda:None,lambda *a:None,should_stop=lambda:next(calls))

    def test_launcher_install_failure_prints_all_eight_planned_candidates(self):
        repo=screen.CONFIG.parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'cityscapes'
            for part in ('leftImg8bit','gtFine'): (data/part/'val').mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'splits':{'train':[{}]*2975,'val':[{}]*500}}))
            python=root/'python'
            python.write_text('#!/bin/bash\nif [[ "$*" == *"-m pip"* ]]; then exit 9; fi\nexec /usr/bin/python3 "$@"\n')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'),
                     CITYSCAPES_CROP512_DATA_DIR=str(data),CITYSCAPES_TI16_START_CANDIDATE='1')
            env.pop('CITYSCAPES_TI16_RESUME',None)
            result=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_ibkd025.sh')],
                                  cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,9,result.stderr)
            decoded=json.loads(result.stdout.strip().splitlines()[-1][len(MARKER):])
            self.assertEqual(decoded['status'],'runtime_failure')
            self.assertEqual(len(decoded['not_run_candidates']),8)
            self.assertEqual(len(decoded['runs']),8)
            self.assertTrue(all(r['miou_pct'] is None for r in decoded['runs']))


class TinyScreenTrainingTests(unittest.TestCase):
    def test_real_loop_pause_resume_matches_uninterrupted_and_evaluates_only_at_endpoint(self):
        # Execute the shared production training loop/checkpoint code on small CPU modules.
        # No H200 speed or large-model numeric claim is inferred from this test.
        class Student(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.Dropout(.1))
                self.decoder=torch.nn.Conv2d(3,19,1)
            def forward(self,x): return self.decoder(self.encoder(x))
        class Guide(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
            def forward(self,features,teacher):
                z=self.weight.square()*features[0].square().mean()
                return z,z*.5
        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
            def extract_feat(self,x): return [x]*4
        class Capture:
            def __init__(self,model): pass
            def forward(self,model,images):
                features=model.encoder(images)
                return model.decoder(features),[features]
        class Dataset:
            def __init__(self,root,manifest,config,split): self.split=split
            def __len__(self): return 10 if self.split=='train' else 3
            def __getitem__(self,index):
                return [torch.ones(1,3,1,19)],{},torch.arange(19)[None,:],str(index)
        def loader(dataset,epoch,next_batch,device):
            for index in range(next_batch,3):
                count=4 if index<2 else 2
                yield torch.full((count,3,1,19),(epoch+index)/10),torch.arange(19)[None,None,:].repeat(count,1,1),[f'{epoch}_{index}_{i}' for i in range(count)]
        class Scheduler:
            iter_max=80000
            def __init__(self): self.last_epoch=-1
            def step_update(self,n): self.last_epoch=n
            def state_dict(self): return {'last_epoch':self.last_epoch}
            def load_state_dict(self,s): self.last_epoch=s['last_epoch']
        def optimizer(model,guide,*args,**kwargs):
            return torch.optim.SGD(list(model.parameters())+list(guide.parameters()),lr=.001,momentum=.9,nesterov=True),Scheduler()
        # Import optimizer internals before replacing torch.device for the CUDA-only entry point.
        torch.optim.SGD(Student().parameters(),lr=.001)
        config=screen.load_config(screen.CONFIG)
        config.update(steps=8,train_samples=10,batch_size=4,validation_samples=3,original_target_hw=[1,19])
        plan=config['runs'][0]
        modules={n:types.ModuleType(n) for n in ('segm','segm.utils','segm.utils.torch','segm.model','segm.model.utils')}
        modules['segm'].utils=modules['segm.utils']; modules['segm.utils'].torch=modules['segm.utils.torch']
        modules['segm.model.utils'].inference=lambda model,ims,*a,**k:model(ims[0])[0]
        cpu=torch.device('cpu')
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text('{}')
            stack.enter_context(patch.dict(sys.modules,modules))
            stack.enter_context(patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
            stack.enter_context(patch('torch.device',return_value=cpu))
            for name,value in [('is_available',True),('device_count',1),('get_device_name','H200'),
                               ('get_rng_state_all',[]),('set_rng_state_all',None),('manual_seed_all',None),
                               ('synchronize',None),('reset_peak_memory_stats',None),('max_memory_allocated',123)]:
                stack.enter_context(patch('torch.cuda.'+name,return_value=value))
            for name,value in [('student',lambda *a,**k:Student()),('guidance',lambda *a:Guide()),
                               ('teacher',lambda *a:Teacher()),('FeatureCapture',Capture),('optimizer_scheduler',optimizer)]:
                stack.enter_context(patch('ibkd_seg.cityscapes.official_api.'+name,side_effect=value))
            stack.enter_context(patch('ibkd_seg.cityscapes.full_data.FullDataset',Dataset))
            stack.enter_context(patch('ibkd_seg.cityscapes.full_data.train_loader',loader))
            stack.enter_context(patch('ibkd_seg.cityscapes.ibkd_deterministic.apply_deterministic_candidate',return_value={'applied':True}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            def execute(name,resume=None,pause_step=None):
                output=root/name;output.mkdir()
                report={'assets':{},'run_id':plan['id']}
                args=SimpleNamespace(cache_root=root/'cache',data_dir=root,manifest=manifest,resume=resume)
                args.should_stop=lambda:pause_step is not None and report.get('completed_steps',0)>=pause_step
                tiny_grid.run(args,config,plan,output,report)
                pointer=json.loads((output/'resume.json').read_text())
                payload=torch.load(output/pointer['current']['file'],weights_only=True)
                return report,payload,output/'resume.json'
            full,full_state,_=execute('full')
            paused,paused_state,pointer=execute('paused',pause_step=4)
            self.assertEqual(paused['status'],'paused')
            self.assertEqual(paused['completed_steps'],4)
            self.assertIsNone(paused['selected_step'])
            self.assertFalse(paused['full_validation'])
            resumed,resumed_state,_=execute('resumed',resume=pointer)
            self.assertEqual(resumed['status'],'passed')
            self.assertEqual(resumed['selected_step'],8)
            self.assertEqual(resumed['validation_samples'],3)
            self.assertTrue(resumed['student_unchanged_during_validation'])
            self.assertEqual(resumed['input_hashes'],full['input_hashes'])
            for section in ('model','guide'):
                for k,v in full_state[section].items(): torch.testing.assert_close(v,resumed_state[section][k],rtol=0,atol=0)
            self.assertEqual(full_state['scheduler'],resumed_state['scheduler'])
            self.assertEqual(full_state['controller'],resumed_state['controller'])
            self.assertEqual(full['diagnostic_metrics'],resumed['diagnostic_metrics'])
            self.assertEqual(paused_state['progress']['next_batch'],1)
            from ibkd_seg.cityscapes.tiny_smoke import assert_state_close
            assert_state_close(full_state['optimizer'],resumed_state['optimizer'],rtol=0,atol=0)
            endpoint_paused,_,pointer=execute('endpoint_paused',pause_step=8)
            self.assertEqual(endpoint_paused['completed_steps'],8)
            self.assertEqual(endpoint_paused['status'],'paused')
            self.assertFalse(endpoint_paused['full_validation'])
            endpoint_resumed,_,_=execute('endpoint_resumed',resume=pointer)
            self.assertEqual(endpoint_resumed['diagnostic_metrics'],full['diagnostic_metrics'])
            self.assertEqual(endpoint_resumed['final_student_sha256'],full['final_student_sha256'])
            self.assertEqual((root/'endpoint_resumed/steps.jsonl').read_text(),'')


if __name__=='__main__': unittest.main()
