import contextlib
import copy
import io
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ibkd_seg.cityscapes import tiny_screen2000 as screen
from ibkd_seg.cityscapes import tiny_grid as grid
from ibkd_seg.cityscapes.tiny_screen2000_report import final_line, MARKER, select_plans
from ibkd_seg.phase1.controllers import GuidanceController
from test_cityscapes_tiny_grid import append_step


class TinyLGALG2000Tests(unittest.TestCase):
    def setUp(self):
        self.config=screen.load_config(screen.LG_ALG_CONFIG)

    def test_locked_interleaved_16_runs_match_500_betas_and_full_val_protocol(self):
        c=self.config
        old=grid.load_config(grid.CONFIG_DIR/grid.GRID_V6)
        betas=[p['beta'] for p in old['runs'] if p['method']=='lg']
        self.assertEqual([p['beta'] for p in c['runs']], [b for b in betas for _ in range(2)])
        self.assertEqual([p['method'] for p in c['runs']], ['lg','alg']*8)
        self.assertEqual([p['run_index'] for p in c['runs']],list(range(1,17)))
        self.assertEqual([p['candidate'] for p in c['runs']],[i for i in range(1,9) for _ in range(2)])
        self.assertTrue(all(p.get('lambda') is None for p in c['runs']))
        self.assertEqual((c['steps'],c['total_steps'],c['validation_samples']),(2000,80000,500))
        self.assertEqual((c['alg_warmup_epochs'],c['alg_earliest_off_step']),(0,745))
        self.assertEqual((c['job_budget_seconds'],c['save_reserve_seconds']),(36000,120))
        self.assertFalse(c['lg_alg_shared_screen'])
        self.assertFalse(c['automatic_next_stage'])
        for field,value in [('learning_rate',.001),('alg_warmup_epochs',20),('runs',c['runs'][:-1])]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/screen.LG_ALG_CONFIG.name
                bad=copy.deepcopy(c);bad[field]=value;p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): screen.load_config(p)

    def test_select_exact_resume_position_without_rerunning_completed_partner(self):
        self.assertEqual([p['id'] for p in select_plans(self.config,start_run=14)],['alg_b7','lg_b8','alg_b8'])
        self.assertEqual([p['id'] for p in select_plans(self.config,start_run=16)],['alg_b8'])
        self.assertEqual([p['id'] for p in select_plans(self.config,start_run=14,run_id='lg_b8')],['lg_b8'])
        for kwargs in [dict(start_candidate=2),dict(start_run=0),dict(start_run=17),dict(start_run=14,run_id='lg_b7')]:
            with self.assertRaises(ValueError): select_plans(self.config,**kwargs)
        with self.assertRaises(ValueError): select_plans(screen.load_config(screen.CONFIG),start_run=1)

    def test_cli_parent_and_child_route_last_alg_run_without_lambda(self):
        from ibkd_seg.cityscapes import official_api, official_assets, full_data
        for child_mode in (False,True):
            with self.subTest(child_mode=child_mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);output=root/'out';plan=self.config['runs'][-1]
                argv=['tiny_screen2000','--config',str(screen.LG_ALG_CONFIG),'--cache-root',str(root/'cache'),
                      '--data-dir',str(root/'data'),'--output-dir',str(output),'--manifest',str(root/'manifest'),
                      '--start-run','16']
                if child_mode: argv+=['--run-id','alg_b8']
                def train(args,config,selected,destination,report):
                    self.assertEqual(selected,plan)
                    self.assertIsNone(report['lambda'])
                    report.update(self.fixture(selected))
                def launch(command,should_stop):
                    self.assertEqual(command[command.index('--run-id')+1],'alg_b8')
                    self.assertEqual(command[command.index('--start-run')+1],'16')
                    dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
                    (dest/'summary.json').write_text(json.dumps(self.fixture(plan)))
                    return SimpleNamespace(returncode=0)
                with patch.object(sys,'argv',argv), patch.object(screen.signal,'signal'), \
                     patch.dict(os.environ,{'CITYSCAPES_TI16_JOB_STARTED':str(time.time())}), \
                     patch.object(official_api,'bootstrap'), patch.object(official_assets,'verify',return_value={'weights':{}}), \
                     patch.object(full_data,'prepare_labels'), patch.object(grid,'run',side_effect=train) as run, \
                     patch.object(screen,'launch_child',side_effect=launch) as spawn, \
                     contextlib.redirect_stdout(io.StringIO()):
                    screen.main()
                self.assertEqual(run.call_count,1 if child_mode else 0)
                self.assertEqual(spawn.call_count,0 if child_mode else 1)
                result=json.loads((output/'terminal_summary.log').read_text()[len(MARKER):])
                self.assertEqual((result['status'],result['pack'],result['expected_runs']),('passed','lg_alg',1))
                self.assertEqual((result['runs'][0]['run_id'],result['runs'][0]['run_index']),('alg_b8',16))
                self.assertEqual(result['runs'][0]['stop_step'],745)

    def test_real_controller_can_stop_alg_at745_while_lg_stays_on(self):
        for method in ('lg','alg'):
            p=grid.initial_progress();controller=GuidanceController(kind=method,beta=.02,warmup_epochs=0)
            for step in range(1,2001): append_step(p,controller,step)
            self.assertEqual((p['epoch'],p['next_batch']),(6,140))
            if method=='alg':
                self.assertEqual(controller.stop_epoch,2)
                self.assertTrue(all(r['beta']==.02 for r in p['rows'][:744]))
                self.assertTrue(all(r['beta']==0 for r in p['rows'][744:]))
            else:
                self.assertIsNone(controller.stop_epoch)
                self.assertTrue(all(r['beta']==.02 for r in p['rows']))

    def fixture(self,plan,status='passed'):
        n=2000 if status=='passed' else 900
        losses=[]
        for step in range(1,n+1):
            on=plan['method']=='lg' or step<745
            losses.append(dict(step=step,beta=plan['beta'] if on else 0.,epoch=(step-1)//372+1,
                               loss=1.5 if on else 1.,ce=1.,guidance=.5 if on else 0.,
                               weighted_guidance=.5*plan['beta'] if on else 0.,
                               weighted_guidance_to_seg_loss=.5*plan['beta'] if on else 0.,
                               grad_norm_unclipped=3. if on else 2.,lr=.01))
        scores=dict(pixel_accuracy=.8,miou=.4,class_iou={k:.4 for k in
                    ('road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation',
                     'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')},
                    evaluated_classes=19,valid_pixels=123)
        passed=status=='passed'
        return dict(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],initial_beta=plan['beta'],
                    **{'lambda':None},status=status,completed_steps=n,attempted_step=n,
                    selected_step=n if passed else None,selected_epoch=6 if passed else None,
                    validation_samples=500 if passed else 0,full_validation=passed,
                    diagnostic_metrics=scores if passed else None,teacher_frozen_verified=True,
                    checkpoint={'strict_state_roundtrip':'passed','saved_step':n,'pointer':'/out/'+plan['id']+'/resume.json'},
                    student_initial_state_sha256='student',teacher_state_sha256='teacher',
                    guidance_initial_state_sha256='guide',input_hashes=[str(i) for i in range(n)],
                    losses=losses,last=losses[-1],trajectory=grid.trajectory_summary(losses),
                    controller={'active':plan['method']=='lg'},guidance_stop_step=745 if plan['method']=='alg' else None,
                    guidance_stop_epoch=2 if plan['method']=='alg' else None)

    def execute(self,root,status=None,start=1,resume=None):
        plans=select_plans(self.config,start_run=start)
        def child(command,should_stop):
            self.assertFalse(should_stop())
            plan=next(p for p in plans if p['id']==command[command.index('--run-id')+1])
            state=status if plan['id']=='alg_b3' and status else 'passed'
            dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
            (dest/'summary.json').write_text(json.dumps(self.fixture(plan,state)))
            return SimpleNamespace(returncode=1 if state=='numerical_failure' else 0)
        args=SimpleNamespace(cache_root=root/'cache',data_dir=root/'data',manifest=root/'manifest',
                             config=screen.LG_ALG_CONFIG,deadline=time.time()+36000,
                             start_candidate=1,start_run=start,resume=resume)
        report=dict(protocol_id=self.config['protocol_id'],pack='lg_alg',runs=[],start_run=start)
        with patch.object(screen,'launch_child',side_effect=child) as launch, contextlib.redirect_stdout(io.StringIO()):
            screen.execute_pack(args,self.config,plans,root,report,lambda:False)
        return report,plans,launch

    def test_all_16_continue_after_candidate_failure_and_report_every_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'numerical_failure')
        self.assertEqual(launch.call_count,16)
        self.assertEqual(report['status'],'needs_review')
        self.assertEqual(report['review_items'],[])
        result=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual((result['pack'],result['expected_runs'],result['configured_pack_runs']),('lg_alg',16,16))
        self.assertEqual(result['failed_candidates'],['alg_b3'])
        self.assertIsNone(result['runs'][5]['miou_pct'])
        self.assertEqual(result['runs'][15]['miou_pct'],40.)
        self.assertEqual(len(result['runs'][15]['class_iou_pct']),19)
        self.assertEqual(result['runs'][15]['stop_step'],745)
        self.assertEqual(result['runs'][15]['run_index'],16)
        self.assertEqual(len(result['lg_alg_active_prefix_pairs']),8)
        self.assertTrue(all(p['matches'] and p['compared_active_steps']==744 for p in result['lg_alg_active_prefix_pairs']))

    def test_pause_retains_unstarted_runs_and_resume_targets_exact_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'paused')
        self.assertEqual(launch.call_count,6)
        self.assertEqual(report['status'],'paused')
        result=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(result['paused_candidates'],['alg_b3'])
        self.assertEqual(len(result['not_run_candidates']),10)
        self.assertIsNone(result['runs'][5]['miou_pct'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report,plans,launch=self.execute(root,start=6,resume=root/'restored/resume.json')
        self.assertEqual(launch.call_count,11)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(plans[0]['id'],'alg_b3')
        self.assertIn('--resume',launch.call_args_list[0].args[0])
        self.assertNotIn('--resume',launch.call_args_list[1].args[0])
        self.assertEqual(report['lg_alg_active_prefix_pairs'][0]['candidate'],4)

    def test_active_prefix_mismatch_is_detected_post_shutdown_difference_is_expected(self):
        pair=[self.fixture(p) for p in self.config['runs'][:2]]
        self.assertTrue(screen.compare_active_lg_alg(pair,self.config)[0]['matches'])
        pair[1]['losses'][499]['loss']+=.1
        actual=screen.compare_active_lg_alg(pair,self.config)[0]
        self.assertFalse(actual['matches'])
        self.assertEqual(actual['first_mismatch']['step'],500)
        pair[1]['input_hashes'][500]='bad-input'
        self.assertIn('same_observed_input_prefixes',screen.identity_checks(pair)[1])

    def test_16_row_final_even_with_failures_fits_last65000_characters(self):
        rows=[self.fixture(p) for p in self.config['runs']]
        for failed in (False,True):
            for r in rows:
                r['warning_summary']=[dict(message='warning'*10000,count=2000)]
                r['checkpoint']['pointer']='x'*10000
                if failed:
                    r.update(status='runtime_failure',error='한글 오류\n'*10000,diagnostic_metrics=None,
                             selected_step=None,selected_epoch=None,full_validation=False,validation_samples=0)
            report=dict(protocol_id=self.config['protocol_id'],pack='lg_alg',status='needs_review' if failed else 'passed',runs=rows)
            line=final_line(report,self.config['runs'])
            self.assertLess(len(line)+1,50000)
            tail=('old log\n'*20000+line+'\n[3] 완료\n')[-65000:]
            self.assertIn(line,tail)
            self.assertEqual(len(json.loads(line[len(MARKER):])['runs']),16)
            print('LG/ALG 16-run final bytes:',len(line))

    def test_install_failure_prints_all_16_and_preserves_job_start(self):
        repo=screen.CONFIG.parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'cityscapes'
            for part in ('leftImg8bit','gtFine'): (data/part/'val').mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'splits':{'train':[{}]*2975,'val':[{}]*500}}))
            python=root/'python'
            python.write_text('#!/bin/bash\nprintf "%s" "$CITYSCAPES_TI16_JOB_STARTED" > "$CLOCK_CAPTURE"\n'
                              'if [[ "$*" == *"-m pip"* ]]; then exit 9; fi\nexec '+shlex.quote(sys.executable)+' "$@"\n')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'),
                     CITYSCAPES_CROP512_DATA_DIR=str(data),CITYSCAPES_TI16_START_RUN='1',
                     CITYSCAPES_TI16_JOB_STARTED='1000',CLOCK_CAPTURE=str(root/'clock.txt'))
            env.pop('CITYSCAPES_TI16_RESUME',None)
            result=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_lg_alg.sh')],
                                  cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,9,result.stderr)
            self.assertEqual((root/'clock.txt').read_text(),'1000')
            decoded=json.loads(result.stdout.strip().splitlines()[-1][len(MARKER):])
            self.assertEqual(decoded['pack'],'lg_alg')
            self.assertEqual(len(decoded['not_run_candidates']),16)
            self.assertEqual(decoded['training_stop_after_seconds'],35880)


if __name__=='__main__': unittest.main()
