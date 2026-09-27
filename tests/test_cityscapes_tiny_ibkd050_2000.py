import contextlib
import copy
import io
import json
import os
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


class TinyIBKD0502000Tests(unittest.TestCase):
    def setUp(self):
        self.config=screen.load_config(screen.IBKD050_CONFIG)

    def test_fixed_original_candidates_and_unchanged_training_evaluation(self):
        c=self.config
        old=grid.load_config(grid.CONFIG_DIR/grid.GRID_V6)
        self.assertEqual(c['runs'],[p for p in old['runs'] if p.get('lambda')==.5 and p['candidate'] in (1,3,6,8)])
        self.assertEqual([p['candidate'] for p in c['runs']],[1,3,6,8])
        self.assertEqual([p['initial_target_ratio'] for p in c['runs']],[.015,.045,.12,.24])
        self.assertEqual(c['ibkd_lambdas'],[.5])
        base=screen.load_config(screen.CONFIG)
        changed={k for k in c if c[k]!=base[k]}
        self.assertEqual(changed,{'protocol_id','pack','runs','ibkd_lambdas','beta_multipliers','continuation_policy'})
        self.assertEqual((c['steps'],c['total_steps'],c['validation_samples'],c['ibkd_warmup_epochs']),(2000,80000,500,20))
        self.assertEqual((c['job_budget_seconds'],c['save_reserve_seconds']),(36000,120))
        for field,value in [('learning_rate',.001),('ibkd_warmup_epochs',0),('runs',c['runs'][:-1])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/screen.IBKD050_CONFIG.name;bad=copy.deepcopy(c);bad[field]=value
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):screen.load_config(p)

    def test_sparse_original_candidate_ids_and_report_inventory(self):
        for start,expected in [(1,[1,3,6,8]),(3,[3,6,8]),(6,[6,8]),(8,[8])]:
            plans=select_plans(self.config,start_candidate=start)
            self.assertEqual([p['candidate'] for p in plans],expected)
            d=json.loads(final_line(dict(protocol_id=self.config['protocol_id'],start_candidate=start),plans)[len(MARKER):])
            self.assertEqual((d['pack'],d['configured_pack_runs'],d['expected_runs']),('ibkd_l050_4betas',4,len(expected)))
            self.assertEqual(d['configured_candidates'],[1,3,6,8])
            self.assertEqual(d['ibkd_warmup_epochs'],20)
            self.assertTrue(all(r['lambda']==.5 and r['miou_pct'] is None for r in d['runs']))
        for kwargs in [dict(start_candidate=2),dict(start_candidate=4),dict(start_candidate=5),dict(start_run=1),
                       dict(start_candidate=6,run_id='ibkd_l050_b3'),dict(run_id='ibkd_l025_b1')]:
            with self.assertRaises(ValueError): select_plans(self.config,**kwargs)

    def fixture(self,plan,status='passed'):
        passed=status=='passed';n=2000 if passed else 777
        scores=dict(pixel_accuracy=.8,miou=.4,class_iou={k:.4 for k in
                    ('road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation',
                     'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')},
                    evaluated_classes=19,valid_pixels=123)
        return dict(run_id=plan['id'],method='ibkd',candidate=plan['candidate'],initial_beta=plan['beta'],
                    **{'lambda':.5},status=status,completed_steps=n,selected_step=2000 if passed else None,
                    selected_epoch=6 if passed else None,full_validation=passed,validation_samples=500 if passed else 0,
                    diagnostic_metrics=scores if passed else None,student_initial_state_sha256='student',
                    teacher_state_sha256='teacher',guidance_initial_state_sha256='guide',
                    input_hashes=[str(i) for i in range(n)],teacher_frozen_verified=True,
                    checkpoint={'strict_state_roundtrip':'passed','saved_step':n,'pointer':'/out/'+plan['id']+'/resume.json'},
                    last=dict(loss=1.,ce=.9,guidance=.1,alignment=.1,fusion=.1,epoch=6,beta=plan['beta']),
                    controller={'active':True},guidance_stop_epoch=None,guidance_stop_step=None)

    def execute(self,root,status='passed',start=1,resume=None):
        plans=select_plans(self.config,start_candidate=start)
        def child(command,should_stop):
            plan=next(p for p in plans if p['id']==command[command.index('--run-id')+1])
            row=self.fixture(plan,status if plan['candidate']==3 else 'passed')
            dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
            (dest/'summary.json').write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if row['status']=='numerical_failure' else 0)
        args=SimpleNamespace(cache_root=root/'cache',data_dir=root/'data',manifest=root/'manifest',
                             config=screen.IBKD050_CONFIG,deadline=time.time()+36000,start_candidate=start,resume=resume)
        report=dict(protocol_id=self.config['protocol_id'],pack=self.config['pack'],runs=[],start_candidate=start)
        with patch.object(screen,'launch_child',side_effect=child) as launch,contextlib.redirect_stdout(io.StringIO()):
            screen.execute_pack(args,self.config,plans,root,report,lambda:False)
        return report,plans,launch

    def test_all_four_execute_and_continue_after_numerical_failure(self):
        for status in ('passed','numerical_failure'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as tmp:
                report,plans,launch=self.execute(Path(tmp),status)
            self.assertEqual(launch.call_count,4)
            self.assertEqual(report['status'],'passed' if status=='passed' else 'needs_review')
            self.assertEqual(report['review_items'],[])
            d=json.loads(final_line(report,plans)[len(MARKER):])
            self.assertEqual([r['candidate'] for r in d['runs']],[1,3,6,8])
            self.assertEqual(d['runs'][-1]['pixel_accuracy_pct'],80.)
            self.assertEqual(d['runs'][-1]['miou_pct'],40.)
            self.assertEqual(len(d['runs'][-1]['class_iou_pct']),19)
            self.assertEqual(d['runs'][-1]['checkpoint_step'],2000)
            self.assertTrue(d['runs'][-1]['guidance_active'])
            if status!='passed':
                self.assertEqual(d['failed_candidates'],['ibkd_l050_b3'])
                self.assertIsNone(d['runs'][1]['miou_pct'])

    def test_pause_then_resume_original_candidate3_only_for_first_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'paused')
        self.assertEqual(launch.call_count,2)
        self.assertEqual(report['status'],'paused')
        d=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(d['paused_candidates'],['ibkd_l050_b3'])
        self.assertEqual(d['not_run_candidates'],['ibkd_l050_b6','ibkd_l050_b8'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,plans,launch=self.execute(root,start=3,resume=root/'restored/resume.json')
        self.assertEqual(launch.call_count,3)
        self.assertEqual([p['candidate'] for p in plans],[3,6,8])
        self.assertIn('--resume',launch.call_args_list[0].args[0])
        self.assertTrue(all('--resume' not in call.args[0] for call in launch.call_args_list[1:]))
        self.assertEqual(report['status'],'passed')

    def test_cli_parent_and_child_keep_lambda050_and_original_candidate8(self):
        from ibkd_seg.cityscapes import official_api,official_assets,full_data
        for child_mode in (False,True):
            with self.subTest(child_mode=child_mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);output=root/'out';plan=self.config['runs'][-1]
                argv=['tiny_screen2000','--config',str(screen.IBKD050_CONFIG),'--cache-root',str(root/'cache'),
                      '--data-dir',str(root/'data'),'--output-dir',str(output),'--manifest',str(root/'manifest'),
                      '--start-candidate','8']
                if child_mode:argv+=['--run-id',plan['id']]
                def train(args,config,selected,destination,report):
                    self.assertEqual(selected,plan);self.assertEqual(report['lambda'],.5)
                    report.update(self.fixture(selected))
                def launch(command,should_stop):
                    self.assertEqual(command[command.index('--run-id')+1],plan['id'])
                    dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
                    (dest/'summary.json').write_text(json.dumps(self.fixture(plan)))
                    return SimpleNamespace(returncode=0)
                with patch.object(sys,'argv',argv),patch.object(screen.signal,'signal'), \
                     patch.dict(os.environ,{'CITYSCAPES_TI16_JOB_STARTED':str(time.time())}), \
                     patch.object(official_api,'bootstrap'),patch.object(official_assets,'verify',return_value={'weights':{}}), \
                     patch.object(full_data,'prepare_labels'),patch.object(grid,'run',side_effect=train) as run, \
                     patch.object(screen,'launch_child',side_effect=launch) as spawn,contextlib.redirect_stdout(io.StringIO()):
                    screen.main()
                self.assertEqual(run.call_count,1 if child_mode else 0)
                self.assertEqual(spawn.call_count,0 if child_mode else 1)
                d=json.loads((output/'terminal_summary.log').read_text()[len(MARKER):])
                self.assertEqual((d['status'],d['pack'],d['configured_pack_runs']),('passed','ibkd_l050_4betas',4))
                self.assertEqual((len(d['runs']),d['runs'][0]['candidate'],d['runs'][0]['lambda']),(1,8,.5))

    def test_final_four_results_fit_tail_with_long_errors(self):
        rows=[self.fixture(p) for p in self.config['runs']]
        for r in rows:
            r.update(status='runtime_failure',error='한글 오류\n'*10000,diagnostic_metrics=None,
                     warning_summary=[dict(count=2000,message='warning'*10000)])
            r['checkpoint']['pointer']='x'*10000
        report=dict(protocol_id=self.config['protocol_id'],status='needs_review',runs=rows)
        line=final_line(report,self.config['runs'])
        self.assertLess(len(line)+1,50000)
        self.assertIn(line,('old log\n'*20000+line+'\n[3] 완료\n')[-65000:])
        self.assertEqual(len(json.loads(line[len(MARKER):])['runs']),4)
        print('iBKD lambda0.5 four-run final bytes:',len(line))

    def test_install_failure_final_reports_only_four_and_preserves_deadline(self):
        repo=screen.CONFIG.parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'cityscapes'
            for part in ('leftImg8bit','gtFine'):(data/part/'val').mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'splits':{'train':[{}]*2975,'val':[{}]*500}}))
            python=root/'python'
            python.write_text('#!/bin/bash\nprintf "%s" "$CITYSCAPES_TI16_JOB_STARTED" > "$CLOCK_CAPTURE"\n'
                              'if [[ "$*" == *"-m pip"* ]]; then exit 9; fi\nexec '+sys.executable+' "$@"\n')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'),
                     CITYSCAPES_CROP512_DATA_DIR=str(data),CITYSCAPES_TI16_START_CANDIDATE='1',
                     CITYSCAPES_TI16_JOB_STARTED='1000',CLOCK_CAPTURE=str(root/'clock.txt'))
            env.pop('CITYSCAPES_TI16_RESUME',None);env.pop('CITYSCAPES_TI16_JOB_DEADLINE',None)
            result=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_ibkd050_4betas.sh')],
                                  cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,9,result.stderr)
            self.assertEqual((root/'clock.txt').read_text(),'1000')
            d=json.loads(result.stdout.strip().splitlines()[-1][len(MARKER):])
            self.assertEqual(d['pack'],'ibkd_l050_4betas')
            self.assertEqual(d['expected_runs'],4)
            self.assertEqual(d['not_run_candidates'],[p['id'] for p in self.config['runs']])
            self.assertEqual(d['training_stop_after_seconds'],35880)


if __name__=='__main__':unittest.main()
