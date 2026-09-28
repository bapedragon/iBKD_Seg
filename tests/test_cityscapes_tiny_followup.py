import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ibkd_seg.cityscapes import tiny_screen2000 as screen,tiny_followup as followup
from ibkd_seg.cityscapes.tiny_grid_report import CLASS_NAMES
from ibkd_seg.cityscapes.tiny_followup_report import MARKER
from ibkd_seg.cityscapes.tiny_screen2000_report import final_line,select_plans


class TinyFollowupTests(unittest.TestCase):
    def setUp(self):
        self.config_path=screen.FOLLOWUP_CONFIG
        self.config=screen.load_config(screen.FOLLOWUP_CONFIG)

    def test_exact_top1_betas_mixed_endpoints_and_common_80k_protocol(self):
        c=self.config
        self.assertEqual([(p['method'],p['beta'],p['target_steps']) for p in c['runs']],
                         [('vanilla',0.,2000),('alg',.11195046919685056,10000),('ibkd',.02461834122158468,10000)])
        self.assertEqual((c['alg_warmup_epochs'],c['ibkd_warmup_epochs']),(0,20))
        self.assertEqual(screen.stopping_deadlines(1000,c),(37000,36880))
        for p in c['runs']:
            effective=followup.effective_config(c,p)
            self.assertEqual(effective['steps'],p['target_steps'])
            self.assertEqual((effective['total_steps'],effective['learning_rate'],effective['selection_metric']),
                             (80000,.01,'miou'))
        for field,value in [('total_steps',10000),('ibkd_warmup_epochs',0),('selection_metric','pixel_accuracy')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/screen.FOLLOWUP_CONFIG.name
                bad=copy.deepcopy(c);bad[field]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):screen.load_config(path)
        self.assertEqual([p['id'] for p in select_plans(c,start_run=2)],['alg_b7','ibkd_l025_b1'])
        for kwargs in [dict(start_candidate=2),dict(start_run=4),dict(start_run=2,run_id='vanilla_2k')]:
            with self.assertRaises(ValueError):select_plans(c,**kwargs)

    def fixture(self,plan,status='passed'):
        passed=status=='passed';vanilla=plan['method']=='vanilla';n=plan['target_steps'] if passed else 1001
        epoch=(n+371)//372
        return dict(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],initial_beta=plan['beta'],
                    **{'lambda':plan.get('lambda')},target_steps=plan['target_steps'],
                    status=status,completed_steps=n,selected_step=n if passed else None,selected_epoch=epoch if passed else None,
                    full_validation=passed,validation_samples=500 if passed else 0,
                    diagnostic_metrics=dict(pixel_accuracy=.8,miou=.4,class_iou={k:.4 for k in CLASS_NAMES},
                                            evaluated_classes=19,valid_pixels=123) if passed else None,
                    student_initial_state_sha256='student',teacher_state_sha256=None if vanilla else 'teacher',
                    guidance_initial_state_sha256=None if vanilla else 'adapter_'+plan['method'],
                    teacher_loaded=not vanilla,guidance_loaded=not vanilla,
                    teacher_frozen_verified=None if vanilla else True,student_unchanged_during_validation=True,
                    input_hashes=[str(i) for i in range(n)],
                    checkpoint={'strict_state_roundtrip':'passed','saved_step':n,'pointer':'/out/'+plan['id']+'/resume.json'},
                    last=dict(loss=.9,ce=.9,guidance=0.,epoch=epoch,beta=0.),
                    controller=None if vanilla else dict(active=False,warmup_epochs=0 if plan['method']=='alg' else 20,
                        threshold=-.02,smoothing_window=50,loss_history=[.2]*20,smoothed_derivative_history=[None]*19+[0.]),
                    guidance_stop_epoch=None if vanilla else 2 if plan['method']=='alg' else 20,
                    guidance_stop_step=None if vanilla else 745 if plan['method']=='alg' else 7441)

    def execute(self,root,status='passed',start=1,resume=None):
        plans=select_plans(self.config,start_run=start)
        def child(command,should_stop):
            plan=next(p for p in plans if p['id']==command[command.index('--run-id')+1])
            row=self.fixture(plan,status if plan['method']=='alg' else 'passed')
            dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
            (dest/'summary.json').write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if row['status']=='numerical_failure' else 0)
        args=SimpleNamespace(cache_root=root/'cache',data_dir=root/'data',manifest=root/'manifest',
                             config=self.config_path,deadline=time.time()+36000,
                             start_candidate=1,start_run=start,resume=resume)
        report=dict(protocol_id=self.config['protocol_id'],pack=self.config['pack'],runs=[],start_run=start)
        with patch.object(screen,'launch_child',side_effect=child) as launch,contextlib.redirect_stdout(io.StringIO()):
            screen.execute_pack(args,self.config,plans,root,report,lambda:False)
        return report,plans,launch

    def test_standalone_vanilla10k_plan_endpoints_ce_only_and_no_extra_runs(self):
        self.config_path=screen.VANILLA_CONFIG
        self.config=screen.load_config(self.config_path)
        c=self.config
        self.assertEqual([(p['id'],p['method'],p['target_steps']) for p in c['runs']],
                         [('vanilla_10k','vanilla',10000)])
        self.assertEqual((c['steps'],c['total_steps'],c['selection_metric']),(10000,80000,'miou'))
        self.assertIsNone(c['beta_initial_ce_ratio']);self.assertFalse(c['ibkd_lambdas'])
        self.assertEqual(screen.stopping_deadlines(1000,c),(37000,36880))
        for kwargs in (dict(start_run=2),dict(start_candidate=2),dict(run_id='alg_b7')):
            with self.assertRaises(ValueError):select_plans(c,**kwargs)
        for field,value in [('total_steps',10000),('steps',2000),('learning_rate',.001),
                            ('runs',self.config['runs']+screen.load_config(screen.FOLLOWUP_CONFIG)['runs'][1:])]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):followup.validate_vanilla_config(bad)
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp))
        self.assertEqual(launch.call_count,1);self.assertEqual(report['status'],'passed')
        self.assertTrue(all(report['identity_checks'].values()))
        line=final_line(report,plans);d=json.loads(line[len(MARKER):])
        self.assertLess(len(line.encode('ascii'))+1,50000)
        self.assertEqual(d['configured_pack_runs'],1)
        self.assertEqual(d['planned_steps'],{'vanilla_10k':10000})
        self.assertEqual(d['completed_runs'],['vanilla_10k'])
        self.assertIsNone(d['ibkd_earliest_possible_off_step'])
        self.assertEqual(d['selection_rule'],'fixed_10000_endpoint_not_best_checkpoint')
        self.assertIn('same training budget',d['score_scope'])
        row=d['runs'][0]
        self.assertEqual((row['selected_step'],row['selected_epoch'],row['miou_pct']),(10000,27,40.))
        self.assertFalse(row['teacher_loaded']);self.assertFalse(row['controller_applicable'])
        self.assertEqual(len(row['class_iou_pct']),19)
        bad=copy.deepcopy(report['runs']);bad[0]['guidance_loaded']=True
        self.assertIn('method_specific_teacher_and_guidance',followup.identity_checks(bad,plans)[1])

    def test_all_three_mixed_identities_and_wrong_endpoint_or_teacher_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp))
        self.assertEqual(launch.call_count,3)
        self.assertEqual(report['status'],'passed')
        self.assertTrue(all(report['identity_checks'].values()))
        for index,field,value in [(1,'completed_steps',2000),(2,'input_hashes',['wrong']*10000),
                                  (0,'teacher_loaded',True),(0,'guidance_initial_state_sha256','unexpected'),
                                  (2,'teacher_state_sha256','wrong')]:
            rows=copy.deepcopy(report['runs']);rows[index][field]=value
            self.assertTrue(followup.identity_checks(rows,plans)[1],field)

    def test_failure_continues_but_pause_stops_and_resume_targets_first_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'numerical_failure')
        self.assertEqual(launch.call_count,3);self.assertEqual(report['status'],'needs_review')
        d=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(d['failed_runs'],['alg_b7']);self.assertEqual(d['runs'][2]['miou_pct'],40.)
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'paused')
        self.assertEqual(launch.call_count,2);self.assertEqual(report['status'],'paused')
        d=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(d['not_run_runs'],['ibkd_l025_b1'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,plans,launch=self.execute(root,start=2,resume=root/'restored/resume.json')
        self.assertIn('--resume',launch.call_args_list[0].args[0])
        self.assertNotIn('--resume',launch.call_args_list[1].args[0])
        self.assertEqual(report['status'],'passed')

    def test_child_cli_uses_2k_for_vanilla_10k_for_kd_without_changing_lr_horizon(self):
        from ibkd_seg.cityscapes import official_api,official_assets,tiny_grid
        cases=[(screen.FOLLOWUP_CONFIG,p) for p in self.config['runs']]
        cases += [(screen.VANILLA_CONFIG,p) for p in screen.load_config(screen.VANILLA_CONFIG)['runs']]
        for config_path,plan in cases:
            with self.subTest(method=plan['method']),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                argv=['tiny_screen2000','--config',str(config_path),'--cache-root',str(root/'cache'),
                      '--data-dir',str(root/'data'),'--output-dir',str(root/'out'),'--manifest',str(root/'manifest'),
                      '--start-run',str(plan['run_index']),'--run-id',plan['id']]
                def train(args,config,selected,destination,report):
                    self.assertEqual(selected,plan);self.assertEqual(config['steps'],plan['target_steps'])
                    self.assertEqual(config['total_steps'],80000)
                    report.update(self.fixture(plan))
                with patch.object(sys,'argv',argv),patch.object(official_api,'bootstrap'),\
                     patch.object(official_assets,'verify',return_value={'weights':{}}),\
                     patch.object(tiny_grid,'run',side_effect=train),contextlib.redirect_stdout(io.StringIO()) as out:
                    screen.main()
                d=json.loads(out.getvalue().strip().splitlines()[-1][len(MARKER):])
                self.assertEqual(d['completed_runs'],[plan['id']])
                self.assertEqual(d['runs'][0]['target_steps'],plan['target_steps'])
                self.assertEqual(json.loads((root/'out/effective_config.json').read_text())['steps'],plan['target_steps'])

    def test_final_tail_keeps_all_scores_endpoints_and_controller_stop_steps(self):
        plans=self.config['runs'];rows=[self.fixture(p) for p in plans]
        for r in rows:r.update(error='긴오류'*200000,losses=[{}]*10000)
        line=final_line(dict(protocol_id=self.config['protocol_id'],status='passed',runs=rows),plans)
        self.assertLess(len(line.encode('ascii'))+1,50000)
        self.assertIn(line,('earlier\n'*100000+line+'\n[3] 완료\n')[-65000:])
        d=json.loads(line[len(MARKER):]);self.assertEqual(d['primary_metric'],'miou')
        self.assertEqual([r['target_steps'] for r in d['runs']],[2000,10000,10000])
        self.assertEqual([r['stop_step'] for r in d['runs']],[None,745,7441])
        self.assertFalse(d['runs'][0]['teacher_loaded']);self.assertFalse(d['runs'][0]['guidance_active'])
        for r in d['runs']:
            self.assertEqual((r['miou_pct'],r['pixel_accuracy_pct']),(40.,80.))
            self.assertEqual(len(r['class_iou_pct']),19)
        print('Mixed 2k/10k final bytes:',len(line.encode('ascii'))+1)

    def test_launcher_install_failure_preserves_deadline_and_all_three_plans(self):
        self.check_launcher_failure('run_followup10k_alg_ibkd025_top1_vanilla2k.sh',
                                    ['vanilla_2k','alg_b7','ibkd_l025_b1'])

    def test_vanilla_launcher_setup_failure_reports_only_unrun_vanilla10k(self):
        self.check_launcher_failure('run_vanilla10k.sh',['vanilla_10k'])

    def check_launcher_failure(self,script,planned_ids):
        repo=screen.CONFIG.parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'cityscapes'
            for part in ('leftImg8bit','gtFine'): (data/part/'val').mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'splits':{'train':[{}]*2975,'val':[{}]*500}}))
            python=root/'python'
            python.write_text('#!/bin/bash\nif [[ "$*" == *"-m pip"* ]]; then exit 9; fi\nexec '+shlex.quote(sys.executable)+' "$@"\n')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'),
                     CITYSCAPES_CROP512_DATA_DIR=str(data),CITYSCAPES_TI16_START_RUN='1',CITYSCAPES_TI16_JOB_STARTED='1000')
            env.pop('CITYSCAPES_TI16_RESUME',None)
            result=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts'/script)],
                                  cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,9,result.stderr)
            d=json.loads(result.stdout.strip().splitlines()[-1][len(MARKER):])
            self.assertEqual(d['not_run_runs'],planned_ids)
            self.assertTrue(all(r['miou_pct'] is None for r in d['runs']))
            self.assertEqual(d['training_stop_after_seconds'],35880)


if __name__=='__main__':unittest.main()
