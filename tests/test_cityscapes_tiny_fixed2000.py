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
from ibkd_seg.cityscapes.tiny_grid_report import CLASS_NAMES
from ibkd_seg.cityscapes.tiny_screen2000_report import final_line, MARKER, select_plans


class TinyFixed2000Tests(unittest.TestCase):
    def setUp(self):
        self.config=screen.load_config(screen.FIXED_CONFIG)

    def test_common_protocol_and_smoke_recipes_are_locked_without_beta_search(self):
        c=self.config
        smoke=json.loads((screen.CONFIG_DIR/'smoke25_fskd_c2vkd_v3.json').read_text())
        self.assertEqual([p['method'] for p in c['runs']],['fskd','c2vkd'])
        for method in ('fskd','c2vkd'):
            self.assertEqual(c[method],smoke[method])
            self.assertFalse(c[method]['beta_search'])
            self.assertIsNone(c[method]['controller'])
        self.assertEqual((c['steps'],c['total_steps'],c['validation_samples']),(2000,80000,500))
        self.assertEqual(c['selection_metric'],'miou')
        self.assertEqual(c['c2vkd']['coefficients']['ce'],0.)
        self.assertEqual(c['fskd']['coefficients']['ce'],1.)
        self.assertEqual(screen.stopping_deadlines(1000,c),(37000,36880))
        for field,value in [('learning_rate',.001),('selection_metric','pixel_accuracy'),('steps',10000)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/screen.FIXED_CONFIG.name
                bad=copy.deepcopy(c);bad[field]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): screen.load_config(path)
        for kwargs in [dict(start_candidate=2),dict(start_run=3),dict(run_id='lg_b1')]:
            with self.assertRaises(ValueError): select_plans(c,**kwargs)
        self.assertEqual(select_plans(c,start_run=2),[c['runs'][1]])

    def fixture(self,plan,status='passed'):
        passed=status=='passed';n=2000 if passed else 777
        method=plan['method'];protocol=self.config[method]
        return dict(run_id=plan['id'],method=method,candidate=None,initial_beta=None,**{'lambda':None},
                    status=status,completed_steps=n,selected_step=2000 if passed else None,
                    selected_epoch=6 if passed else None,full_validation=passed,validation_samples=500 if passed else 0,
                    diagnostic_metrics=dict(pixel_accuracy=.8,miou=.4,class_iou={k:.4 for k in CLASS_NAMES},
                                            evaluated_classes=19,valid_pixels=123) if passed else None,
                    student_initial_state_sha256='student',teacher_state_sha256='teacher',
                    guidance_initial_state_sha256='different_adapter_'+method,
                    input_hashes=[str(i) for i in range(n)],teacher_frozen_verified=True,
                    checkpoint={'strict_state_roundtrip':'passed','saved_step':n,'pointer':'/out/'+plan['id']+'/resume.json'},
                    last=dict(loss=1.,ce=.9,guidance=.1,epoch=6,beta=None,guidance_multiplier=1.,
                              standalone_ce_coefficient=protocol['coefficients']['ce'],components={'global':.01},
                              weighted_components={'global':.1}),controller=None,
                    fixed_loss_coefficients=protocol['coefficients'],method_provenance=protocol,
                    soft_rank_execution={'forward_backward':'passed','device':'cuda'},
                    clip_pool={'asset':self.config['c2vkd']['pool_asset'],'frozen_no_grad_unchanged':True})

    def execute(self,root,status='passed',start=1,resume=None):
        plans=select_plans(self.config,start_run=start)
        def child(command,should_stop):
            plan=next(p for p in plans if p['id']==command[command.index('--run-id')+1])
            row=self.fixture(plan,status if plan['method']=='fskd' else 'passed')
            dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
            (dest/'summary.json').write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if row['status']=='numerical_failure' else 0)
        args=SimpleNamespace(cache_root=root/'cache',data_dir=root/'data',manifest=root/'manifest',
                             config=screen.FIXED_CONFIG,deadline=time.time()+36000,
                             start_candidate=1,start_run=start,resume=resume)
        report=dict(protocol_id=self.config['protocol_id'],pack=self.config['pack'],runs=[],start_run=start)
        with patch.object(screen,'launch_child',side_effect=child) as launch,contextlib.redirect_stdout(io.StringIO()):
            screen.execute_pack(args,self.config,plans,root,report,lambda:False)
        return report,plans,launch

    def test_separate_adapters_allowed_but_student_input_and_recipe_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp))
        self.assertEqual(launch.call_count,2)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['review_items'],[])
        self.assertTrue(report['identity_checks']['same_adapter_within_method'])
        for field,value in [('student_initial_state_sha256','wrong'),('teacher_state_sha256','wrong'),
                            ('input_hashes',['wrong']*2000),('clip_pool',{'frozen_no_grad_unchanged':False})]:
            rows=copy.deepcopy(report['runs']);rows[1][field]=value
            self.assertTrue(screen.identity_checks(rows,fixed_recipes=True)[1],field)

    def test_failure_keeps_other_method_and_pause_stops_with_resume_only_on_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'numerical_failure')
        self.assertEqual(launch.call_count,2)
        self.assertEqual(report['status'],'needs_review')
        result=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(result['failed_candidates'],['fskd'])
        self.assertEqual(result['runs'][1]['miou_pct'],40.)
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),'paused')
        self.assertEqual(launch.call_count,1)
        self.assertEqual(report['status'],'paused')
        result=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(result['not_run_candidates'],['c2vkd_clip_pool'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,plans,launch=self.execute(root,resume=root/'restored/resume.json')
        self.assertIn('--resume',launch.call_args_list[0].args[0])
        self.assertNotIn('--resume',launch.call_args_list[1].args[0])
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(Path(tmp),start=2)
        self.assertEqual(launch.call_count,1)
        self.assertEqual(plans[0]['method'],'c2vkd')

    def test_final_tail_contains_both_metrics_components_provenance_and_failure_inventory(self):
        plans=self.config['runs'];rows=[self.fixture(p) for p in plans]
        for row in rows:
            row.update(error='긴오류'*200000,losses=[{}]*2000)
            row['method_provenance']=dict(row['method_provenance'],interpretation='설명'*200000)
        line=final_line(dict(protocol_id=self.config['protocol_id'],status='passed',runs=rows),plans)
        self.assertLess(len(line.encode('ascii'))+1,50000)
        log=('earlier log\n'*100000+line+'\n[3] 완료\n')[-65000:]
        self.assertIn(line,log)
        result=json.loads(line[len(MARKER):])
        self.assertEqual((result['pack'],result['configured_pack_runs'],result['primary_metric']),('fskd_c2vkd',2,'miou'))
        self.assertFalse(result['automatic_next_stage'])
        for row in result['runs']:
            self.assertEqual((row['miou_pct'],row['pixel_accuracy_pct']),(40.,80.))
            self.assertEqual(len(row['class_iou_pct']),19)
            self.assertIsNone(row['beta'])
            self.assertFalse(row['controller_applicable'])
            self.assertIn('global',row['weighted_loss_components'])
            self.assertIsNotNone(row['source_commit'])

    def test_cli_parent_and_child_follow_exact_fixed_plan(self):
        from ibkd_seg.cityscapes import official_api,official_assets,full_data,tiny_grid
        for child_mode in (False,True):
            with self.subTest(child_mode=child_mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan=self.config['runs'][1]
                argv=['tiny_screen2000','--config',str(screen.FIXED_CONFIG),'--cache-root',str(root/'cache'),
                      '--data-dir',str(root/'data'),'--output-dir',str(root/'out'),
                      '--manifest',str(root/'manifest'),'--start-run','2']
                if child_mode: argv+=['--run-id',plan['id']]
                def train(args,config,selected,destination,report):
                    self.assertEqual(selected,plan);report.update(self.fixture(plan))
                def pack(args,config,plans,destination,report,should_stop):
                    self.assertEqual(plans,[plan]);report.update(status='passed',runs=[self.fixture(plan)])
                with patch.object(sys,'argv',argv),patch.object(official_api,'bootstrap'),\
                     patch.object(official_assets,'verify',return_value={'weights':{}}),\
                     patch.object(full_data,'prepare_labels'),patch.object(tiny_grid,'run',side_effect=train),\
                     patch.object(screen,'execute_pack',side_effect=pack),contextlib.redirect_stdout(io.StringIO()) as out:
                    screen.main()
                result=json.loads(out.getvalue().strip().splitlines()[-1][len(MARKER):])
                self.assertEqual(result['completed_candidates'],['c2vkd_clip_pool'])
                self.assertEqual(result['runs'][0]['standalone_ce_coefficient'],0.)

    def test_setup_failure_still_prints_both_planned_methods_with_null_scores(self):
        repo=screen.CONFIG.parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'cityscapes'
            for part in ('leftImg8bit','gtFine'): (data/part/'val').mkdir(parents=True)
            (data/'manifest.json').write_text(json.dumps({'splits':{'train':[{}]*2975,'val':[{}]*500}}))
            python=root/'python'
            python.write_text('#!/bin/bash\nif [[ "$*" == *"-m pip"* ]]; then exit 9; fi\nexec /usr/bin/python3 "$@"\n')
            python.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'),
                     CITYSCAPES_CROP512_DATA_DIR=str(data),CITYSCAPES_TI16_START_RUN='1',CITYSCAPES_TI16_JOB_STARTED='1000')
            env.pop('CITYSCAPES_TI16_RESUME',None)
            result=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_grid2000_fskd_c2vkd.sh')],
                                  cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,9,result.stderr)
            decoded=json.loads(result.stdout.strip().splitlines()[-1][len(MARKER):])
            self.assertEqual(decoded['status'],'runtime_failure')
            self.assertEqual(decoded['not_run_candidates'],['fskd','c2vkd_clip_pool'])
            self.assertTrue(all(r['miou_pct'] is None for r in decoded['runs']))
            self.assertEqual(decoded['training_stop_after_seconds'],35880)


if __name__=='__main__':
    unittest.main()
