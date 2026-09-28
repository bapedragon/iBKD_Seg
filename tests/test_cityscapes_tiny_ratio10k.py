import contextlib
import copy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_cityscapes_tiny_followup as helpers
from ibkd_seg.cityscapes import tiny_screen2000 as screen,tiny_ratio10k as ratio
from ibkd_seg.cityscapes.tiny_followup_report import MARKER
from ibkd_seg.cityscapes.tiny_screen2000_report import final_line,select_plans


class TinyRatio10kTests(unittest.TestCase):
    def setUp(self):
        self.config=screen.load_config(screen.RATIO10K_CONFIG)

    def fixture(self,plan,status='passed'):
        row=helpers.TinyFollowupTests().fixture(plan,status)
        row['losses']=[dict(step=1,ce=3.79216,guidance=5.93974 if plan['method']=='alg' else 2.25903)]
        return row

    def execute(self,tmp,status='passed',pause_method='alg',start=1,resume=None):
        case=helpers.TinyFollowupTests();case.setUp()
        case.config_path=screen.RATIO10K_CONFIG;case.config=self.config
        def fixture(plan,unused='passed'):
            return self.fixture(plan,status if plan['method']==pause_method else 'passed')
        with patch.object(case,'fixture',side_effect=fixture):
            return case.execute(Path(tmp),start=start,resume=resume)

    def test_sources_new_rank1_exact_ibkd_ratio_and_unchanged_training_protocol(self):
        c=self.config;alg,ibkd=c['runs']
        self.assertEqual((alg['id'],alg['beta']),('alg_l16r3_10k',.08804125116886172))
        self.assertEqual((ibkd['lambda'],ibkd['beta']),(.25,.24648659785043606))
        self.assertAlmostEqual(100*ibkd['initial_target_ratio'],14.683468501911065)
        self.assertEqual((c['steps'],c['total_steps'],c['alg_warmup_epochs'],c['ibkd_warmup_epochs']),
                         (10000,80000,0,20))
        self.assertEqual(screen.stopping_deadlines(1000,c),(37000,36880))
        self.assertEqual([p['id'] for p in select_plans(c,start_run=2)],[ibkd['id']])
        for args in (dict(start_run=3),dict(start_candidate=3),dict(run_id='alg_b7')):
            with self.assertRaises(ValueError):select_plans(c,**args)
        for key,value in [('ibkd_warmup_epochs',0),('total_steps',10000),('steps',2000),
                          ('selection_metric','pixel_accuracy')]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ratio.validate_config(bad)
        for key,value in [('beta',.29542009465901614),('lambda',.5)]:
            bad=copy.deepcopy(c);bad['runs'][1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):ratio.validate_config(bad)

    def test_both_endpoints_identity_ratio_guards_and_complete_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(tmp)
        self.assertEqual(launch.call_count,2);self.assertEqual(report['status'],'passed')
        self.assertTrue(all(report['identity_checks'].values()))
        line=final_line(report,plans);d=json.loads(line[len(MARKER):])
        self.assertEqual((d['protocol_id'],d['pack'],d['configured_pack_runs']),(ratio.PROTOCOL,ratio.PACK,2))
        self.assertEqual(list(d['planned_steps'].values()),[10000,10000])
        self.assertIn(line,('older\n'*100000+line+'\n[3] done\n')[-65000:])
        self.assertLess(len(line.encode())+1,50000)
        for row in d['runs']:
            self.assertEqual((row['selected_step'],row['miou_pct'],len(row['class_iou_pct'])),(10000,40.,19))
            self.assertTrue(row['first_step_ratio_matches_target'])
        self.assertEqual([r['stop_step'] for r in d['runs']],[745,7441])
        self.assertEqual([r['same_beta_tiny2000_completed'] for r in d['runs']],[True,False])
        self.assertEqual(d['selection_rule'],'fixed_10000_endpoint_not_best_checkpoint')
        self.assertFalse(d['automatic_next_stage'])
        bad=copy.deepcopy(report);bad['runs'][1]['losses'][0]['guidance']*=2
        result=json.loads(final_line(bad,plans)[len(MARKER):])
        self.assertFalse(result['runs'][1]['first_step_ratio_matches_target'])
        print('Ratio10k terminal bytes:',len(line.encode())+1)

    def test_parent_wrong_ratio_flags_review_and_missing_first_step_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=helpers.TinyFollowupTests();case.setUp()
            case.config_path=screen.RATIO10K_CONFIG;case.config=self.config
            def bad(plan,status='passed'):
                row=self.fixture(plan,status);row['losses'][0]['guidance']*=2;return row
            with patch.object(case,'fixture',side_effect=bad):
                report,plans,_=case.execute(Path(tmp))
        self.assertEqual(report['status'],'needs_review')
        self.assertIn('first_step_ratio_matches_l16_target',report['review_items'])
        empty=json.loads(final_line(dict(protocol_id=ratio.PROTOCOL,status='runtime_failure'),plans)[len(MARKER):])
        self.assertEqual(len(empty['not_run_runs']),2)
        self.assertTrue(all(r['miou_pct'] is None and r['first_step_ratio_matches_target'] is None for r in empty['runs']))

    def test_failure_pause_and_resume_select_only_remaining_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(tmp,status='numerical_failure')
        self.assertEqual(launch.call_count,2);self.assertEqual(report['status'],'needs_review')
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(tmp,status='paused')
        self.assertEqual(launch.call_count,1);self.assertEqual(report['status'],'paused')
        d=json.loads(final_line(report,plans)[len(MARKER):])
        self.assertEqual(d['not_run_runs'],['ibkd_l025_l16b05_10k'])
        with tempfile.TemporaryDirectory() as tmp:
            report,plans,launch=self.execute(tmp,start=2,resume=Path(tmp)/'restored/resume.json')
        self.assertEqual(launch.call_count,1);self.assertIn('--resume',launch.call_args.args[0])

    def test_child_cli_uses_both_10k_endpoints_and_80k_lr_schedule(self):
        from ibkd_seg.cityscapes import official_api,official_assets,tiny_grid
        for plan in self.config['runs']:
            with self.subTest(method=plan['method']),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                argv=['tiny_screen2000','--config',str(screen.RATIO10K_CONFIG),'--cache-root',str(root/'cache'),
                      '--data-dir',str(root/'data'),'--output-dir',str(root/'out'),'--manifest',str(root/'manifest'),
                      '--start-run',str(plan['run_index']),'--run-id',plan['id']]
                def train(args,config,selected,destination,report):
                    self.assertEqual(config['steps'],10000);self.assertEqual(config['total_steps'],80000)
                    self.assertEqual(selected,plan);report.update(self.fixture(plan))
                with patch.object(sys,'argv',argv),patch.object(official_api,'bootstrap'),\
                     patch.object(official_assets,'verify',return_value={'weights':{}}),\
                     patch.object(tiny_grid,'run',side_effect=train),contextlib.redirect_stdout(io.StringIO()) as out:
                    screen.main()
                d=json.loads(out.getvalue().strip().splitlines()[-1][len(MARKER):])
                self.assertEqual(d['completed_runs'],[plan['id']])

    def test_setup_failure_reports_both_runs_without_installing_dependencies(self):
        case=helpers.TinyFollowupTests();case.setUp()
        case.check_launcher_failure('run_followup10k_alg_top1_ibkd025_l16ratio.sh',
                                    [p['id'] for p in self.config['runs']])

    def test_common_training_controller_off_and_checkpoint_resume_for_both_methods(self):
        for name in ('official_api','official_assets'):
            importlib.import_module('ibkd_seg.cityscapes.'+name)
        from ibkd_seg.cityscapes import tiny_ratio2000
        import test_cityscapes_tiny_screen2000 as training
        for method in ('alg','ibkd'):
            with self.subTest(method=method),patch.object(tiny_ratio2000,'check_first_step',return_value=None) as guard:
                training.TinyScreenTrainingTests().check_real_loop(screen.RATIO10K_CONFIG,followup_method=method)
            self.assertEqual(guard.call_count,3)


if __name__=='__main__':unittest.main()
