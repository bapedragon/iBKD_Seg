import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_cityscapes_tiny_followup as helpers
from ibkd_seg.cityscapes import tiny_screen2000 as screen,tiny_ratio2000 as ratio
from ibkd_seg.cityscapes.tiny_screen2000_report import final_line,select_plans


class TinyRatio2000Tests(unittest.TestCase):
    def setUp(self):
        self.config=screen.load_config(screen.ALG_RATIO_CONFIG)
        self.ref=json.loads(ratio.REFERENCE.read_text())

    def fixture(self,plan,status='passed'):
        case=helpers.TinyFollowupTests()
        row=case.fixture(plan,status)
        calibration=self.ref['tiny_calibration']
        row['losses']=[dict(step=1,ce=calibration['ce'],guidance=calibration['guidance'])]
        score=self.ref['l16_candidates'][plan['candidate']-1]['miou_pct']/100
        if status=='passed':row['diagnostic_metrics']['miou']=score
        return row

    def test_locked_four_betas_miou_80k_and_separate_plan_ids(self):
        c=self.config
        self.assertEqual([p['id'] for p in c['runs']],['alg_l16r1','alg_l16r2','alg_l16r3','alg_l16r4'])
        self.assertEqual((c['steps'],c['total_steps'],c['selection_metric'],c['alg_warmup_epochs']),(2000,80000,'miou',0))
        self.assertEqual(screen.stopping_deadlines(1000,c),(37000,36880))
        for p,reference in zip(c['runs'],self.ref['l16_candidates']):
            self.assertEqual(p['initial_target_ratio'],reference['l16_beta']*reference['guidance']/reference['ce'])
            ratio.check_first_step(self.fixture(p)['losses'][0],p,c['first_step_ratio_check'])
        for field,value in [('steps',500),('total_steps',2000),('alg_warmup_epochs',20),('selection_metric','pixel_accuracy')]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):ratio.validate_config(bad)
        with self.assertRaises(ValueError):select_plans(c,start_candidate=2)

    def test_ratio_guard_accepts_rounding_but_rejects_wrong_scale(self):
        p=self.config['runs'][0];first=self.fixture(p)['losses'][0]
        near=dict(first,guidance=first['guidance']*(1+1e-6))
        ratio.check_first_step(near,p,self.config['first_step_ratio_check'])
        for bad in (dict(first,guidance=first['guidance']*1.01),dict(first,ce=0),dict(first,ce=float('nan'))):
            with self.assertRaises(ValueError):ratio.check_first_step(bad,p,self.config['first_step_ratio_check'])

    def test_parent_runs_all_four_and_reports_ranking_and_complete_tail(self):
        case=helpers.TinyFollowupTests();case.setUp()
        case.config_path=screen.ALG_RATIO_CONFIG;case.config=self.config
        with tempfile.TemporaryDirectory() as tmp,patch.object(case,'fixture',side_effect=self.fixture):
            report,plans,launch=case.execute(Path(tmp))
        self.assertEqual(launch.call_count,4);self.assertEqual(report['status'],'passed')
        self.assertTrue(report['identity_checks']['first_step_ratio_matches_l16_target'])
        line=final_line(report,plans);d=json.loads(line[len(ratio.MARKER):])
        self.assertEqual(d['ranking']['tiny'],[2,1,4,3]);self.assertEqual(d['ranking']['spearman'],1)
        self.assertTrue(d['ranking']['full_order_matches'])
        self.assertLess(len(line.encode('ascii'))+1,50000)
        self.assertIn(line,('older\n'*100000+line+'\n[3] 완료\n')[-65000:])
        for row in d['runs']:
            self.assertEqual((row['selected_step'],row['selected_epoch']),(2000,6))
            self.assertTrue(row['first_step_ratio_matches_target'])
            self.assertEqual(len(row['class_iou_pct']),19)
            self.assertEqual(row['initial_ratio_basis'],'l16_first_training_batch_target')
        for r in report['runs']:r['diagnostic_metrics']['miou']=.3
        tied=json.loads(final_line(report,plans)[len(ratio.MARKER):])
        self.assertTrue(tied['ranking']['exact_score_tie']);self.assertIsNone(tied['ranking']['tiny'])
        report['status']='paused';report['runs']=report['runs'][:2]
        partial=json.loads(final_line(report,plans)[len(ratio.MARKER):])
        self.assertFalse(partial['ranking']['complete']);self.assertIsNone(partial['ranking']['suggested_top2_candidates'])
        self.assertEqual(len(partial['not_run_runs']),2)

    def test_incorrect_ratio_cannot_receive_passed_status_or_rank(self):
        case=helpers.TinyFollowupTests();case.setUp();case.config_path=screen.ALG_RATIO_CONFIG;case.config=self.config
        def wrong(plan,status='passed'):
            row=self.fixture(plan,status);row['losses'][0]['guidance']*=2;return row
        with tempfile.TemporaryDirectory() as tmp,patch.object(case,'fixture',side_effect=wrong):
            report,plans,_=case.execute(Path(tmp))
        self.assertEqual(report['status'],'needs_review')
        self.assertIn('first_step_ratio_matches_l16_target',report['review_items'])
        d=json.loads(final_line(report,plans)[len(ratio.MARKER):])
        self.assertFalse(d['ranking']['complete']);self.assertIsNone(d['ranking']['tiny'])

    def test_setup_failure_retains_four_unrun_candidates_and_save_reserve(self):
        case=helpers.TinyFollowupTests();case.setUp()
        with patch.object(helpers,'MARKER',ratio.MARKER):
            case.check_launcher_failure('run_alg_l16_ratio_grid2000.sh',[p['id'] for p in self.config['runs']])

    def test_shared_training_loop_checks_first_ratio_only_for_fresh_runs(self):
        import importlib
        # Keep these real modules registered across the helper's temporary sys.modules patch;
        # otherwise later CLI mocks can target stale package attributes after it restores the dict.
        for name in ('official_api','official_assets'):
            importlib.import_module('ibkd_seg.cityscapes.'+name)
        import test_cityscapes_tiny_screen2000 as training
        # Small CPU modules are not a Tiny calibration. Test guard wiring here;
        # numeric acceptance/rejection of real reference values is tested separately.
        case=training.TinyScreenTrainingTests()
        with patch.object(ratio,'check_first_step',return_value=None) as guard:
            case.check_real_loop(screen.ALG_RATIO_CONFIG,followup_method='alg')
        # Full, mid-run pause, endpoint pause each start fresh; both resumes skip the guard.
        self.assertEqual(guard.call_count,3)


if __name__=='__main__':unittest.main()
