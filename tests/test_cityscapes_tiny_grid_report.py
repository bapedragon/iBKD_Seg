import copy
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ibkd_seg.cityscapes.data import CLASS_NAMES as DATA_CLASSES
from ibkd_seg.cityscapes.tiny_grid import CONFIG_DIR, GRID_V6, load_config
from ibkd_seg.cityscapes.tiny_grid_report import CLASS_NAMES, MAX_FINAL_BYTES, TRAJECTORY_KEYS, final_line, main


def filled_report():
    config=load_config(CONFIG_DIR / GRID_V6)
    rows=[]
    for plan in config['runs']:
        row=dict(run_id=plan['id'],status='passed',completed_steps=500,attempted_step=500,
                 selected_step=500,selected_epoch=2,validation_samples=2,
                 last={k:9.12345678912345e+297 for k in ('loss','ce','guidance','alignment','fusion',
                       'weighted_guidance','weighted_guidance_to_seg_loss','grad_norm_unclipped','lr')},
                 diagnostic_metrics_percent=dict(pixel_accuracy=88.123456789,miou=55.123456789,
                     class_iou={name:(None if i==18 else 15.123456789+i) for i,name in enumerate(CLASS_NAMES)},
                     valid_pixels=3_999_999,evaluated_classes=18),
                 trajectory={k:{s:9.12345678912345e+297 for s in ('first25_median','last25_median','maximum','minimum')}
                             for k in TRAJECTORY_KEYS},
                 teacher_frozen_verified=True,controller={'active':True},
                 checkpoint={'saved_step':500,'strict_state_roundtrip':'passed','pointer':'x'*10000},
                 warning_summary=[dict(message='한글 경고\n'*1000,count=500) for _ in range(20)],
                 deterministic_warning_count=500,summary_path='x'*10000,
                 median_step_seconds=1.123456789,train_wall_seconds=99999.123456789,
                 train_peak_allocated_bytes=100_000_000_000)
        row['last'].update(epoch=2,beta=plan['beta'])
        rows.append(row)
    return config,dict(status='passed',protocol_id=config['protocol_id'],lg_alg_shared_screen=True,
                       runs=rows,summary_path='한글/"\n'*10000,
                       cross_checks={'identity_checks':{'same_inputs':True},'review_items':[]})


class TinyGridReportTests(unittest.TestCase):
    def test_complete_metrics_and_all_24_candidates_fit_last_65000_chars(self):
        config,report=filled_report()
        self.assertEqual(CLASS_NAMES,DATA_CLASSES)
        lengths=[]
        for mode in ('success','mixed','all_failed'):
            r=copy.deepcopy(report)
            if mode!='success':
                r['status']='needs_review'
                for i,row in enumerate(r['runs']):
                    if mode=='all_failed' or i%2:
                        row.update(status='numerical_failure',selected_step=None,selected_epoch=None,
                                   error='오류\n"\\'*10000,failure_stage='training')
            line=final_line(r,config['runs']); lengths.append(len(line)+1)
            self.assertEqual(len(line),len(line.encode('utf-8')))
            self.assertLessEqual(len(line)+1,MAX_FINAL_BYTES)
            tail=('old log\n'*20000+line+'\n[3] 완료\n')[-65000:]
            final_text=next(s for s in tail.splitlines() if s.startswith('[CITYSCAPES_TI16_GRID500_FINAL] '))
            result=json.loads(final_text.split('] ',1)[1])
            self.assertEqual(len(result['runs']),24)
            for actual,plan in zip(result['runs'],config['runs'],strict=True):
                self.assertEqual(actual['run_id'],plan['id'])
                self.assertEqual(actual['beta'],plan['beta'])
                self.assertEqual(actual['lambda'],plan.get('lambda'))
                self.assertEqual(len(actual['class_iou_pct']),19)
                self.assertIsNone(actual['class_iou_pct'][-1])
                self.assertAlmostEqual(actual['miou_pct'],55.1235)
                self.assertEqual(actual['loss'],9.12346e297)
                self.assertEqual(actual['warning_count'],10000)
                self.assertEqual(set(actual['trajectory']),set(TRAJECTORY_KEYS))
            self.assertEqual(result['reported_runs'],24)
        print(f'24-run final log bytes (success/mixed/all_failed): {lengths}; cap={MAX_FINAL_BYTES}')

    def test_not_run_candidates_remain_visible_without_invented_metrics(self):
        config,report=filled_report()
        report['runs']=report['runs'][:3]; report['status']='runtime_failure'
        result=json.loads(final_line(report,config['runs']).split('] ',1)[1])
        self.assertEqual(len(result['runs']),24)
        self.assertEqual(len(result['not_run_candidates']),21)
        for row in result['runs'][3:]:
            self.assertEqual(row['status'],'not_run')
            self.assertIsNone(row['loss']); self.assertIsNone(row['miou_pct'])

    def test_exit_fallback_preserves_failed_suite_values_and_budget(self):
        config,report=filled_report()
        report.update(status='needs_review')
        report['runs'][0].update(status='numerical_failure',error='long error '*10000)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'artifacts').mkdir()
            (root/'artifacts/grid_summary.json').write_text(json.dumps(report))
            capture=io.StringIO()
            with patch('sys.argv',['report','--output-root',str(root),'--config',str(CONFIG_DIR/GRID_V6),'--exit-code','1']),contextlib.redirect_stdout(capture):
                main()
            line=capture.getvalue().strip()
            self.assertLessEqual(len(line)+1,MAX_FINAL_BYTES)
            result=json.loads(line.split('] ',1)[1])
            self.assertEqual(result['status'],'needs_review')
            self.assertEqual(result['pipeline_exit_code'],1)
            self.assertEqual(len(result['runs']),24)
            self.assertEqual(result['runs'][0]['status'],'numerical_failure')
            self.assertEqual(result['runs'][-1]['loss'],9.12346e297)

    def test_shell_install_failure_prints_bounded_24_run_final_without_torch(self):
        repo=CONFIG_DIR.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bindir=root/'bin'; bindir.mkdir()
            fake=bindir/'python'
            fake.write_text('#!/bin/bash\nif [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 73; fi\nexec /usr/bin/python3 "$@"\n')
            fake.chmod(0o755)
            env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],CITYSCAPES_TI16_OUTPUT=str(root/'out'))
            p=subprocess.run(['bash',str(repo/'phase4/Cityscapes_Segmenter-Ti16/scripts/run_beta_grid500.sh')],
                             cwd=repo,env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,73,p.stderr)
            line=p.stdout.strip().splitlines()[-1]
            self.assertLessEqual(len(line)+1,MAX_FINAL_BYTES)
            result=json.loads(line.split('] ',1)[1])
            self.assertEqual(result['status'],'runtime_failure')
            self.assertEqual(result['pipeline_exit_code'],73)
            self.assertEqual(len(result['runs']),24)
            self.assertTrue(all(r['status']=='not_run' for r in result['runs']))
            self.assertFalse((root/'out'/'manifest.json').exists())


if __name__=='__main__': unittest.main()
