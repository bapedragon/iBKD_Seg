import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ibkd_seg.phase1.controllers import GuidanceController
from ibkd_seg.cityscapes.full_checkpoint import save_checkpoint, load_checkpoint
from ibkd_seg.cityscapes.runtime import restore_controller
from ibkd_seg.cityscapes.tiny_grid import (BASE_BETA, CONFIG_DIR, load_config, initial_progress,
                                         record_step, compare_grid, main)


def batch_record(step):
    return dict(loss=2., ce=1.5, guidance=10. if step == 372 else 1., weighted_guidance=.1,
                weighted_guidance_to_seg_loss=.1/1.5, grad_norm_unclipped=3., seconds=.1, lr=.01)


def append_step(progress, controller, step):
    if progress['beta'] is None:
        progress['beta'] = controller.beta_for_epoch(progress['epoch'])
    count = 7 if step % 372 == 0 else 8
    return record_step(progress, batch_record(step), count, str(step), controller, train_samples=2975)


class TinyGridTests(unittest.TestCase):
    def test_fixed_sixteen_run_config_keeps_both_lambdas_and_shared_lg_alg_betas(self):
        c = load_config(CONFIG_DIR / 'beta_grid500_both_lambdas_v5.json')
        self.assertEqual(len(c['runs']), 16)
        self.assertEqual(c['steps'], 500)
        self.assertEqual(c['ibkd_lambdas'], [.25, .5])
        self.assertEqual([r['lambda'] for r in c['runs'][8:]], [.25]*4+[.5]*4)
        self.assertFalse(c['record_gradient_hash_each_step'])
        self.assertFalse(c['finite_spike_auto_stop'])
        self.assertEqual([r['beta'] for r in c['runs'][:4]], [r['beta'] for r in c['runs'][4:8]])
        self.assertEqual(c['runs'][0]['beta'], BASE_BETA['lg'])
        self.assertEqual(c['runs'][3]['beta'], BASE_BETA['lg']*8)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'beta_grid500_both_lambdas_v5.json'
            bad=copy.deepcopy(c); bad['runs'][4]['beta']+=1e-12
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError): load_config(p)

    def test_500_updates_cross_epoch_and_do_not_observe_partial_epoch(self):
        for method in ('lg','alg','ibkd'):
            p=initial_progress()
            ctrl=GuidanceController(kind=method,beta=.02,warmup_epochs=20 if method=='ibkd' else 0)
            for step in range(1,501): append_step(p,ctrl,step)
            self.assertEqual(p['global_step'],500)
            self.assertEqual(p['epoch'],2)
            self.assertEqual(p['next_batch'],128)
            self.assertEqual(p['samples'],1024)
            self.assertEqual(sum(r['batch_samples'] for r in p['rows']),3999)
            self.assertEqual(p['rows'][371]['batch_samples'],7)
            self.assertEqual(p['rows'][372]['epoch'],2)
            self.assertEqual(len(ctrl.losses),1)
            self.assertAlmostEqual(ctrl.losses[0],(371*8+7*10)/2975)
            self.assertEqual(ctrl.beta_history,[.02,.02])
            self.assertIsNone(ctrl.stop_epoch)

    def test_partial_epoch_checkpoint_restores_sums_controller_and_rng(self):
        # A lightweight training trajectory checks real SGD momentum and dropout RNG,
        # with a pause immediately before crossing the 372-update epoch boundary.
        def setup():
            torch.manual_seed(5)
            model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.2),torch.nn.Linear(3,1))
            opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,nesterov=True)
            ctrl=GuidanceController(kind='alg',beta=.02)
            return model,opt,ctrl,initial_progress()
        def update(model,opt,ctrl,p,step):
            x=torch.ones(2,2)*(step%7)/7
            opt.zero_grad(); loss=model(x).square().mean(); loss.backward(); opt.step()
            append_step(p,ctrl,step)
        model,opt,ctrl,p=setup()
        for step in range(1,371): update(model,opt,ctrl,p,step)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); identity={'test':'partial_epoch'}
            save_checkpoint(root,dict(signature=identity,progress=p,model=model.state_dict(),optimizer=opt.state_dict(),
                                      controller=ctrl.state_dict(),torch_rng=torch.get_rng_state()))
            for step in range(371,501): update(model,opt,ctrl,p,step)
            expected=copy.deepcopy(model.state_dict()); expected_p=copy.deepcopy(p)
            model,opt,ctrl,p=setup()
            saved,_=load_checkpoint(root/'resume.json',identity,root)
            model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
            restore_controller(ctrl,saved['controller']); p=saved['progress']; torch.set_rng_state(saved['torch_rng'])
            for step in range(371,501): update(model,opt,ctrl,p,step)
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            self.assertEqual(p,expected_p)
            self.assertEqual(len(ctrl.losses),1)

    def fixture(self):
        config=load_config(CONFIG_DIR/'beta_grid500_both_lambdas_v5.json')
        rows=[]
        for plan in config['runs']:
            losses=[dict(batch_record(i),step=i,beta=plan['beta']) for i in range(1,501)]
            rows.append(dict(run_id=plan['id'],method=plan['method'],candidate=plan['candidate'],initial_beta=plan['beta'],
                             status='passed',completed_steps=500,last=losses[-1],losses=losses,
                             input_hashes=[str(i) for i in range(1,501)],student_initial_state_sha256='student',
                             teacher_state_sha256='teacher',guidance_initial_state_sha256='ibkd' if plan['method']=='ibkd' else 'lg'))
        return config,rows

    def test_grid_detects_input_or_adapter_drift_and_reports_all_beta_pairs(self):
        c,rows=self.fixture()
        self.assertEqual(compare_grid(rows,c)['review_items'],[])
        rows[4]['losses'][3]['guidance']+=.1
        r=compare_grid(rows,c)
        self.assertEqual(len(r['lg_alg_pairs']),4)
        self.assertEqual(r['lg_alg_pairs'][0]['first_scalar_mismatch']['step'],4)
        rows[8]['input_hashes'][0]='different'; rows[9]['guidance_initial_state_sha256']='wrong'
        r=compare_grid(rows,c)
        self.assertIn('input_prefixes',r['review_items'])
        self.assertIn('ibkd_same_adapter_initialization',r['review_items'])

    def test_parent_continues_all_sixteen_after_numerical_failure_and_prints_results(self):
        c,rows=self.fixture(); by_id={r['run_id']:r for r in rows}
        def fake_child(command,check):
            run_id=command[command.index('--run-id')+1]
            out=Path(command[command.index('--output-dir')+1]); out.mkdir(parents=True)
            row=copy.deepcopy(by_id[run_id])
            failed=run_id=='lg_b2'
            if failed:
                row.update(status='numerical_failure',completed_steps=3,selected_step=None,selected_epoch=None,
                           losses=row['losses'][:3],input_hashes=row['input_hashes'][:3],
                           error='FloatingPointError: nonfinite gradient')
            (out/'summary.json').write_text(json.dumps(row))
            return SimpleNamespace(returncode=1 if failed else 0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); capture=io.StringIO()
            argv=['tiny_grid','--cache-root',str(root/'cache'),'--data-dir',str(root/'data'),
                  '--manifest',str(root/'manifest.json'),'--output-dir',str(root/'out'),
                  '--config',str(CONFIG_DIR/'beta_grid500_both_lambdas_v5.json')]
            with patch('sys.argv',argv),patch('ibkd_seg.cityscapes.official_api.bootstrap'), \
                 patch('ibkd_seg.cityscapes.official_assets.verify',return_value={'weights':{}}), \
                 patch('ibkd_seg.cityscapes.full_data.prepare_labels'), \
                 patch('subprocess.run',side_effect=fake_child) as child,contextlib.redirect_stdout(capture):
                with self.assertRaises(SystemExit) as error: main()
                self.assertEqual(error.exception.code,1)
            self.assertEqual(child.call_count,16)
            final=json.loads(capture.getvalue().strip().splitlines()[-1].split('] ',1)[1])
            self.assertEqual(final['status'],'needs_review')
            self.assertEqual(len(final['runs']),16)
            self.assertEqual(final['failed_candidates'],['lg_b2'])
            self.assertEqual(len(final['finite_candidates']),15)
            self.assertFalse(final['beta_ranking_performed'])
            self.assertFalse(final['automatic_next_stage'])


if __name__=='__main__': unittest.main()
