"""Read-only paper routing and explicit full-pricing regression tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_driver_tested', ROOT/'examples/run_paper_verified.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')

class PaperDriverTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base/'repository'
        self.rawdir = self.base/'inputs'
        self.rawdir.mkdir()
        self.artifacts, self.routes, self.inputs, self.expected = [], [], [], []
        for name in driver.CANONICAL:
            raw = self.rawdir/(name+'.bin')
            raw.write_bytes(b'\0'*(4096*8))
            model_rel = 'models/frozen/'+name+'/compact20.joblib'
            metadata_rel = 'models/frozen/'+name+'/compact20.json'
            model = self.repo/model_rel
            model.parent.mkdir(parents=True)
            model.write_bytes(b'not a real pickle; must never be deserialized by plan validation')
            model_sha = driver.sha256(model)
            write(self.repo/metadata_rel, {'feature_count':20,'model_sha256':model_sha})
            artifact = dict(id=name+':compact20',variant='compact20',feature_count=20,
                            model_path=model_rel,metadata_path=metadata_rel,
                            model_sha256=model_sha,model_bytes=model.stat().st_size,
                            metadata_sha256=driver.sha256(self.repo/metadata_rel))
            route = dict(canonical_name=name,primary_model_id=artifact['id'],
                         input_sha256=driver.sha256(raw),values=4096)
            record = dict(canonical_name=name,canonical_filename=raw.name,dataset='fixture',
                          sha256=driver.sha256(raw),values=4096,bytes=raw.stat().st_size,
                          included_in_main12=name!='BIDMC_03_PLETH',
                          primary_model_id=artifact['id'],primary_model_path=model_rel,
                          primary_metadata_path=metadata_rel)
            self.artifacts.append(artifact)
            self.routes.append(route)
            self.inputs.append(record)
            self.expected.append(dict(series=name,new_primary_bits=64,final_segments=1,
                                      final_container_sha256='0'*64))
        self.commit_metadata()

    def commit_metadata(self):
        write(self.repo/'docs/experiment_inputs.json', {'inputs':self.inputs})
        write(self.repo/'models/release_manifest.json', {
            'artifacts':self.artifacts,'input_model_routing':self.routes,
            'experiment_inputs_path':'docs/experiment_inputs.json',
            'experiment_inputs_sha256':driver.sha256(self.repo/'docs/experiment_inputs.json')})
        write(self.repo/'evidence/primary13_summary.json', {'rows':self.expected})

    def prepare(self, **kwargs):
        return driver.prepare_plan(input_dir=self.rawdir,root=self.repo,**kwargs)

    def test_default_plan_retains_all13_without_writing_or_unpickling(self):
        before = {p.relative_to(self.base):driver.sha256(p) for p in self.base.rglob('*') if p.is_file()}
        plan = self.prepare()
        after = {p.relative_to(self.base):driver.sha256(p) for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(plan['selected_series'],13)
        self.assertEqual([x['series'] for x in plan['jobs']],list(driver.CANONICAL))
        self.assertEqual(sum(x['included_in_main12'] for x in plan['jobs']),12)
        self.assertEqual(plan['jobs'][-1]['series'],'BIDMC_03_PLETH')
        self.assertEqual(plan['boundary_value_budget_multiplier'],128)
        self.assertEqual(plan['boundary_trial_limit'],8192)

    def test_explicit_input_map_and_selection_use_exact_file(self):
        mapping = self.base/'input_map.json'
        write(mapping,{'BT':'inputs/BT.bin'})
        plan = driver.prepare_plan(input_map=mapping,series=['BT'],root=self.repo)
        self.assertEqual(plan['selected_series'],1)
        self.assertEqual(plan['jobs'][0]['raw'],str((self.rawdir/'BT.bin').resolve()))
        command = driver.command_for(plan['jobs'][0],self.base/'new','exporter','fixed',root=self.repo)
        self.assertIn('--paper-full-pricing',command)
        self.assertNotIn('--dp-value-budget-multiplier',command)
        self.assertNotIn('--resume-boundary',command)
        self.assertEqual(command[command.index('--dataset')+1],'fixture')

    def test_input_tampering_and_fold_route_mismatch_are_rejected(self):
        raw = self.rawdir/'BT.bin'
        original = raw.read_bytes()
        raw.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA/size mismatch'):
            self.prepare(series=['BT'])
        raw.write_bytes(original)
        self.routes[2]['primary_model_id']=self.artifacts[0]['id']
        self.commit_metadata()
        with self.assertRaisesRegex(ValueError,'fold route differs'):
            self.prepare(series=['BT'])

    def test_wrong_cohort_or_duplicate_requested_series_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'repeated'):
            self.prepare(series=['BT','BT'])
        self.inputs.pop()
        self.commit_metadata()
        with self.assertRaisesRegex(ValueError,'13-series cohort'):
            self.prepare()

    def test_result_comparison_reports_regression_without_inventing_match(self):
        job = self.prepare(series=['BT'])['jobs'][0]
        stored = self.base/'new.f64pool'
        stored.write_bytes(b'new fake stored bytes')
        report = dict(strict8_at_all_stages=True,bitwise_identical=True,
                      dp_full_pricing_complete=True,dp_pricing_mode='paper_full_graph',
                      boundary_resumed=False,input_sha256=job['input_sha256'],
                      stored_file=str(stored),stored_sha256=driver.sha256(stored),
                      decoded_file=job['raw'],final_complete_file_bits=stored.stat().st_size*8,
                      final_segments=3)
        result = driver.compare_result(job,report)
        self.assertFalse(result['all_preserved_fields_match'])
        self.assertFalse(result['matches_preserved']['complete_file_bits'])
        report['dp_full_pricing_complete']=False
        with self.assertRaisesRegex(ValueError,'full-DP'):
            driver.compare_result(job,report)

if __name__ == '__main__':
    unittest.main()
