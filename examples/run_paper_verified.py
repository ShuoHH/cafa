"""Run exact declared research inputs with frozen folds and full DP pricing.

Default: all 13 evaluated series. The main12 subset is reporting only; BIDMC-03
is never silently dropped. Original data and native dependencies are external.
No training, geometry/price reuse, automatic retries or best-repeat selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    '202005_SPlite2', '202001_SPlite2', 'BT', 'BM', 'geolife_074_lon',
    'Kelmarsh_1_power', 'ais_20200801_20200815_366932970_sog',
    'ais_20200801_20200815_366934290_sog', 'ais_20200801_20200815_367690430_sog',
    'CamelMonitor_CPU', 'CamelMonitor_Disk', 'BIDMC_01_PLETH', 'BIDMC_03_PLETH',
)
THREAD_ENV = {
    'OMP_NUM_THREADS':'1', 'NUMBA_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1', 'BLIS_NUM_THREADS':'1', 'NUMEXPR_NUM_THREADS':'1',
    'PYTHONDONTWRITEBYTECODE':'1', 'CLASS_NUMBA_DISK_CACHE':'0', 'MALLOC_ARENA_MAX':'2',
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def repo_file(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('invalid repository artifact path')
    return path

def authenticate(path, digest, size=None):
    if not path.is_file() or sha256(path) != digest or (size is not None and path.stat().st_size != size):
        raise ValueError('SHA/size mismatch: ' + str(path))

def prepare_plan(input_dir=None, input_map=None, series=None, *, root=ROOT):
    """Read-only validation, including weight hashes; never unpickles or writes."""
    root = Path(root).resolve(strict=True)
    models = load(root/'models/release_manifest.json')
    inputs_path = repo_file(root, models['experiment_inputs_path'])
    authenticate(inputs_path, models['experiment_inputs_sha256'])
    inputs = load(inputs_path)['inputs']
    by_name = {r['canonical_name']:r for r in inputs}
    if len(inputs) != 13 or len(by_name) != 13 or set(by_name) != set(CANONICAL):
        raise ValueError('paper protocol requires the original 13-series cohort')
    if sum(bool(r['included_in_main12']) for r in inputs) != 12 or by_name['BIDMC_03_PLETH']['included_in_main12']:
        raise ValueError('main12 is a reporting subset excluding only BIDMC-03')
    selected = list(CANONICAL) if not series else list(series)
    if len(selected) != len(set(selected)) or not set(selected) <= set(CANONICAL):
        raise ValueError('unknown or repeated series selection')
    selected = [s for s in CANONICAL if s in selected]
    if (input_dir is None) == (input_map is None):
        raise ValueError('supply exactly one of input-dir and input-map')
    mapping = None if input_map is None else load(input_map)
    if mapping is not None and (not isinstance(mapping, dict) or not set(mapping) <= set(CANONICAL)
                                or any(not isinstance(p, str) for p in mapping.values())):
        raise ValueError('input-map must map known canonical series names to exact file paths')
    base = None if input_dir is None else Path(input_dir).resolve(strict=True)
    if base is not None and not base.is_dir():
        raise ValueError('input-dir must be a directory')
    map_parent = None if input_map is None else Path(input_map).resolve(strict=True).parent
    artifacts = {r['id']:r for r in models['artifacts']}
    routes = {r['canonical_name']:r for r in models['input_model_routing']}
    expected = {r['series']:r for r in load(root/'evidence/primary13_summary.json')['rows']}
    if set(expected) != set(CANONICAL):
        raise ValueError('preserved comparison summary must retain all 13 results')
    jobs = []
    for name in selected:
        record = by_name[name]
        if mapping is None:
            filename = record['canonical_filename']
            if Path(filename).name != filename:
                raise ValueError('canonical filename must be a plain filename')
            raw = (base / filename).resolve(strict=True)
            if not raw.is_relative_to(base):
                raise ValueError('canonical input escapes input-dir')
        else:
            if name not in mapping:
                raise ValueError('input-map missing selected series: ' + name)
            raw = (map_parent / mapping[name]).resolve(strict=True)
        authenticate(raw, record['sha256'], record['bytes'])
        if record['bytes'] != record['values']*8 or not 4096 <= record['values'] <= 102400:
            raise ValueError('paper input count/format invalid: ' + name)
        route = routes[name]
        artifact = artifacts[record['primary_model_id']]
        if (artifact['variant'] != 'compact20' or artifact['feature_count'] != 20
            or artifact['model_path'] != record['primary_model_path']
            or artifact['metadata_path'] != record['primary_metadata_path']
            or route['primary_model_id'] != artifact['id']
            or route['input_sha256'] != record['sha256'] or route['values'] != record['values']):
            raise ValueError('frozen input/fold route differs: ' + name)
        model = repo_file(root, artifact['model_path'])
        metadata = repo_file(root, artifact['metadata_path'])
        authenticate(model, artifact['model_sha256'], artifact['model_bytes'])
        authenticate(metadata, artifact['metadata_sha256'])
        meta = load(metadata)
        if meta['model_sha256'] != artifact['model_sha256'] or meta['feature_count'] != 20:
            raise ValueError('model metadata differs: ' + name)
        jobs.append(dict(series=name, dataset=record['dataset'], raw=str(raw),
                         values=record['values'], input_sha256=record['sha256'],
                         model=str(model), model_metadata=str(metadata),
                         included_in_main12=record['included_in_main12'],
                         expected={k:expected[name][k] for k in
                                   ('new_primary_bits','final_segments','final_container_sha256')}))
    return dict(version='cacf_strict8_paper_input_driver_v1', jobs=jobs,
                selected_series=len(jobs), full_evaluated_cohort=13,
                main12_is_reporting_only=True, frozen_models_refitted=False,
                boundary_value_budget_multiplier=128, boundary_trial_limit=8192,
                detector=dict(history=4096, subsequence=32, jump=5),
                dp_pricing='full frozen Top-2 graph; auto-funded as in preserved strict8 run',
                fresh_geometry_and_prices=True, paper_timing_claim=False,
                raw_data_and_clean_host_native_build_not_supplied=True)

def command_for(job, out, exporter, fixed_exe, *, root=ROOT):
    return [sys.executable, '-B', str(Path(root)/'examples/run_adaptive_verified.py'),
            '--input', job['raw'], '--dataset', job['dataset'], '--series', job['series'],
            '--model', job['model'], '--model-metadata', job['model_metadata'],
            '--exporter', str(exporter), '--fixed-btr-exe', str(fixed_exe), '--out', str(out),
            '--class-history','4096','--class-timeout','1800','--timeout','300',
            '--boundary-value-budget-multiplier','128','--boundary-trial-limit','8192',
            '--paper-full-pricing']

def compare_result(job, report):
    bits = report.get('final_complete_file_bits')
    required = (report.get('strict8_at_all_stages') is True,
                report.get('bitwise_identical') is True,
                report.get('dp_full_pricing_complete') is True,
                report.get('dp_pricing_mode') == 'paper_full_graph',
                report.get('boundary_resumed') is False,
                report.get('input_sha256') == job['input_sha256'],
                type(bits) is int and bits > 0 and bits % 8 == 0)
    if not all(required):
        raise ValueError('run lacks fresh strict8/full-DP/bitwise proof')
    stored, decoded = Path(report['stored_file']), Path(report['decoded_file'])
    authenticate(stored, report['stored_sha256'], report['final_complete_file_bits']//8)
    authenticate(decoded, job['input_sha256'], job['values']*8)
    authenticate(Path(job['raw']), job['input_sha256'], job['values']*8)
    expected = job['expected']
    matches = dict(complete_file_bits=report['final_complete_file_bits']==expected['new_primary_bits'],
                   final_segments=report['final_segments']==expected['final_segments'],
                   stored_sha256=report['stored_sha256']==expected['final_container_sha256'])
    return dict(series=job['series'], included_in_main12=job['included_in_main12'],
                status='verified', complete_file_bits=report['final_complete_file_bits'],
                final_segments=report['final_segments'], stored_sha256=report['stored_sha256'],
                expected=expected, matches_preserved=matches,
                all_preserved_fields_match=all(matches.values()),
                bitwise_identical=True, full_pricing_complete=True)

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir', type=Path, help='directory with exact canonical filenames')
    source.add_argument('--input-map', type=Path, help='JSON object: canonical series -> exact raw path; relative paths resolve beside map')
    parser.add_argument('--series', action='append', choices=CANONICAL, help='explicit subset; repeat flag; default all13')
    parser.add_argument('--dry-run', action='store_true', help='validate all selected inputs/folds and print plan; write nothing')
    parser.add_argument('--out', type=Path, help='fresh output directory for serial execution')
    parser.add_argument('--exporter', type=Path)
    parser.add_argument('--fixed-btr-exe', type=Path)
    args = parser.parse_args()
    plan = prepare_plan(args.input_dir, args.input_map, args.series)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.out is None or args.exporter is None or args.fixed_btr_exe is None:
        parser.error('actual execution requires --out, --exporter and --fixed-btr-exe')
    out = args.out.resolve()
    if out.exists() or not out.parent.is_dir():
        parser.error('--out must be fresh under an existing parent directory')
    exporter, fixed = args.exporter.resolve(strict=True), args.fixed_btr_exe.resolve(strict=True)
    if not exporter.is_file() or not fixed.is_file():
        parser.error('both native executables must be regular files')
    out.mkdir()
    write(out/'paper_run_plan.json', plan)
    environment = dict(os.environ, **THREAD_ENV)
    rows = []
    failed = None
    for job in plan['jobs']:
        command = command_for(job, out/job['series'], exporter, fixed)
        write(out/(job['series']+'.command.json'), command)
        try:
            with (out/(job['series']+'.stdout')).open('xb') as stdout, (out/(job['series']+'.stderr')).open('xb') as stderr:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            if result.returncode:
                raise RuntimeError('worker exit code ' + str(result.returncode))
            row = compare_result(job, load(out/job['series']/'example_report.json'))
            write(out/(job['series']+'.comparison.json'), row)
            rows.append(row)
        except Exception as exc:
            failed = dict(series=job['series'], exception=type(exc).__name__, detail=str(exc))
            break
    all_selected = len(rows)==len(plan['jobs']) and failed is None
    report = dict(version=plan['version'], selected_series=len(plan['jobs']), completed_series=len(rows),
                  all_selected_verified=all_selected, all13_verified=all_selected and len(rows)==13,
                  all_preserved_results_match=all_selected and all(r['all_preserved_fields_match'] for r in rows),
                  rows=rows, failure=failed, automatic_retry=False, models_refitted=False,
                  paper_timing_claim=False, main12_is_reporting_only=True)
    if all_selected:
        report['selected_total_bits']=sum(r['complete_file_bits'] for r in rows)
        report['selected_main12_members_total_bits']=sum(r['complete_file_bits'] for r in rows if r['included_in_main12'])
    write(out/'paper_run_summary.json', report)
    print(json.dumps(report, indent=2))
    if not all_selected:
        raise SystemExit(1)
    if not report['all_preserved_results_match']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
