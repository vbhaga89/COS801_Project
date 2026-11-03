"""Reproduce RQ7.1 tables from the three adjacent saved seed directories."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = (14, 27, 42)
IGNORED_CONFIG = {'seed', 'output_dir', 'data_dir', 'manifest_dir', '_config_path'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    assert rows
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(results_root, output):
    signatures, evidence, comparison, calibration, groups, subgroup_records = [], [], [], [], [], []
    keys_by_seed = []
    for seed in SEEDS:
        directory = results_root / f'RQ7_1_Model_Compression_Seed{seed}'
        config = read_json(directory / f'config_seed{seed}.json')
        protocol = read_json(directory / 'split_protocol.json')
        environment = read_json(directory / 'environment.json')
        provenance = read_json(directory / 'source_provenance.json')
        manifest = read_json(directory / 'full_run_manifest.json')
        assert config['seed'] == manifest['seed'] == seed
        assert config['batch_size'] == 64 and config['max_train_batches'] is None and config['max_eval_batches'] is None
        assert protocol['counts'] == {'train': 38147, 'val': 5448, 'test': 10704}
        assert manifest['incomplete_stages'] == ['quantization_mobilenet_v3_small']
        assert next(s for s in manifest['stages'] if s['stage'] == 'quantization_mobilenet_v3_small')['test_evaluated'] is False
        signature = ({k: v for k, v in config.items() if k not in IGNORED_CONFIG}, protocol['files_sha256'], provenance['bundle_sha256'], provenance['inventory']['files_sha256'], environment)
        signatures.append(signature)
        evidence.append(dict(seed=seed, config_hash=manifest['config_hash'], scientific_status=manifest['status'], incomplete_stages=manifest['incomplete_stages'], source_bundle_sha256=provenance['bundle_sha256'], input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.suffix in ('.csv', '.json')}))
        rows = list(csv.DictReader((directory / 'final_comparison.csv').read_text(encoding='utf-8').splitlines()))
        assert len(rows) == 9
        keys = []
        for row in rows:
            assert int(row['seed']) == seed and row['valid_result'] == 'True'
            key = (row['model'], row['variant'], row['teacher_model'])
            assert key not in keys
            keys.append(key)
            comparison.append(dict(seed=seed, model=key[0], variant=key[1], teacher=key[2], compression=row['compression'], accuracy=float(row['accuracy']), macro_f1=float(row['macro_f1']), accuracy_drop_pp=100 * float(row['accuracy_drop']) if row['accuracy_drop'] else None, size_mib=float(row['model_size_mb']), median_latency_ms=float(row['median_latency_ms']), meets_accuracy_budget=row['meets_accuracy_budget']))
        keys_by_seed.append(set(keys))
        evaluated = read_json(directory / 'evaluation_metrics.json')
        assert evaluated['seed'] == seed and len(evaluated['records']) == 9
        evaluated_keys = set()
        for record in evaluated['records']:
            key = (record['model'], record['variant'], record['teacher_model'])
            assert key not in evaluated_keys, 'Duplicate evaluation row'
            evaluated_keys.add(key)
            row = next(r for r in rows if (r['model'], r['variant'], r['teacher_model']) == key)
            assert record['provenance']['artifact'] == row['artifact']
            assert record['provenance']['split_hashes'] == protocol['files_sha256']
            assert record['selection_split'] == 'validation' and record['selection_metric'] == 'NLL'
            best_nll = min(x['validation_nll'] for x in record['validation_search'])
            chosen = record['test_calibrated']['temperature']
            assert any(x['temperature'] == chosen and math.isclose(x['validation_nll'], best_nll, abs_tol=1e-12) for x in record['validation_search'])
            for mode in ['test_uncalibrated', 'test_calibrated']:
                test = record[mode]
                assert test['n'] == 10704 and abs(test['accuracy'] - float(row['accuracy'])) < 1e-12
                calibration.append(dict(seed=seed, model=key[0], variant=key[1], teacher=key[2], mode=mode, n=test['n'], temperature=test['temperature'], ece_top_label=test['ece_top_label'], nll=test['nll'], brier_multiclass_sum=test['brier_multiclass_sum'], aurc_right_endpoint=record['id_selective_risk']['aurc_right_endpoint'] if mode == 'test_uncalibrated' else None))
            test = record['test_uncalibrated']
            subgroup_records.append((seed, key, test['per_class'], test['per_crop']))
            worst_class = test['worst_supported_class_recall']
            worst_crop = test['worst_supported_crop_accuracy']
            groups.append(dict(seed=seed, model=key[0], variant=key[1], teacher=key[2], worst_class=worst_class['name'], worst_class_recall=worst_class['recall'], class_support=worst_class['support'], worst_crop=worst_crop['crop'], worst_crop_accuracy=worst_crop['accuracy'], crop_support=worst_crop['support']))
        assert evaluated_keys == set(keys), 'Evaluation/comparison identity mismatch'
    assert all(s == signatures[0] for s in signatures[1:]), 'Scientific settings, split, source or environment mismatch'
    assert all(k == keys_by_seed[0] for k in keys_by_seed[1:]), 'Missing/extra comparison rows'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'comparison_by_seed.csv', comparison)
    write_csv(output / 'calibration_by_seed.csv', calibration)
    write_csv(output / 'worst_groups_by_seed.csv', groups)
    subgroup_summary = []
    for key in sorted(keys_by_seed[0]):
        selected = [r for r in subgroup_records if r[1] == key]
        classes = {r['name'] for r in selected[0][2]}
        crops = {r['crop'] for r in selected[0][3]}
        assert all(r['support'] > 0 for v in selected for r in v[2] + v[3])
        assert all({r['name'] for r in v[2]} == classes and {r['crop'] for r in v[3]} == crops for v in selected)
        class_values = {name: [next(r for r in v[2] if r['name'] == name) for v in selected] for name in classes}
        crop_values = {name: [next(r for r in v[3] if r['crop'] == name) for v in selected] for name in crops}
        class_name = min(sorted(classes), key=lambda n: statistics.mean(r['recall'] for r in class_values[n]))
        crop_name = min(sorted(crops), key=lambda n: statistics.mean(r['accuracy'] for r in crop_values[n]))
        c = class_values[class_name]
        p = crop_values[crop_name]
        subgroup_summary.append(dict(model=key[0], variant=key[1], teacher=key[2], n=3, worst_mean_class=class_name, class_recall_mean=statistics.mean(r['recall'] for r in c), class_recall_sample_sd=statistics.stdev(r['recall'] for r in c), class_support_each_seed=';'.join(str(r['support']) for r in c), worst_mean_crop=crop_name, crop_accuracy_mean=statistics.mean(r['accuracy'] for r in p), crop_accuracy_sample_sd=statistics.stdev(r['accuracy'] for r in p), crop_support_each_seed=';'.join(str(r['support']) for r in p)))
    write_csv(output / 'subgroup_summary.csv', subgroup_summary)
    summary = []
    for key in sorted(keys_by_seed[0]):
        values = [r for r in comparison if (r['model'], r['variant'], r['teacher']) == key]
        out = dict(model=key[0], variant=key[1], teacher=key[2], n=3, compression_by_seed='; '.join(f"{r['seed']}:{r['compression']}" for r in values))
        for metric in ['accuracy', 'macro_f1', 'accuracy_drop_pp', 'size_mib', 'median_latency_ms']:
            nums = [r[metric] for r in values if r[metric] is not None]
            out[metric + '_n'] = len(nums)
            out[metric + '_mean'] = statistics.mean(nums) if nums else None
            out[metric + '_sample_sd'] = statistics.stdev(nums) if len(nums) > 1 else None
        summary.append(out)
    write_csv(output / 'comparison_summary.csv', summary)
    cal_summary = []
    for key in sorted(keys_by_seed[0]):
        for mode in ['test_uncalibrated', 'test_calibrated']:
            out = dict(model=key[0], variant=key[1], teacher=key[2], mode=mode, n=3)
            selected = [r for r in calibration if (r['model'], r['variant'], r['teacher']) == key and r['mode'] == mode]
            for metric in ['ece_top_label', 'nll', 'brier_multiclass_sum', 'aurc_right_endpoint']:
                nums = [r[metric] for r in selected if r[metric] is not None]
                out[metric + '_n'] = len(nums)
                out[metric + '_mean'] = statistics.mean(nums) if nums else None
                out[metric + '_sample_sd'] = statistics.stdev(nums) if len(nums) > 1 else None
            cal_summary.append(out)
    write_csv(output / 'calibration_summary.csv', cal_summary)
    audit = dict(seeds=list(SEEDS), scientific_settings_match=True, split_hashes_match=True, source_bundle_and_file_hashes_match=True, recorded_environments_match=True, expected_rows_per_seed=9, evaluation_records=27, statuses=evidence, interpretation='Three training seeds on one fixed split, not independent dataset replications. SD is sample SD (ddof=1). MobileNet INT8 missing in all three seeds; no score imputed. Selected pruning is an adaptive validation-selected policy, not a fixed-ratio comparison. CPU latency describes hosted machines, not controlled device performance.')
    (output / 'compatibility_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    labels = {'baseline':'FP32', 'quantized':'INT8 selected', 'pruned':'Pruning selected', 'student_control':'CE student', 'distilled':'KD student'}
    md = ['# RQ7.1 three-seed results', '', 'Seeds **14, 27, 42**. Each value is mean ± sample standard deviation across three matched training seeds on one fixed split. Accuracy is percent; macro F1 is 0–1; size is MiB; latency is hosted-CPU median in ms.', '', '| Backbone / teacher track | Variant | Accuracy (%) | Macro F1 | Size (MiB) | Latency (ms) |', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        track = r['teacher'] or r['model']
        vals = []
        for metric, scale, digits in [('accuracy',100,3), ('macro_f1',1,4), ('size_mib',1,3), ('median_latency_ms',1,3)]:
            vals.append(f"{scale*r[metric+'_mean']:.{digits}f} ± {scale*r[metric+'_sample_sd']:.{digits}f}")
        md.append(f"| {track} | {labels[r['variant']]} | " + ' | '.join(vals) + ' |')
    md += ['', 'MobileNet INT8 failed the validation gate for all three seeds. Its test score is unavailable, not zero. MobileNet pruning selected 20%, 20%, 10% for seeds 14, 27, 42; EfficientNet selected 20% in all three. The table compares the validation-selected procedures.', '', '## Reproduction', '', 'From the RQ7.1 project directory:', '', '```bash', 'python results/three_seed_summary/reproduce_summary.py --results-root results --output reproduced_summary', '```', '', 'The script verifies source-file and bundle fingerprints, scientific configuration, fixed split hashes, recorded environments, row identities and validation-only temperature selection. It reads the original comparison files and compact extracts in the adjacent seed directories. No training, image access or cloud credentials are required.', '', '## Calibration and worst groups', '', '`calibration_summary.csv` reports ECE, NLL and multiclass Brier sum before and after validation-selected temperature scaling. AURC is reported only for the saved uncalibrated ID confidence ranking. `calibration_by_seed.csv` preserves all 54 model/mode records. `worst_groups_by_seed.csv` reports the worst supported class recall and crop accuracy for each of 27 evaluated artifacts, with labels and supports. These are label-derived groups, not sites or devices; the worst-group identity may differ across seeds. Do not treat a mean of per-seed minima as the performance of one fixed group.', '', 'Original evaluation files remain in each linked Kaggle output. Compact evaluation extracts preserve scalar metrics, per-class/per-crop records, validation temperature search and original evaluation SHA256; they omit bulky prediction arrays and risk-curve points. No legacy results or exploratory QAT scores are pooled.', '', '## Limitations', '', 'Three seeds quantify training randomness on one fixed split, not sampling uncertainty. Historical test feedback preceded protocol revision. Missing leaf identities and unscreened near duplicates remain limitations. Hosted CPU timings share the recorded environment but not a verified identical physical machine or load. Smaller models are not necessarily faster. Energy and monetary cost were not measured; no field or target-device deployment validation is claimed. Full checkpoint archives remain external.', '']
    md += ['## Fixed-group worst-case summary', '', '`subgroup_summary.csv` first averages each named class recall and crop accuracy across the same three seeds, then selects the lowest mean group for each model/teacher track. The SD belongs to that same selected group. It is not the average of different worst groups. All declared classes/crops have positive support in these records.', '', '| Track | Variant | Worst class | Recall (%) | Worst crop | Accuracy (%) |', '|---|---|---|---:|---|---:|']
    for r in subgroup_summary:
        md.append(f"| {r['teacher'] or r['model']} | {labels[r['variant']]} | {r['worst_mean_class']} | {100*r['class_recall_mean']:.2f} ± {100*r['class_recall_sample_sd']:.2f} | {r['worst_mean_crop']} | {100*r['crop_accuracy_mean']:.2f} ± {100*r['crop_accuracy_sample_sd']:.2f} |")
    md += ['']
    (output / 'README.md').write_text('\n'.join(md), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.results_root, args.output)
    print(json.dumps({k: result[k] for k in ['seeds', 'scientific_settings_match', 'split_hashes_match', 'source_bundle_and_file_hashes_match', 'recorded_environments_match', 'evaluation_records']}))
