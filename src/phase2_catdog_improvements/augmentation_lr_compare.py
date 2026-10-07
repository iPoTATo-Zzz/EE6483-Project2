"""Controlled augmented-input learning-rate experiment, using the original D2 engine."""
import argparse
import csv
import json
import subprocess
import sys
from src.common.utils import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if not args.run_name or any(c in args.run_name for c in '/\\:') or args.run_name in {'.', '..'}:
        parser.error('run-name must be a single directory name')
    stage = 'phase2_catdog_improvements'
    config = ROOT / 'configs' / stage / 'augmentation_lr0003.json'
    reference = ROOT / 'runs' / stage / 'augmentation_full_20261006_001'
    original = json.loads((reference / 'config.json').read_text(encoding='utf-8'))
    candidate = json.loads(config.read_text(encoding='utf-8'))
    assert set(original) == set(candidate)
    assert [k for k in original if original[k] != candidate[k]] == ['learning_rate']
    assert original['learning_rate'] == 0.001 and candidate['learning_rate'] == 0.0003
    pipeline = ROOT / 'runs' / stage / ('pipeline_' + args.run_name)
    pipeline.mkdir(parents=True, exist_ok=False)
    run = ROOT / 'runs' / stage / args.run_name
    result = ROOT / 'results' / stage / args.run_name
    checkpoint = run / 'best.pth'
    commands = [
        ('training', ['src.phase2_catdog_improvements.train', '--config', str(config), '--run-name', args.run_name]),
        ('evaluation', ['src.phase1_catdog_baseline.evaluate', '--checkpoint', str(checkpoint), '--output', str(result)]),
        ('prediction', ['src.phase1_catdog_baseline.predict', '--checkpoint', str(checkpoint), '--output', str(ROOT / 'submission' / (args.run_name + '.csv'))]),
    ]
    current = 'setup'
    try:
        for current, arguments in commands:
            save_json(pipeline / 'status.json', {'state': 'running', 'stage': current, 'run_name': args.run_name})
            print('Starting ' + current, flush=True)
            with (pipeline / (current + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', '-m', *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        current = 'verification'
        read = lambda p: json.loads(p.read_text(encoding='utf-8'))
        assert read(run / 'data_manifest.json') == read(reference / 'data_manifest.json')
        rows = list(csv.DictReader((run / 'history.csv').open()))
        assert [int(x['epoch']) for x in rows] == list(range(1, 11))
        metrics = read(result / 'metrics.json')
        best = read(run / 'best_metrics.json')
        chosen = max(rows, key=lambda x: (float(x['val_accuracy']), -float(x['val_loss'])))
        assert metrics['epoch'] == best['epoch'] == int(chosen['epoch'])
        assert metrics['accuracy'] == best['accuracy'] == float(chosen['val_accuracy'])
        assert abs(metrics['loss'] - best['loss']) < 1e-9
        before = read(ROOT / 'results' / stage / 'augmentation_full_20261006_001' / 'predictions.json')
        after = read(result / 'predictions.json')
        assert [(x['path'], x['label']) for x in before] == [(x['path'], x['label']) for x in after]
        assert sum(x['label'] == x['prediction'] for x in after) / 5000 == metrics['accuracy']
        fixed = [b for a, b in zip(before, after) if a['prediction'] != a['label'] and b['prediction'] == b['label']]
        new = [b for a, b in zip(before, after) if a['prediction'] == a['label'] and b['prediction'] != b['label']]
        persistent = [b for a, b in zip(before, after) if a['prediction'] != a['label'] and b['prediction'] != b['label']]
        for name, items in [('fixed', fixed), ('introduced', new), ('persistent', persistent)]:
            with (result / (name + '_errors.csv')).open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'label', 'prediction'])
                writer.writeheader()
                writer.writerows(items)
        save_json(result / 'augmentation_lr_comparison.json', {'reference_run': 'augmentation_full_20261006_001', 'reference_metrics': read(ROOT / 'results' / stage / 'augmentation_full_20261006_001' / 'metrics.json'), 'candidate_metrics': metrics, 'fixed_count': len(fixed), 'introduced_count': len(new), 'persistent_count': len(persistent), 'only_config_change': 'learning_rate', 'limitations': 'Single seed 42; best validation checkpoints; no labeled test accuracy.'})
        with (ROOT / 'submission' / (args.run_name + '.csv')).open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ['id', 'label']
            predictions = list(reader)
        assert len(predictions) == 500
        assert [int(x['id']) for x in predictions] == list(range(1, 501))
        assert all(x['label'] in ['0', '1'] for x in predictions)
        status = {'state': 'complete', 'run_name': args.run_name, 'only_learning_rate_changed': True, 'history_verified': True, 'original_image_reload_verified': True, 'csv_verified': True}
        save_json(result / 'status.json', status)
        save_json(pipeline / 'status.json', status)
        print(json.dumps(status), flush=True)
    except Exception as error:
        save_json(pipeline / 'status.json', {'state': 'failed', 'stage': current, 'run_name': args.run_name, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
