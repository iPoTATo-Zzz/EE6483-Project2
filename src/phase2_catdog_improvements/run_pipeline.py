"""Sequential D2 formal training, reloaded evaluation and D1 comparison."""
import argparse
import json
import subprocess
import sys
from src.common.utils import ROOT, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    # Pilot must finish successfully before a full training run is launched.
    pilot=ROOT/'runs/phase2_catdog_improvements/augmentation_pilot_20261006_001'
    timing=json.loads((pilot/'timing.json').read_text())
    assert timing['epochs']==1 and (pilot/'best.pth').exists()
    status_dir=ROOT/'runs/phase2_catdog_improvements'/('pipeline_'+args.run_name)
    status_dir.mkdir(parents=True,exist_ok=False)
    result=f'results/phase2_catdog_improvements/{args.run_name}'
    checkpoint=f'runs/phase2_catdog_improvements/{args.run_name}/best.pth'
    commands=[
        ('training',['src.phase2_catdog_improvements.train','--run-name',args.run_name]),
        ('evaluation',['src.phase1_catdog_baseline.evaluate','--checkpoint',checkpoint,'--output',result]),
        ('comparison',['src.phase2_catdog_improvements.compare','--d2-name',args.run_name]),
        ('prediction',['src.phase1_catdog_baseline.predict','--checkpoint',checkpoint,'--output',f'submission/{args.run_name}.csv']),
    ]
    try:
        for stage,arguments in commands:
            save_json(status_dir/'status.json',{'stage':stage,'state':'running','run_name':args.run_name})
            print(f'Starting {stage}',flush=True)
            with (status_dir/f'{stage}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-u','-m',*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((ROOT/result/'metrics.json').read_text())
        best=json.loads((ROOT/'runs/phase2_catdog_improvements'/args.run_name/'best_metrics.json').read_text())
        assert metrics['accuracy']==best['accuracy'] and abs(metrics['loss']-best['loss'])<1e-9
        import csv
        rows=list(csv.DictReader((ROOT/f'submission/{args.run_name}.csv').open()))
        assert [int(x['id']) for x in rows]==list(range(1,501))
        assert all(x['label'] in ['0','1'] for x in rows)
        save_json(status_dir/'status.json',{'stage':'complete','state':'complete','run_name':args.run_name,'reload_and_csv_verified':True})
        print('D2 pipeline completed and verified',flush=True)
    except Exception as error:
        save_json(status_dir/'status.json',{'stage':stage,'state':'failed','error':str(error),'run_name':args.run_name})
        raise


if __name__=='__main__':main()
