"""Compare matched D1/D2 validation predictions and record changed errors."""
import argparse
import csv
import json
from pathlib import Path
from src.common.utils import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-name", required=True)
    args = parser.parse_args()
    d1_name = "baseline_full_20261006_001"
    d1 = ROOT / "results/phase1_catdog_baseline" / d1_name
    d2 = ROOT / "results/phase2_catdog_improvements" / args.d2_name
    run1 = ROOT / "runs/phase1_catdog_baseline" / d1_name
    run2 = ROOT / "runs/phase2_catdog_improvements" / args.d2_name
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    assert read(run1 / "data_manifest.json") == read(run2 / "data_manifest.json"), "Different data manifests"
    c1,c2=read(run1/'config.json'),read(run2/'config.json')
    for key in c1: assert c1[key]==c2[key], f"Different {key}"
    p1,p2=read(d1/'predictions.json'),read(d2/'predictions.json')
    assert [(x['path'],x['label']) for x in p1]==[(x['path'],x['label']) for x in p2]
    fixed=[]; introduced=[]; persistent=[]
    for a,b in zip(p1,p2):
        if a['prediction']!=a['label'] and b['prediction']==b['label']: fixed.append(b)
        elif a['prediction']==a['label'] and b['prediction']!=b['label']: introduced.append(b)
        elif a['prediction']!=a['label'] and b['prediction']!=b['label']: persistent.append(b)
    m1,m2=read(d1/'metrics.json'),read(d2/'metrics.json')
    save_json(d2/'comparison.json',{'d1':m1,'d2':m2,'accuracy_change_percentage_points':100*(m2['accuracy']-m1['accuracy']),'fixed_errors':fixed,'introduced_errors':introduced,'persistent_errors':persistent,'limitations':'Single seed; validation-selected checkpoints; no independent test accuracy.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for folder,label in [(run1,'D1'),(run2,'D2')]:
        rows=list(csv.DictReader((folder/'history.csv').open()))
        for ax,metric in zip(axes,['loss','accuracy']):
            for split in ['train','val']: ax.plot([int(x['epoch']) for x in rows],[float(x[split+'_'+metric]) for x in rows],label=f'{label} {split}')
    for ax,metric in zip(axes,['loss','accuracy']):ax.set(xlabel='Epoch',ylabel=metric);ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(d2/'comparison_curves.png',dpi=160);plt.close(fig)
    text=f"# D1 / D2 comparison\n\nD1 accuracy: {m1['accuracy']:.4%}\n\nD2 accuracy: {m2['accuracy']:.4%}\n\nFixed errors: {len(fixed)}; introduced errors: {len(introduced)}; persistent errors: {len(persistent)}.\n\nD2 training accuracy uses randomly augmented images and is not directly comparable to clean D1 training accuracy. Validation preprocessing is identical. RandomResizedCrop can use its fallback crop on extreme aspect ratios; scale bounds are sampling targets, not a guarantee for every output. This single-seed result does not establish a statistically stable improvement. D3 fine-tuning has not been executed.\n"
    (d2/'comparison.md').write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__ == '__main__': main()
