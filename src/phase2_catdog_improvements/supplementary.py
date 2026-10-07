"""Frozen deterministic features: controlled learning-rate and model comparisons.

Feature extraction is timed separately; training-time comparisons with D1/D2
are not valid because this implementation caches frozen representations.
"""
import csv
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import classification_report, confusion_matrix
from src.common.utils import ROOT, seed_everything, save_json
from src.common.data import CatDogDataset, LABELS
from src.models.resnet18 import build_model, baseline_transform


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-only',action='store_true')
    args=parser.parse_args()
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.hub.set_dir(str(ROOT/'runs/pretrained_cache'))
    reference=json.loads((ROOT/'runs/phase1_catdog_baseline/baseline_full_20261006_001/data_manifest.json').read_text())
    trials=[('resnet18',.001,'resnet18_cached_lr001_control')] if args.control_only else [('resnet18',.0003,'resnet18_lr0003'),('mobilenet_v3_small',.001,'mobilenetv3small_baseline')]
    for family,lr,name in trials:
        out=ROOT/'runs/phase2_catdog_improvements'/name
        out.mkdir(parents=True,exist_ok=False)
        result=ROOT/'results/phase2_catdog_improvements'/name;result.mkdir(parents=True,exist_ok=False)
        seed_everything(42)
        if family=='resnet18':
            model=build_model();head=model.fc;model.fc=nn.Identity()
        else:
            model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            for p in model.parameters():p.requires_grad=False
            head=nn.Linear(model.classifier[-1].in_features,2);model.classifier[-1]=nn.Identity()
        model=model.to(device).eval();head=head.to(device)
        config={'architecture':family,'seed':42,'learning_rate':lr,'epochs':10,'batch_size':32,'augmentation':False,'execution':'cached frozen features','trainable_parameters':sum(p.numel() for p in head.parameters())}
        save_json(out/'config.json',config)
        features={};data={};started=time.perf_counter()
        for split in ['train','val','test']:
            ds=CatDogDataset(ROOT/'datasets',split,baseline_transform());data[split]=ds
            if split!='test':assert [str(p.relative_to(ROOT/'datasets')) for p,y in ds.samples]==reference[split]
            xs=[];ys=[]
            with torch.inference_mode():
                for index,(images,labels) in enumerate(DataLoader(ds,batch_size=32,shuffle=False)):
                    xs.append(model(images.to(device)).cpu());ys.append(labels)
                    if index%100==0:print(f'{name} {split}: batch {index}',flush=True)
            features[split]=(torch.cat(xs),torch.cat(ys))
        extraction=time.perf_counter()-started
        save_json(out/'data_manifest.json',reference)
        optimizer=torch.optim.Adam(head.parameters(),lr=lr)
        generator=torch.Generator().manual_seed(42)
        loader=DataLoader(TensorDataset(*features['train']),batch_size=32,shuffle=True,generator=generator)
        best=(-1,float('-inf'));history=[]
        for epoch in range(1,11):
            started=time.perf_counter();loss_sum=correct=0;head.train()
            for x,y in loader:
                x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True);logits=head(x);loss=nn.functional.cross_entropy(logits,y);loss.backward();optimizer.step()
                loss_sum+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item()
            head.eval()
            with torch.inference_mode():
                vx,vy=features['val'];logits=head(vx.to(device));loss=nn.functional.cross_entropy(logits,vy.to(device)).item();pred=logits.argmax(1).cpu();acc=(pred==vy).float().mean().item()
            # Compute exact fraction for direct comparison.
            acc=int((pred==vy).sum())/len(vy)
            record={'epoch':epoch,'train_loss':loss_sum/len(data['train']),'train_accuracy':correct/len(data['train']),'val_loss':loss,'val_accuracy':acc,'seconds':time.perf_counter()-started};history.append(record)
            key=(acc,-loss)
            if key>best:
                best=key
                torch.save({'backbone_state':model.cpu().state_dict(),'head_state':head.cpu().state_dict(),'config':config,'epoch':epoch,'label_mapping':LABELS},out/'best.pth');model.to(device);head.to(device)
                predictions=pred.tolist();metrics={'epoch':epoch,'accuracy':acc,'loss':loss,'count':len(vy),'confusion_matrix':confusion_matrix(vy,pred,labels=[0,1]).tolist(),'classification_report':classification_report(vy,pred,labels=[0,1],target_names=['cat','dog'],output_dict=True,zero_division=0)}
                save_json(result/'metrics.json',metrics);save_json(out/'best_metrics.json',metrics)
                save_json(result/'predictions.json',[{'path':str(p.relative_to(ROOT/'datasets')),'label':y,'prediction':z} for (p,y),z in zip(data['val'].samples,predictions)])
            print(name,record,flush=True)
        with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        checkpoint=torch.load(out/'best.pth',weights_only=True,map_location=device);model.load_state_dict(checkpoint['backbone_state']);head.load_state_dict(checkpoint['head_state']);model.eval();head.eval()
        # Re-evaluate saved model on original validation images, not cached logits.
        predictions=[];total_loss=0
        with torch.inference_mode():
            for images,labels in DataLoader(data['val'],batch_size=32):
                logits=head(model(images.to(device)));total_loss+=nn.functional.cross_entropy(logits,labels.to(device),reduction='sum').item();predictions.extend(logits.argmax(1).cpu().tolist())
        cached= json.loads((result/'predictions.json').read_text());assert predictions==[x['prediction'] for x in cached]
        save_json(result/'reload_verification.json',{'predictions_identical':True,'reloaded_loss':total_loss/5000,'cached_loss':json.loads((result/'metrics.json').read_text())['loss'],'note':'Batch aggregation can cause tiny floating-point loss differences.'})
        with torch.inference_mode():test=head(features['test'][0].to(device)).argmax(1).cpu().tolist()
        with (ROOT/'submission'/f'{name}.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['id','label']);w.writerows((i,z) for (p,i),z in zip(data['test'].samples,test))
        save_json(out/'timing.json',{'feature_extraction_seconds':extraction,'head_training_seconds':sum(x['seconds'] for x in history),'not_comparable_to_uncached_runs':True})
        save_json(result/'status.json',{'state':'complete','reloaded_original_images_verified':True})
    print('SUPPLEMENTARY EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
