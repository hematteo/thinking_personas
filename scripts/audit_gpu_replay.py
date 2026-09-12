"""Independent, read-only residual replay; writes only a separate audit artifact.

Does not call production extraction or projection functions. Reconstructs E2 IDs
with the tokenizer utility, then checks saved scalar values with NumPy float64.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from persona_dynamics.transcripts import build_transplants

p = argparse.ArgumentParser()
p.add_argument('run_dir'); p.add_argument('output')
a = p.parse_args(); root = Path(a.run_dir)
cfg = json.loads((root/'config.json').read_text())
z = np.load(root/'directions.npz'); center=z['center']; matrix=z['directions'].T
meta=json.loads(z['metadata'].item()); names=meta['direction_names']
tokens=pd.read_csv(root/'tokens.csv.gz',low_memory=False)
records=[json.loads(f.read_text()) for f in sorted((root/'transcripts').glob('*.json'))]
by_id={r['record_id']:r for r in records}
selected=[]
for domain in ('advice','math'):
    for condition in ('natural_think','natural_nothink'):
        rid=tokens.loc[tokens.domain.eq(domain)&tokens.condition.eq(condition),'record_id'].iloc[0]
        selected.append(by_id[rid])
tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True)
selected += [r for r in build_transplants(selected[0],tok) if r['condition'] in ('cot_as_answer','answer_as_cot')]
print('Loading pinned checkpoint for independent replay',flush=True)
model=AutoModelForCausalLM.from_pretrained(cfg['model'],revision=cfg['revision'],local_files_only=True,
    torch_dtype=torch.bfloat16,device_map='auto',max_memory={i:'40GiB' for i in range(torch.cuda.device_count())},
    attn_implementation='sdpa').eval()
base=model.model; device=base.embed_tokens.weight.device
result={'model':cfg['model'],'revision':cfg['revision'],'layer':cfg['layer'],'records':[],
        'torch':torch.__version__,'devices':torch.cuda.device_count()}

def forward(record, indices):
    captured=[]
    def hook(_m,_i,output):
        h=output[0] if isinstance(output,(tuple,list)) else output
        captured.append(h[0,indices].detach().float().cpu().numpy().astype(np.float64))
    handle=base.layers[cfg['layer']].register_forward_hook(hook)
    try:
        ids=torch.tensor([record['input_ids']],device=device)
        with torch.inference_mode():
            base(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
    finally:
        handle.remove()
    assert len(captured)==1
    return captured[0]

for r in selected:
    rows=tokens.loc[tokens.record_id.eq(r['record_id'])].sort_values('token_idx')
    assert len(rows)>0
    # All recorded positions, including boundary and excluded answer tokens.
    h=forward(r,rows.token_idx.tolist()); c=h-center
    values={'resid_norm':np.linalg.norm(h,axis=1),'centered_norm':np.linalg.norm(c,axis=1)}
    cos=c@matrix/values['centered_norm'][:,None]; dot=h@matrix
    for i,n in enumerate(names): values['cos_'+n]=cos[:,i]; values['dot_'+n]=dot[:,i]
    errors={k:float(np.max(np.abs(v-rows[k].to_numpy()))) for k,v in values.items()}
    assert max(errors['cos_'+n] for n in names)<2e-5,errors
    assert max(errors.values())<.02,errors
    result['records'].append({'record_id':r['record_id'],'domain':r['domain'],'condition':r['condition'],
                              'tokens':len(rows),'max_absolute_errors':errors})
    print(result['records'][-1],flush=True)
# Independently recover the held-out center and random-half control from all
# answer-content tokens after the first five, preserving frozen prompt ordering.
cal={r['prompt_id']:r for r in records if r['experiment']=='calibration'}
means=[]; counts=[]
for pid in meta['calibration_prompt_ids']:
    r=cal[pid]; spans=[s for s in r['spans'] if s['segment']=='answer']; assert len(spans)==1
    indices=list(range(spans[0]['start']+cfg['first_answer_tokens'],spans[0]['end']))
    h=forward(r,indices); means.append(h.mean(0)); counts.append(len(h))
center_replay=np.average(means,axis=0,weights=counts)
order={p:i for i,p in enumerate(meta['calibration_prompt_ids'])}
a_ids,b_ids=meta['null_halves']; means=np.asarray(means)
null=means[[order[p] for p in a_ids]].mean(0)-means[[order[p] for p in b_ids]].mean(0)
null/=np.linalg.norm(null)
result['calibration']={'prompts':len(means),'tokens':sum(counts),
 'center_max_absolute_error':float(np.max(abs(center_replay-center))),
 'null_max_absolute_error':float(np.max(abs(null-matrix[:,names.index('null')])))}
assert result['calibration']['center_max_absolute_error']<.002,result['calibration']
assert result['calibration']['null_max_absolute_error']<2e-5,result['calibration']
result['passed']=True
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['calibration']),flush=True)
