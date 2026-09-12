"""Independently verify frozen artifacts and recompute main paired statistics.

No production analysis, geometry, cohort-selection, or bootstrap helpers used.
The tokenizer transplant builder is used only to enumerate exact replay records.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from persona_dynamics.transcripts import build_transplants

p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('output_dir');p.add_argument('--tokenizer',required=True)
a=p.parse_args();root=Path(a.run_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
cfg=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
t=pd.read_csv(root/'tokens.csv.gz',low_memory=False)
z=np.load(root/'directions.npz');meta=json.loads(z['metadata'].item());names=meta['direction_names']
assert not t.duplicated(['record_id','token_idx']).any()
assert np.isfinite(t[['cos_'+n for n in names]+['centered_norm','resid_norm']].to_numpy()).all()
assert np.allclose(np.linalg.norm(z['directions'],axis=1),1,atol=1e-12)
orthogonality_error=max(abs(z['directions'][names.index(n)]@z['directions'][0]) for n in ('ctrl_1','ctrl_2'))
# Published vectors are float32; orthogonalization inherits that precision.
assert orthogonality_error<1e-6
projection_errors={}
for n,d in zip(names,z['directions']):
    derived=(t['dot_'+n]-z['center']@d)/t.centered_norm
    # Saved GPU dots use float32 accumulation; subtracting the center amplifies
    # rounding in this independent float64 algebraic reconstruction.
    error=float(abs(derived-t['cos_'+n]).max());assert error<1e-5,(n,error)
    projection_errors[n]=error
records=[json.loads(f.read_text()) for f in sorted((root/'transcripts').glob('*.json'))]
by_id={r['record_id']:r for r in records};assert len(by_id)==len(records)
tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
for r in records:
    assert r['input_ids']==r['prefix_ids']+r['generated_ids']
    assert r['generated_text']==tok.decode(r['generated_ids'],skip_special_tokens=False)
    if r['valid']:
        for s in r['spans']:
            assert 0<=s['start']<s['end']<=len(r['input_ids'])
        if r['condition']=='natural_think':
            assert r['input_ids'][r['boundary_idx']]==tok.encode('</think>',add_special_tokens=False)[0]
for rid in t.loc[t.experiment.eq('E1')&t.domain.eq('advice')&t.condition.eq('natural_think'),'record_id'].unique():
    for r in build_transplants(by_id[rid],tok):
        source=by_id[r['source_record_id']];ss=next(s for s in source['spans'] if s['segment']==r['source_content'])
        assert r['content_token_ids']==source['input_ids'][ss['start']:ss['end']]
        by_id[r['record_id']]=r
for rid,rows in t.groupby('record_id',sort=False):
    r=by_id[rid];expected=[]
    for s in r['spans']:
        for offset,i in enumerate(range(s['start'],s['end'])):
            expected.append((i,r['input_ids'][i],s['segment'],offset,
                 s['segment']!='boundary' and (r['experiment']=='E2' or s['segment']=='think' or offset>=5)))
    actual=list(rows.sort_values('token_idx')[['token_idx','token_id','segment','segment_idx','primary_include']].itertuples(index=False,name=None))
    assert actual==sorted(expected),(rid,len(actual),len(expected))
# Independently validate eligibility and frozen-order selection.
req=[json.loads(x) for x in (root/'requests_experiments.jsonl').read_text().splitlines()]
cohorts=json.loads((root/'cohort.json').read_text());eligible={}
for r in records:
    counts={s:sum(x['end']-x['start'] for x in r['spans'] if x['segment']==s) for s in ('think','answer')}
    eligible[r['record_id']]=bool(r['valid'] and r['finish_reason']!='length' and counts['answer']>5 and
        (r['condition']!='natural_think' or counts['think']>=cfg['min_think_tokens']))
attrition=pd.read_csv(root/'attrition.csv')
assert all(bool(row.eligible)==eligible[row.record_id] for row in attrition.itertuples())
for c in cohorts:
    order=list(dict.fromkeys(r['prompt_id'] for r in req if r['domain']==c['domain'] and r['experiment']=='E1'))
    survivors=[pid for pid in order if all(eligible[r['record_id']] for r in req if r['prompt_id']==pid)]
    assert survivors[:40]==c['selected_prompt_ids']
    assert len(survivors)==c['eligible_all_seeds']
    actual=set(t.loc[t.domain.eq(c['domain'])&t.experiment.eq('E1'),'prompt_id'])
    assert actual==set(c['selected_prompt_ids'])
# Directly compute means from tokens, differences within prompts, CIs over prompts.
rows=[];all_gaps={}
for domain in ('advice','math'):
    natural=t.loc[t.experiment.eq('E1')&t.domain.eq(domain)&t.condition.eq('natural_think')&t.primary_include]
    for direction in ['cos_'+n for n in names]:
        m=natural.groupby(['prompt_id','segment'])[direction].mean().unstack()
        gap=m.think-m.answer;all_gaps[domain,direction]=gap
        rows.append((domain,direction,'think_minus_answer',gap,'e1_effects'))
e2=t.loc[t.experiment.eq('E2')&t.primary_include]
for direction in ['cos_'+n for n in names]:
    m=e2.groupby(['prompt_id','condition'])[direction].mean().unstack()
    ct,cp,at,ap=(m[c] for c in ('cot_as_cot','cot_as_answer','answer_as_cot','answer_as_answer'))
    values={'cot_tag_think_minus_plain':ct-cp,'answer_tag_think_minus_plain':at-ap,
            'factorial_tag_main':(ct-cp+at-ap)/2,'factorial_content_main':(ct-at+cp-ap)/2,
            'factorial_tag_content_interaction':ct-cp-at+ap}
    for contrast,v in values.items():rows.append(('advice',direction,contrast,v,'e2_effects'))
for domain in ('advice','math'):
    for n in names:
        if n!='assistant':
            v=abs(all_gaps[domain,'cos_assistant'])-abs(all_gaps[domain,'cos_'+n])
            rows.append((domain,'cos_'+n,'think_minus_answer',v,'specificity'))
checks=[]
for domain,direction,contrast,v,table in rows:
    x=v.sort_index().to_numpy();assert len(x)==40
    saved=pd.read_csv(root/(table+'.csv'))
    col='control' if table=='specificity' else 'direction'
    match=saved.loc[saved.domain.eq(domain)&saved[col].eq(direction)&saved.contrast.eq(contrast)&saved.seed_scope.eq('pooled')]
    assert len(match)==1
    record=match.iloc[0];rng=np.random.default_rng(cfg['analysis_seed'])
    draws=x[rng.integers(len(x),size=(cfg['bootstrap_samples'],len(x)))].mean(1)
    low,high=np.quantile(draws,[.025,.975])
    assert np.allclose([x.mean(),low,high],[record['mean'],record.ci_low,record.ci_high],atol=1e-12,rtol=0),(table,contrast)
    # More resamples check Monte Carlo stability; these are sensitivity intervals.
    rng=np.random.default_rng(2026);ci10k=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975])
    checks.append(dict(table=table,domain=domain,direction=direction,contrast=contrast,n=len(x),mean=x.mean(),
                      ci_low=low,ci_high=high,sensitivity_10000_low=ci10k[0],sensitivity_10000_high=ci10k[1]))
pd.DataFrame(checks).to_csv(out/'independent_statistics.csv',index=False)
source=out/'original_source'
source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name,sha in manifest['code'].items()}
assert all(source_hashes.values()),source_hashes
invalid=Counter((r['domain'],r['finish_reason'],r.get('failure_reason')) for r in records if not r['valid'])
result={'passed':True,'token_rows':len(t),'measured_records':t.record_id.nunique(),'natural_records':len(records),
        'statistics_recomputed':len(checks),'source_hashes_verified':len(source_hashes),
        'projection_identity_max_errors':projection_errors,'control_orthogonality_error':float(orthogonality_error),
        'invalid_records':[{'domain':k[0],'finish_reason':k[1],'reason':k[2],'n':v} for k,v in invalid.items()],
        'length_terminated':dict(Counter(r['domain'] for r in records if r['experiment']=='E1' and r['finish_reason']=='length'))}
(out/'saved_run_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
