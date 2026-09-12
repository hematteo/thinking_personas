"""Recompute follow-up interactions from token shards without study estimators."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.run_dir);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'manifest.json').read_text());spec=manifest['spec'];seeds=spec['seeds']
requests=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines() if x]
by_id={r['record_id']:r for r in requests}
summary=json.loads((root/'summary.json').read_text())
t=pd.read_csv(root/'tokens.csv.gz',low_memory=False)
assert not t.duplicated(['record_id','token_idx']).any()
assert set(t.seed)==set(seeds)
records={p.stem:json.loads(p.read_text()) for p in (root/'transcripts').glob('*.json')}
assert len(records)==len(requests)
for rid,r in records.items():
 req=by_id[rid];assert r['prefix_ids']==req['prefix_ids']
 assert r['input_ids']==r['prefix_ids']+r['generated_ids']
 assert r['messages']==req['messages']
 assert r['generation_seed']==req['generation_seed']
for rid,g in t.groupby('record_id',sort=False):
 r=records[rid];assert r['finish_reason']!='length' and r['valid']
 expected=[]
 for span in r['spans']:
  for offset,i in enumerate(range(span['start'],span['end'])):
   expected.append((i,r['input_ids'][i],span['segment'],offset,span['segment'] in ['think','answer']))
 actual=list(g.sort_values('token_idx')[['token_idx','token_id','segment','segment_idx','primary_include']].itertuples(index=False,name=None))
 assert actual==sorted(expected)
# These identities also detect accidental token shifting between saved metrics.
z=np.load(root/'directions.npz');meta=json.loads(z['metadata'].item());errors={}
for name,d in zip(meta['direction_names'],z['directions']):
 error=float(abs((t['dot_'+name]-z['center']@d)/t.centered_norm-t['cos_'+name]).max())
 assert error<1e-5;errors[name]=error
complete=t[t.segment.isin(['think','answer'])]
means=complete.groupby(['seed','prompt_id','arm','segment'])[['dot_assistant','cos_assistant']].mean()
saved=pd.read_csv(root/'effects.csv');checks=[]
for metric in ['dot_assistant','cos_assistant']:
 for arm in sorted(t.arm.unique()):
  if not arm.startswith('role_'):continue
  refs=['default']+(['style_'+arm[5:]] if 'style_'+arm[5:] in set(t.arm) else [])
  for ref in refs:
   w=means[metric].unstack(['arm','segment'])
   need=[(arm,'think'),(arm,'answer'),(ref,'think'),(ref,'answer')]
   w=w.dropna(subset=need)
   counts=w.reset_index().groupby('prompt_id').seed.nunique();ids=counts[counts.eq(len(seeds))].index
   w=w[w.index.get_level_values('prompt_id').isin(ids)]
   effect=((w[arm,'think']-w[ref,'think'])-(w[arm,'answer']-w[ref,'answer'])).groupby('prompt_id').mean().sort_index().to_numpy()
   rng=np.random.default_rng(spec['bootstrap_seed']);draws=effect[rng.integers(len(effect),size=(spec['bootstrap_samples'],len(effect)))].mean(1)
   ci=np.quantile(draws,[.025,.975])
   target=saved[saved.arm.eq(arm)&saved.reference_arm.eq(ref)&saved.metric.eq(metric)&saved.policy.eq('all_content')&saved.contrast.eq('segment_interaction')&saved.seed_scope.eq('pooled')].iloc[0]
   assert len(effect)==target.n
   assert np.allclose([effect.mean(),*ci],[target['mean'],target.ci_low,target.ci_high],atol=1e-10,rtol=0)
   check={'metric':metric,'arm':arm,'reference':ref,'n':len(effect),'mean':float(effect.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])}
   if metric=='dot_assistant':
    rng=np.random.default_rng(spec['bootstrap_seed']);draws=effect[rng.integers(len(effect),size=(10000,len(effect)))].mean(1)
    family=np.quantile(draws,[.05/24,1-.05/24])
    assert np.allclose(family,[target.familywise_ci_low,target.familywise_ci_high],atol=1e-10,rtol=0)
    check.update(familywise_low=float(family[0]),familywise_high=float(family[1]))
   checks.append(check)
pd.DataFrame(checks).to_csv(out/'independent_interactions.csv',index=False)
result={'passed':True,'run_id':manifest['run_id'],'records_checked':len(records),'measured_records':t.record_id.nunique(),'token_rows':len(t),'interactions_recomputed':len(checks),'projection_identity_max_errors':errors,
        'tokens_sha256':hashlib.sha256((root/'tokens.csv.gz').read_bytes()).hexdigest()}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
