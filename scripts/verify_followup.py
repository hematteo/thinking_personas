"""Verify completed follow-up artifacts independently of narrative conclusions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();r=Path(a.run_dir)
s=json.loads((r/'summary.json').read_text());m=json.loads((r/'manifest.json').read_text())
assert s['validation']['passed'];assert s['n_requested']==m['expected_main_records']
t=pd.read_csv(r/'tokens.csv.gz',low_memory=False)
assert len(t)==s['n_token_rows'];assert t.record_id.nunique()==s['n_valid']
assert not t.duplicated(['record_id','token_idx']).any()
metrics=[c for c in t if c.startswith(('dot_','cos_')) and t[c].notna().any()]
assert np.isfinite(t[metrics].to_numpy()).all()
assert t.loc[t.segment.eq('answer'),'primary_include'].all()
assert not t.loc[t.segment.eq('boundary'),'primary_include'].any()
z=np.load(r/'directions.npz');meta=json.loads(z['metadata'].item());names=meta['direction_names']
assert len(names)==14
for name,d in zip(names,z['directions']):
 assert np.linalg.norm(d)>0.999999
 predicted=(t['dot_'+name]-z['center']@d)/t.centered_norm
 assert float(abs(predicted-t['cos_'+name]).max())<1e-5
pairs=pd.read_csv(r/'paired_question_effects.csv')
for keys,g in pairs.groupby(['arm','reference_arm','metric','policy','contrast']):
 assert g.groupby('prompt_id').seed.nunique().eq(len(s['seeds'])).all()
e=pd.read_csv(r/'effects.csv');main=e[e.seed_scope.eq('pooled') & e.metric.eq('dot_assistant') & e.policy.eq('all_content') & e.contrast.eq('segment_interaction') & e.arm.str.startswith('role_')]
assert len(main)==12
assert main.n.between(1,12).all()
for f in s['figures']:
 assert (r/f).stat().st_size>1000 and (r/f).with_suffix('.pdf').stat().st_size>1000
out={'verified':True,'synthetic':s['synthetic'],'token_rows':len(t),'records':s['n_valid'],'requested':s['n_requested'],'directions':len(names),'primary_interactions':len(main),'minimum_paired_questions':int(main.n.min()),'figures':len(s['figures'])}
(r/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
