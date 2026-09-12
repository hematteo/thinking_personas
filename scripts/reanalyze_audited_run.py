"""Regenerate a corrected report without modifying a frozen scientific run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
from persona_dynamics.analysis import analyze
from persona_dynamics.judge import export_review

p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('output_dir');a=p.parse_args()
source=Path(a.run_dir).resolve();out=Path(a.output_dir).resolve()
if source==out or source in out.parents:
    raise ValueError('Audit output must be outside the frozen run')
if out.exists():
    raise ValueError('Use a fresh audit output directory')
out.mkdir(parents=True)
cfg=json.loads((source/'config.json').read_text())
for name in ('config.json','e0.json','cohort.json','extraction.json'):
    shutil.copy2(source/name,out/name)
attrition=pd.read_csv(source/'attrition.csv')
changed=attrition.finish_reason.eq('length') & ~attrition.exclusion_reason.eq('truncated')
attrition.loc[attrition.finish_reason.eq('length'),'exclusion_reason']='truncated'
attrition.to_csv(out/'attrition.csv',index=False)
records=[json.loads(x) for x in (source/'review_transcripts.jsonl').read_text().splitlines()]
export_review(records,out/'review',seed=cfg['analysis_seed'],domains=cfg['domains'],require_roles='E3' in cfg['experiments'])
# Preserve existing human work, if supplied, without changing its content.
for name in ('review.csv','manual_calibration.csv'):
    if (source/'review'/name).exists():shutil.copy2(source/'review'/name,out/'review'/name)
if (source/'human_notes.md').exists():shutil.copy2(source/'human_notes.md',out/'human_notes.md')
tokens=pd.read_csv(source/'tokens.csv.gz',low_memory=False)
summary=analyze(tokens,out,bootstrap_samples=cfg['bootstrap_samples'],seed=cfg['analysis_seed'])
# All numeric output tables must remain the same after metadata/report fixes.
verified=[]
for name in ('e1_effects','e2_effects','specificity','segment_levels','boundary_first5'):
    old=pd.read_csv(source/(name+'.csv'));new=pd.read_csv(out/(name+'.csv'))
    pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-12,atol=1e-12)
    verified.append(name)
provenance={'source_run':str(source),'source_run_id':json.loads((source/'manifest.json').read_text())['run_id'],
 'source_tokens_sha256':hashlib.sha256((source/'tokens.csv.gz').read_bytes()).hexdigest(),
 'audit_code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path('src/persona_dynamics').glob('*.py'))},
 'numeric_tables_unchanged':verified,'attrition_reasons_relabelled':int(changed.sum()),
 'original_run_modified':False,'purpose':'Post-run code audit and reporting corrections; no new scientific measurements'}
(out/'audit_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
