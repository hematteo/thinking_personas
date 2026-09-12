"""Frozen, bounded E3 complement; reuse the audited token parser and scalar hook.

Generation/extraction workers are process-isolated and shard only by seed.
The original scientific run and its historical cosine analyzer are untouched.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .config import Config
from .geometry import DirectionBundle, extract_record, orthogonalize
from .io import digest, file_hash, read_jsonl, stable_seed, write_json, write_jsonl
from .backends import tokenizer_for, generate_batches, model_for, fixture_activations
from .transcripts import build_transcript, annotate_transcript


def load(path):
    spec=json.loads(Path(path).read_text())
    config=Config(**spec['generation']).validate()
    assert spec['seeds']==config.seeds
    return spec,config


def make_arms(roles,styles):
    result=[dict(arm='default',role='default',condition='natural_think',system_prompt='You are an AI Assistant.')]
    result += [dict(arm='role_'+r['role'],role=r['role'],condition='role_prompt',system_prompt=r['system_prompt']) for r in roles]
    result += [dict(arm='style_'+role,role=role,condition='style_prompt',system_prompt=text) for role,text in styles.items()]
    if len({r['arm'] for r in result})!=len(result):raise ValueError('Duplicate arms')
    return result


def initialize(path):
    spec,config=load(path);root=Path(spec['output_dir']);inputs=Path(spec['inputs_dir'])
    root.mkdir(parents=True,exist_ok=True)
    files={str(p.relative_to(inputs)):file_hash(p) for p in sorted(inputs.rglob('*')) if p.is_file()}
    versions={n:importlib.metadata.version(n) if importlib.util.find_spec(n) else None for n in ['numpy','pandas','torch','transformers','vllm']}
    source={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity=dict(spec=spec,inputs=files,code=source,versions=versions)
    run_id=digest(identity)[:16]
    if (root/'manifest.json').exists():
        if json.loads((root/'manifest.json').read_text())['run_id']!=run_id:raise ValueError('Frozen run changed; use a new output directory')
        return spec,config,root
    parent=json.loads((inputs/'parent_manifest.json').read_text())
    gate=json.loads((inputs/'parent_e0.json').read_text())
    if config.backend!='fixture':
        assert parent['config']['model']==config.model and parent['config']['revision']==config.revision
        assert parent['config']['layer']==config.layer and gate['passed']
    roles=json.loads((inputs/'roles.json').read_text());styles=json.loads((inputs/'styles.json').read_text())
    arms=make_arms(roles,styles)
    questions=read_jsonl(inputs/'questions.jsonl');pilot=read_jsonl(inputs/'pilot.jsonl')
    assert len({q['prompt_id'] for q in questions+pilot})==len(questions+pilot)
    if config.backend=='fixture':
        d=config.synthetic_dimension;rng=np.random.default_rng(711)
        directions={'assistant':np.eye(d)[0]}
        directions.update({n:rng.normal(size=d) for n in ['ctrl_1','ctrl_2','null','random']})
        directions.update({'role_'+r['role']:orthogonalize(rng.normal(size=d),directions['assistant']) for r in roles})
        bundle=DirectionBundle(np.zeros(d),directions,{'synthetic':True})
    else:
        import torch
        bundle=DirectionBundle.load(inputs/'parent_directions.npz')
        default=torch.load(inputs/'vectors/default_vector.pt',weights_only=True,map_location='cpu')[config.layer].double().numpy()
        for r in roles:
            p=inputs/'vectors'/(r['role']+'.pt');assert file_hash(p)==r['vector_sha256']
            vector=torch.load(p,weights_only=True,map_location='cpu')[config.layer].double().numpy()-default
            bundle.directions['role_'+r['role']]=orthogonalize(vector,bundle.directions['assistant'])
        bundle.metadata.update(parent_run_id=parent['run_id'],study=spec['study'])
    bundle.save(root/'directions.npz')
    tokenizer=tokenizer_for(config)
    requests=[]
    for phase,qs,seeds,aa in [('pilot',pilot,[spec['pilot_seed']],[a for a in arms if a['arm'] in ['default','role_poet','role_doctor']]),('main',questions,spec['seeds'],arms)]:
        for seed in seeds:
            for q in qs:
                for arm in aa:
                    messages=[{'role':'system','content':arm['system_prompt']+'\n\n'+spec['answer_instruction']},{'role':'user','content':q['text']}]
                    prefix=list(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True,return_dict=False))
                    if len(prefix)+config.max_tokens>config.max_context:raise ValueError('Context budget exceeded')
                    row={**q,**arm,'seed':seed,'phase':phase,'messages':messages,'prefix_ids':prefix,'enable_thinking':True,
                         'generation_seed':stable_seed(seed,q['prompt_id'],arm['arm']),
                         'record_id':digest([run_id,phase,seed,q['prompt_id'],arm['arm']])[:24],
                         'run_id':run_id,'experiment':'E3_followup','model':config.model,'synthetic':config.backend=='fixture'}
                    requests.append(row)
    write_jsonl(root/'requests.jsonl',requests)
    write_json(root/'config.json',spec)
    write_json(root/'arms.json',arms)
    write_jsonl(root/'questions.jsonl',questions)
    write_json(root/'manifest.json',{**identity,'run_id':run_id,'created_unix':time.time(),'parent_run_id':parent['run_id'],
        'parent_e0_passed':gate['passed'],'synthetic':config.backend=='fixture','expected_main_records':len(questions)*len(arms)*len(spec['seeds'])})
    print(f'Frozen {len(requests)} requests; {len(arms)} arms; run {run_id}',flush=True)
    return spec,config,root


def requests(root,phase,seed=None):
    return [r for r in read_jsonl(root/'requests.jsonl') if r['phase']==phase and (seed is None or r['seed']==seed)]


def generate(path,phase,seed=None):
    spec,config,root=initialize(path)
    if phase=='main' and not (root/'validation.json').exists():raise RuntimeError('Numerical and pilot validation must pass first')
    wanted=requests(root,phase,seed);cache=root/'transcripts';cache.mkdir(exist_ok=True)
    pending=[r for r in wanted if not (cache/(r['record_id']+'.json')).exists()]
    tokenizer=tokenizer_for(config)
    for i,(r,ids,finish) in enumerate(generate_batches(config,pending,tokenizer)):
        record=build_transcript(tokenizer,r,r['condition'],ids,r['prefix_ids'],r['seed'],r['role'],'E3_followup')
        record.update({k:r[k] for k in ['record_id','run_id','arm','phase','domain','generation_seed','synthetic']})
        record.update(prefix_ids=r['prefix_ids'],generated_ids=ids,finish_reason=finish)
        write_json(cache/(r['record_id']+'.json'),record)
        if (i+1)%13==0:print(f'{phase} seed {seed}: saved {i+1}/{len(pending)}',flush=True)
    print(f'{phase} seed {seed}: generated {len(pending)}, cached {len(wanted)-len(pending)}',flush=True)


def validity(record):
    if record.get('finish_reason')=='length':return False,'truncated'
    if not record.get('valid',False):return False,record.get('failure_reason','malformed')
    counts={k:sum(s['end']-s['start'] for s in record['spans'] if s['segment']==k) for k in ['think','answer']}
    if counts['think']<1 or counts['answer']<1:return False,'empty_segment'
    return True,'eligible'


def measure(config,model,record,bundle,means=True):
    if config.backend=='fixture':
        hidden=fixture_activations(record,config.synthetic_dimension)
        rows=annotate_transcript(record);project=bundle.project(hidden)
        for row in rows:row.update({k:float(v[row['token_idx']]) for k,v in project.items()})
        vectors={s:hidden[[row['token_idx'] for row in rows if row['segment']==s]].mean(0) for s in ['think','answer']}
    else:
        rows,vectors=extract_record(model,record,bundle,config.layer,keep_segment_means=means,answer_skip=0)
    for row in rows:
        row.update({k:record[k] for k in ['record_id','run_id','seed','arm','phase','prompt_id','domain','role','condition','model','synthetic']})
        row['layer']=config.layer
        # Collection retains boundary and every content token. Filters are analysis metadata.
        row['primary_include']=row['segment'] in ['think','answer']
        row['sensitivity_skip5_include']=row['primary_include'] and (row['segment']!='answer' or row['segment_idx']>=5)
    return rows,vectors


def validate(path):
    spec,config,root=initialize(path);inputs=Path(spec['inputs_dir'])
    pilot=[json.loads((root/'transcripts'/(r['record_id']+'.json')).read_text()) for r in requests(root,'pilot')]
    if not all(validity(r)[0] for r in pilot):raise RuntimeError('Pilot generation incomplete; do not start main study')
    bundle=DirectionBundle.load(root/'directions.npz')
    model=None if config.backend=='fixture' else model_for(config)
    checks=[]
    if model is not None:
        golden=pd.read_csv(inputs/'golden_tokens.csv.gz')
        for file in sorted((inputs/'golden_records').glob('*.json')):
            record=json.loads(file.read_text());fresh,_=extract_record(model,record,bundle,config.layer)
            fresh=pd.DataFrame(fresh);old=golden[golden.record_id.eq(record['record_id'])].sort_values('token_idx')
            fresh=fresh.sort_values('token_idx')
            assert fresh.token_idx.tolist()==old.token_idx.tolist()
            cols=['dot_assistant','cos_assistant','resid_norm']
            errors={c:float(abs(fresh[c].to_numpy()-old[c].to_numpy()).max()) for c in cols}
            assert errors['cos_assistant']<1e-5 and errors['dot_assistant']<.01 and errors['resid_norm']<.01,errors
            checks.append(dict(record_id=record['record_id'],max_errors=errors))
    for r in pilot:
        rows,_=measure(config,model,r,bundle)
        values=pd.DataFrame(rows)[['dot_assistant','cos_assistant']].to_numpy()
        assert np.isfinite(values).all()
        assert len(rows)==sum(s['end']-s['start'] for s in r['spans'])
    write_json(root/'validation.json',{'passed':True,'pilot_records':len(pilot),'fresh_golden_replays':checks,
        'reused_parent_e0':json.loads((inputs/'parent_e0.json').read_text()),'synthetic':config.backend=='fixture'})
    print('VALIDATION PASSED: full pilot generation/extraction and frozen numerical references',flush=True)


def extract(path,seed):
    spec,config,root=initialize(path)
    if not json.loads((root/'validation.json').read_text())['passed']:raise RuntimeError('Validation failed')
    wanted=requests(root,'main',seed);bundle=DirectionBundle.load(root/'directions.npz')
    model=None if config.backend=='fixture' else model_for(config)
    shards=root/'token_shards';shards.mkdir(exist_ok=True);meansdir=root/'segment_vectors';meansdir.mkdir(exist_ok=True)
    attrition=[]
    for i,r in enumerate(wanted):
        path_record=root/'transcripts'/(r['record_id']+'.json')
        if not path_record.exists():raise RuntimeError(f'Missing generation {path_record}')
        record=json.loads(path_record.read_text());valid,reason=validity(record)
        attrition.append({k:record[k] for k in ['record_id','arm','seed','prompt_id','domain','phase','finish_reason','think_token_count','answer_token_count']}|{'eligible':valid,'exclusion_reason':reason})
        if not valid:continue
        dest=shards/(record['record_id']+'.csv.gz');vectors_path=meansdir/(record['record_id']+'.npz')
        if not dest.exists() or not vectors_path.exists():
            rows,vectors=measure(config,model,record,bundle)
            pd.DataFrame(rows).to_csv(dest,index=False)
            np.savez_compressed(vectors_path,**vectors)
        if (i+1)%26==0:print(f'Extracted seed {seed} {i+1}/{len(wanted)}',flush=True)
    pd.DataFrame(attrition).to_csv(root/f'attrition-{seed}.csv',index=False)
    print(f'Extraction seed {seed} complete',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['init','generate-pilot','validate','generate','extract','analyze'],required=True);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.stage=='init':initialize(a.config)
    elif a.stage=='generate-pilot':generate(a.config,'pilot')
    elif a.stage=='validate':validate(a.config)
    elif a.stage=='generate':generate(a.config,'main',a.seed)
    elif a.stage=='extract':extract(a.config,a.seed)
    else:
        from .followup_analysis import analyze
        spec,config,root=initialize(a.config);analyze(root)

if __name__=='__main__':main()
