"""Checks of the scientific comparisons behind the publication figures."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('paper_figures',Path(__file__).parents[1]/'scripts/build_paper_figures.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_prompt_means_do_not_weight_long_responses_or_seeds_as_independent():
    rows=[]
    for seed,prompt,think,answer,length in [(0,'p0',3,1,100),(1,'p0',7,1,5),(0,'p1',0,2,3),(1,'p1',0,2,3)]:
        for segment,value in [('think',think),('answer',answer)]:
            for i in range(length):
                rows.append(dict(seed=seed,prompt_id=prompt,domain='advice',experiment='E1',condition='natural_think',
                    segment=segment,primary_include=True,resid_norm=2.,centered_norm=1.,
                    **{f'{p}_{d}':value for p in ['dot','cos'] for d in module.DIRS}))
    means=module.segment_means(pd.DataFrame(rows));effects=module.e1_effects(means)
    a=effects[effects.metric.eq('projection')&effects.direction.eq('assistant')].set_index('prompt_id')
    assert a.loc['p0','value']==4 # equal seed weighting of paired gaps 2 and 6
    assert a.loc['p1','value']==-2
    s=module.summarize(a.reset_index(),['metric','direction','domain'],100)
    assert s.n.iloc[0]==2 and s['mean'].iloc[0]==1


def test_e2_pairs_content_specific_baselines_and_recovers_interaction():
    values={'cot_as_cot':8,'cot_as_answer':3,'cot_in_scratch':1,'cot_in_special':2,
            'answer_as_cot':4,'answer_as_answer':2,'answer_in_scratch':0,'answer_in_special':1}
    means=pd.DataFrame([dict(seed=0,domain='advice',prompt_id='p',experiment='E2',condition=c,
        **{f'{p}_{d}':value for p in ['dot','cos'] for d in module.DIRS}) for c,value in values.items()])
    contrasts,factorial=module.e2_effects(means)
    a=contrasts[contrasts.metric.eq('projection')&contrasts.direction.eq('assistant')].set_index(['content','context']).value
    assert a['cot','Thinking']==5 and a['answer','Thinking']==2
    assert a['cot','Scratch']==-2 and a['answer','Tool-response']==-1
    f=factorial[factorial.metric.eq('projection')&factorial.direction.eq('assistant')].set_index('contrast').value
    assert f['interaction']==3 and f['context_main']==3.5 and f['content_main']==2.5


def test_content_pair_validation_rejects_one_changed_token():
    rows=[]
    for domain in module.DOMAINS:
        for content in ['cot','answer']:
            for context in ['as_cot','as_answer','in_scratch','in_special']:
                rows.append(dict(seed=0,domain=domain,prompt_id=domain,model='m',layer=32,experiment='E2',
                    condition=content+'_'+context,record_id=domain+content+context,token_idx=5,token_id=7,
                    source_record_id=domain,segment_idx=0,primary_include=True,resid_norm=2.,centered_norm=1.,
                    **{f'{p}_{d}':1. for p in ['dot','cos'] for d in module.DIRS}))
    frame=pd.DataFrame(rows)
    assert module.validate_tokens(frame)['matched_e2_content_positions']==4
    frame.loc[0,'token_id']=8
    with pytest.raises(ValueError,match='token IDs'):
        module.validate_tokens(frame)


def test_specificity_is_mean_of_paired_absolute_gaps():
    rows=[]
    for prompt,a,c in [('p0',2,3),('p1',-2,3)]:
        for d,v in [('assistant',a),*[(d,c) for d in module.DIRS[1:]]]:
            rows.append(dict(domain='advice',metric='projection',prompt_id=prompt,direction=d,value=v))
    result=module.specificity(pd.DataFrame(rows))
    assert result.value.eq(-1).all() # not |mean(a)| - |mean(c)| = -3


def test_csv_null_control_is_not_silently_read_as_missing(tmp_path):
    frame=pd.DataFrame({'direction':['assistant','null'],'value':[1.,2.]})
    module.write_table(frame,tmp_path/'table.csv')
    loaded=pd.read_csv(tmp_path/'table.csv')
    assert loaded.direction.tolist()==['assistant','structured_null']
    assert frame.direction.tolist()==['assistant','null']
