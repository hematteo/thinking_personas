import numpy as np
import pandas as pd
import pytest
from persona_dynamics.followup_analysis import paired_effects,summarize
from persona_dynamics.followup import make_arms,validity


def frame():
 rows=[]
 for seed in [17,23]:
  for q in ['q1','q2']:
   for arm in ['default','role_poet','style_poet']:
    for seg in ['think','answer','answer_skip5']:
     v=0 if arm=='default' else 2 if arm=='style_poet' else 1 if seg=='think' else 4
     rows.append(dict(seed=seed,prompt_id=q,domain='identity',arm=arm,segment=seg,dot_assistant=v,cos_assistant=v/10))
 return pd.DataFrame(rows)


def test_followup_interaction_and_style_are_paired():
 pairs=paired_effects(frame(),[17,23])
 chosen=pairs[pairs.metric.eq('dot_assistant') & pairs.policy.eq('all_content') & pairs.arm.eq('role_poet')]
 assert chosen[chosen.contrast.eq('segment_interaction')].value.eq(-3).all()
 style=chosen[chosen.comparison.eq('role_versus_style') & chosen.contrast.eq('answer_shift')]
 assert style.value.eq(2).all()
 effects=summarize(pairs,20,42)
 assert effects[effects.seed_scope.eq('pooled')].n.eq(2).all()


def test_followup_requires_complete_seed_pairs_not_doubled_observations():
 data=frame();data=data[~(data.seed.eq(23)&data.prompt_id.eq('q2')&data.arm.eq('role_poet')&data.segment.eq('answer'))]
 pairs=paired_effects(data,[17,23])
 assert set(pairs[pairs.arm.eq('role_poet') & pairs.policy.eq('all_content')].prompt_id)=={'q1'}
 assert set(pairs[pairs.arm.eq('role_poet') & pairs.policy.eq('skip5_sensitivity')].prompt_id)=={'q1','q2'}


def test_duplicate_followup_observation_is_not_averaged_away():
 f=frame()
 with pytest.raises(ValueError):paired_effects(pd.concat([f,f.iloc[[0]]]),[17,23])


def test_role_and_style_arms_have_distinct_identity():
 arms=make_arms([{'role':'poet','system_prompt':'You are a poet.'}],{'poet':'Use poetic style.'})
 assert [a['arm'] for a in arms]==['default','role_poet','style_poet']


def test_truncation_and_short_thinking_remain_distinct():
 r={'valid':True,'finish_reason':'stop','spans':[{'segment':'think','start':1,'end':2},{'segment':'answer','start':4,'end':5}]}
 assert validity(r)==(True,'eligible')
 assert validity(dict(r,finish_reason='length'))==(False,'truncated')
