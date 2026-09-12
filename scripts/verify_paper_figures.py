"""Independent checks of paper tables against frozen per-token measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(source,output):
    source,output=Path(source),Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    for name,sha in manifest['source_hashes'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
    t=pd.read_csv(source/'tokens.csv.gz',low_memory=False,float_precision='round_trip')
    effects=pd.read_csv(output/'tables/e2_prompt_effects.csv')
    checks=0;max_error=0.
    for content in ['cot','answer']:
        baseline=t[t.condition.eq(content+'_as_answer')]
        for context,suffix in [('Thinking','as_cot'),('Scratch','in_scratch'),('Tool-response','in_special')]:
            other=t[t.condition.eq(content+'_'+suffix)]
            keys=['seed','domain','prompt_id','source_record_id','segment_idx']
            joined=other.merge(baseline,on=keys,how='outer',suffixes=('_context','_plain'),validate='one_to_one',indicator=True)
            assert joined['_merge'].eq('both').all()
            assert joined.token_id_context.eq(joined.token_id_plain).all()
            for metric,prefix in [('projection','dot'),('cosine','cos')]:
                for direction in ['assistant','ctrl_1','ctrl_2','null','random']:
                    # Subtract exact paired token scores FIRST, then average.
                    # Authoring script instead subtracts independently aggregated segment means.
                    delta=joined[keys].copy()
                    delta['value']=joined[f'{prefix}_{direction}_context']-joined[f'{prefix}_{direction}_plain']
                    per=delta.groupby(['seed','domain','prompt_id']).value.mean().groupby(['domain','prompt_id']).mean().sort_index()
                    export_name='structured_null' if direction=='null' else direction
                    saved=effects[effects.content.eq(content)&effects.context.eq(context)&effects.metric.eq(metric)&effects.direction.eq(export_name)]
                    saved=saved.set_index(['domain','prompt_id']).value.sort_index()
                    assert per.index.equals(saved.index)
                    error=float(np.max(np.abs(per-saved)));assert error<1e-12
                    max_error=max(max_error,error);checks+=len(saved)
    # Every prompt-level comparison uses 40 independent prompts in this pilot.
    for name in ['e1_effects','e2_effects','e2_factorial_effects','specificity_effects','answer_exclusion_sensitivity']:
        table=pd.read_csv(output/'tables'/f'{name}.csv')
        assert table.n.eq(40).all()
        assert np.isfinite(table[['mean','ci_low','ci_high']].to_numpy()).all()
        assert table.ci_low.le(table.ci_high).all()
        for field in ['direction','control']:
            if field in table:assert table[field].notna().all()
    e1=pd.read_csv(output/'tables/e1_prompt_effects.csv')
    old=pd.read_csv(source/'e1_prompt_contrasts.csv')
    old=old[old.contrast.eq('think_minus_answer')].copy()
    old.direction=old.direction.str.removeprefix('cos_').replace({'null':'structured_null'})
    old=old.groupby(['domain','prompt_id','direction']).value.mean().sort_index()
    new=e1[e1.metric.eq('cosine')].set_index(['domain','prompt_id','direction']).value.sort_index()
    assert old.index.equals(new.index);assert np.allclose(old,new,rtol=0,atol=1e-12)
    figures=manifest['figures'];assert len(figures)==9
    for f in figures:
        assert (output/'figures'/f'{f}.png').stat().st_size>10000
        assert (output/'figures'/f'{f}.pdf').read_bytes().startswith(b'%PDF')
    assert (output/'paper_figures.pdf').read_bytes().startswith(b'%PDF')
    result={'passed':True,'source_hashes_unchanged':True,'e2_prompt_comparisons_checked':checks,
            'e2_max_absolute_error':max_error,'cosine_e1_matches_original':True,'figure_pairs':len(figures),
            'all_effect_interval_counts':40,'missing_control_labels':False}
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir');p.add_argument('output_dir')
    a=p.parse_args();verify(a.run_dir,a.output_dir)
