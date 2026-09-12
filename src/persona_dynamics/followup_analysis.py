"""Paired question-level analysis of the frozen role-susceptibility follow-up."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import bootstrap_summary
from .io import read_jsonl,write_json,write_jsonl


def paired_effects(means,seeds):
    """Retain a question only when every configured seed has both compared arms.

    All role/default and style/role effects compare identical user questions.
    Neither tokens nor repeated seeds become independent observations.
    """
    parts=[];metrics=[c for c in means if c.startswith(('dot_','cos_')) and means[c].notna().any()]
    arms=sorted(means.arm.unique());pairs=[]
    for arm in arms:
        if arm!='default':pairs.append((arm,'default','versus_default'))
        if arm.startswith('style_') and 'role_'+arm[6:] in arms:pairs.append(('role_'+arm[6:],arm,'role_versus_style'))
    for lhs,rhs,comparison in pairs:
        for metric in metrics:
            selected=means[means.arm.isin([lhs,rhs]) & means.segment.isin(['think','answer','answer_skip5'])]
            wide=selected.pivot(index=['seed','prompt_id','domain'],columns=['arm','segment'],values=metric)
            for answer,policy in [('answer','all_content'),('answer_skip5','skip5_sensitivity')]:
                needed=[(lhs,'think'),(rhs,'think'),(lhs,answer),(rhs,answer)]
                if any(k not in wide for k in needed):continue
                complete=wide.dropna(subset=needed).copy()
                counts=complete.reset_index().groupby('prompt_id').seed.nunique()
                ids=counts[counts.eq(len(seeds))].index
                complete=complete[complete.index.get_level_values('prompt_id').isin(ids)]
                for contrast,values in [('think_shift',complete[lhs,'think']-complete[rhs,'think']),
                                        ('answer_shift',complete[lhs,answer]-complete[rhs,answer]),
                                        ('segment_interaction',(complete[lhs,'think']-complete[rhs,'think'])-(complete[lhs,answer]-complete[rhs,answer]))]:
                    rows=values.rename('value').reset_index()
                    rows['arm']=lhs;rows['reference_arm']=rhs;rows['comparison']=comparison
                    rows['metric']=metric;rows['policy']=policy;rows['contrast']=contrast
                    parts.append(rows)
    cols=['seed','prompt_id','domain','value','arm','reference_arm','comparison','metric','policy','contrast']
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)


def summarize(pairs,draws,seed):
    rows=[];keys=['arm','reference_arm','comparison','metric','policy','contrast']
    for values,group in pairs.groupby(keys,sort=True):
        info=dict(zip(keys,values))
        pools=[('pooled',group.groupby('prompt_id',sort=True).value.mean())]
        if info['metric'] in ['dot_assistant','cos_assistant']:
            pools += [(str(s),g.set_index('prompt_id').value.sort_index()) for s,g in group.groupby('seed')]
        for scope,x in pools:
            stats={'n':len(x),'mean':float(x.mean()) if len(x) else None}
            if info['metric'] in ['dot_assistant','cos_assistant']:
                stats=bootstrap_summary(x,draws,seed)
                # Twelve predeclared primary interactions: nine role/default,
                # three role/style. Pointwise intervals remain available too.
                primary=info['metric']=='dot_assistant' and info['policy']=='all_content' and info['contrast']=='segment_interaction' and info['arm'].startswith('role_') and scope=='pooled'
                if primary and len(x)>=2:
                    arr=x.to_numpy();rng=np.random.default_rng(seed)
                    boot=arr[rng.integers(len(arr),size=(10000,len(arr)))].mean(1)
                    lo,hi=np.quantile(boot,[.05/24,1-.05/24])
                    stats.update(familywise_ci_low=float(lo),familywise_ci_high=float(hi),familywise_method='Bonferroni percentile bootstrap, 12 planned interactions, 10000 draws')
            rows.append({**info,'seed_scope':scope,**stats})
    return pd.DataFrame(rows)


def analyze(root):
    root=Path(root);spec=json.loads((root/'config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    req=[r for r in read_jsonl(root/'requests.jsonl') if r['phase']=='main']
    records=[json.loads((root/'transcripts'/(r['record_id']+'.json')).read_text()) for r in req]
    from .followup import validity
    valid=[r for r in records if validity(r)[0]]
    if len(records)!=manifest['expected_main_records']:raise ValueError('Request count mismatch')
    shards=[pd.read_csv(root/'token_shards'/(r['record_id']+'.csv.gz')) for r in valid]
    tokens=pd.concat(shards,ignore_index=True)
    if tokens.duplicated(['record_id','token_idx']).any():raise ValueError('Duplicate measurements')
    metrics=[c for c in tokens if c.startswith(('dot_','cos_')) and tokens[c].notna().any()]
    if not np.isfinite(tokens[metrics].to_numpy()).all():raise ValueError('Non-finite measurements')
    tokens.to_csv(root/'tokens.csv.gz',index=False)
    attrs=pd.concat([pd.read_csv(root/f'attrition-{s}.csv') for s in spec['seeds']],ignore_index=True)
    attrs.to_csv(root/'attrition.csv',index=False)
    keep=['record_id','seed','prompt_id','domain','arm','role']
    means=[]
    for rid,sub in tokens.groupby('record_id',sort=False):
        for segment in ['think','answer','answer_skip5','boundary','answer_first5','think_first50','think_last50']:
            actual='answer' if segment.startswith('answer') else 'think' if segment.startswith('think') else 'boundary'
            part=sub[sub.segment.eq(actual)]
            if segment=='answer_skip5':part=part[part.segment_idx.ge(5)]
            elif segment=='answer_first5':part=part[part.segment_idx.lt(5)]
            elif segment=='think_first50':part=part.nsmallest(50,'segment_idx')
            elif segment=='think_last50':part=part.nlargest(50,'segment_idx')
            if len(part):means.append({**{k:sub.iloc[0][k] for k in keep},'segment':segment,'n_tokens':len(part),
                **{k:float(part[k].mean()) for k in metrics+['resid_norm','centered_norm']}})
    means=pd.DataFrame(means);means.to_csv(root/'segment_means.csv',index=False)
    pairs=paired_effects(means,spec['seeds']);pairs.to_csv(root/'paired_question_effects.csv',index=False)
    effects=summarize(pairs,spec['bootstrap_samples'],spec['bootstrap_seed']);effects.to_csv(root/'effects.csv',index=False)
    plots=make_plots(root,tokens,means,pairs,effects,manifest['synthetic'])
    # Review export is explicitly unannotated. Full outputs include all valid
    # records, including style controls; no sentence heuristic is treated as truth.
    write_jsonl(root/'review_transcripts.jsonl',valid)
    review=root/'review';review.mkdir(exist_ok=True)
    review_rows=[];lines=['# Blind-to-projection full-output review','',
        'These are model outputs. No human annotation has been completed by this export.','']
    ids=[]
    for arm in sorted({r['arm'] for r in valid}):
        pool=sorted([r for r in valid if r['arm']==arm],key=lambda r:(r['prompt_id'],r['seed']))
        seen=set()
        for r in pool:
            if r['prompt_id'] in seen:continue
            seen.add(r['prompt_id']);ids.append(r)
            if len(seen)==3:break
    for r in ids:
        review_rows.append({k:r[k] for k in ['record_id','prompt_id','arm','seed']}|{'reviewer':'','think_role_adoption':'','answer_role_adoption':'','meta_planning':'','refusal_or_role_break':'','notes':''})
        lines += [f"## {r['record_id']} · {r['arm']}",'','Prompt:',r['messages'][-1]['content'],'','Thinking:',r['think_text'],'','Answer:',r['answer_text'],'']
    if not (review/'review.csv').exists():pd.DataFrame(review_rows).to_csv(review/'review.csv',index=False)
    (review/'review.md').write_text('\n'.join(lines))
    write_json(review/'status.json',{'human_review_complete':False,'exported_examples':len(ids),'external_judge_used':False})
    # Equal task/seed means enable later PCA or independent reanalysis without
    # keeping token-level hidden states. No inferential PCA claim is automated.
    summary={'synthetic':manifest['synthetic'],'run_id':manifest['run_id'],'n_requested':len(req),'n_valid':len(valid),
        'n_token_rows':len(tokens),'seeds':spec['seeds'],'arms':sorted(means.arm.unique()),'figures':plots,
        'metric':'raw dot primary; centered cosine sensitivity','validation':json.loads((root/'validation.json').read_text()),
        'human_review_complete':False,'external_judge_used':False,
        'attrition':attrs.groupby(['arm','exclusion_reason'],dropna=False).size().rename('n').reset_index().to_dict('records')}
    write_json(root/'summary.json',summary)
    write_report(root,summary,effects)
    print(json.dumps({k:summary[k] for k in ['run_id','n_requested','n_valid','n_token_rows','synthetic']},indent=2),flush=True)
    return summary


def make_plots(root,tokens,means,pairs,effects,synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    plots=[]
    def save(fig,name):
        if synthetic:fig.suptitle('SYNTHETIC PIPELINE CHECK — not model evidence')
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',dpi=160)
        plt.close(fig);plots.append(f'figures/{name}.png')
    for metric,label in [('dot_assistant','Raw Assistant projection'),('cos_assistant','Centered Assistant cosine')]:
        s=effects[effects.metric.eq(metric)&effects.policy.eq('all_content')&effects.seed_scope.eq('pooled')&effects.comparison.eq('versus_default')&effects.arm.str.startswith('role_')]
        w=s.pivot(index='arm',columns='contrast',values='mean')
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
        allvals=[]
        for arm,row in w.iterrows():
            sx=s[s.arm.eq(arm)&s.contrast.eq('answer_shift')].iloc[0];sy=s[s.arm.eq(arm)&s.contrast.eq('think_shift')].iloc[0]
            if pd.isna(sx.ci_low) or pd.isna(sy.ci_low):continue
            ax.errorbar(row.answer_shift,row.think_shift,xerr=[[max(0,row.answer_shift-sx.ci_low)],[max(0,sx.ci_high-row.answer_shift)]],
                        yerr=[[max(0,row.think_shift-sy.ci_low)],[max(0,sy.ci_high-row.think_shift)]],fmt='o',color='#2962a3',alpha=.7,capsize=2)
            ax.annotate(arm[5:],(row.answer_shift,row.think_shift),xytext=(5,5),textcoords='offset points',fontsize=8)
            allvals.extend([sx.ci_low,sx.ci_high,sy.ci_low,sy.ci_high])
        low=min([0,*allvals]);high=max([0,*allvals]);pad=max(.01,(high-low)*.12)
        ax.plot([low-pad,high+pad],[low-pad,high+pad],color='#666',ls='--',lw=1)
        ax.axhline(0,color='#aaa',lw=.6);ax.axvline(0,color='#aaa',lw=.6)
        ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),xlabel='Role − default shift in answer',ylabel='Role − default shift in thinking',
               title=label+' · paired questions, seeds averaged')
        ax.grid(alpha=.15);save(fig,'01_role_susceptibility_'+metric)
    s=effects[effects.metric.eq('dot_assistant')&effects.policy.eq('all_content')&effects.seed_scope.eq('pooled')&effects.contrast.eq('segment_interaction')&effects.arm.str.startswith('role_')]
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    s=s.sort_values(['comparison','arm']).reset_index(drop=True)
    y=np.arange(len(s));ax.errorbar(s['mean'],y,xerr=[s['mean']-s.ci_low,s.ci_high-s['mean']],fmt='o',color='#2962a3',capsize=3)
    good=s.familywise_ci_low.notna()
    ax.hlines(y[good],s.loc[good,'familywise_ci_low'],s.loc[good,'familywise_ci_high'],color='#aaa',lw=1,zorder=0)
    ax.axvline(0,color='#555',lw=1);ax.set_yticks(y,[r.arm[5:]+(' vs style' if r.comparison=='role_versus_style' else ' vs default') for r in s.itertuples()])
    ax.set(xlabel='Thinking shift − answer shift (raw Assistant projection)',title='Segment interactions · blue 95% pointwise; gray familywise intervals')
    ax.grid(axis='x',alpha=.15);save(fig,'02_segment_interactions')
    arm_order=json.loads((root/'arms.json').read_text());arms=[a['arm'] for a in arm_order]
    grouped=means[means.segment.isin(['think','answer'])].groupby(['arm','segment']).dot_assistant.mean()
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained');x=np.arange(len(arms))
    for segment,offset,color in [('think',-.18,'#2962a3'),('answer',.18,'#b58a28')]:
        ax.bar(x+offset,[grouped.get((a,segment),np.nan) for a in arms],width=.35,color=color,label=segment.capitalize())
    ax.set_xticks(x,[a.replace('role_','').replace('style_','style: ') for a in arms],rotation=40,ha='right')
    ax.set(ylabel='Mean raw Assistant projection',title='Descriptive levels · equal transcript weight; use paired effects for inference');ax.legend();save(fig,'03_arm_levels')
    # Fixed 20-token bins only for visualization, never replacing token data.
    cols=['dot_assistant','dot_ctrl_1','dot_ctrl_2','dot_null']
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained',sharey=True)
    role_arms=[a for a in arms if a.startswith('role_')]
    for ax,arm in zip(axes.flat,role_arms):
        for segment,offset in [('think',0),('answer',1)]:
            sub=tokens[tokens.arm.eq(arm)&tokens.segment.eq(segment)].copy();sub['bin']=np.minimum((sub.norm_pos*20).astype(int),19)
            curves=sub.groupby(['prompt_id','seed','bin'])[cols].mean().groupby('bin').mean()
            for col,color in zip(cols,['#2962a3','#b58a28','#c76536','#73823f']):ax.plot(offset+(curves.index+.5)/20,curves[col],color=color,lw=1.5 if col=='dot_assistant' else 1,label=col[4:] if segment=='think' else None)
        ax.axvline(1,color='#777',lw=.8);ax.set(title=arm[5:],xlabel='Thinking 0–1 | answer 1–2');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.supylabel('Raw projection · descriptive traces');save(fig,'04_role_traces')
    return plots


def write_report(root,summary,effects):
    primary=effects[effects.metric.eq('dot_assistant')&effects.policy.eq('all_content')&effects.seed_scope.eq('pooled')&effects.contrast.eq('segment_interaction')&effects.arm.str.startswith('role_')]
    lines=['# Complementary role-susceptibility experiment','',
        '**SYNTHETIC PIPELINE CHECK — not model evidence.**' if summary['synthetic'] else 'Actual Qwen3-32B generation and teacher-forced measurements.', '',
        f"Requested {summary['n_requested']} responses; {summary['n_valid']} completed valid transcripts; {summary['n_token_rows']:,} token rows; seeds {summary['seeds']}.",'',
        'Raw linear projections are primary; centered cosine is a separately reported sensitivity. Every content token is collected. Main summaries include all answer tokens; skip-first-five is a sensitivity. Truncated generations remain in attrition; inference is conditional on complete pairs across both seeds.', '',
        '## Primary segment interactions','',
        'Effect = (role − reference in thinking) − (role − reference in answer). Zero is equal susceptibility, not proof of equivalence. Positive values mean a more positive/less negative thinking shift than answer shift. Each interval resamples questions after averaging seeds, never tokens.', '',
        '| Role | Reference | Paired questions | Effect | 95% pointwise CI | Familywise CI, 12 tests |','|---|---|---:|---:|---|---|']
    def f(x):return 'NA' if pd.isna(x) else f'{x:.4f}'
    for r in primary.itertuples():lines.append(f'| {r.arm[5:]} | {r.reference_arm} | {r.n} | {f(r.mean)} | [{f(r.ci_low)}, {f(r.ci_high)}] | [{f(r.familywise_ci_low)}, {f(r.familywise_ci_high)}] |')
    lines += ['', '## Interpretation limits','',
        'The nine roles are fixed, purposively selected interventions. Their own vector readouts are expected positive readouts, not unrelated negative controls when inducing that role. Style controls explicitly preserve assistant identity and therefore differ in more than surface wording. Role/default differences are not causal interventions on an isolated latent persona.', '',
        'Identity questions, social questions and reasoning tasks are mixed in the primary matched set. Domain-specific responses and error cases should be inspected. Raw units are not calibrated persona probabilities. The parent E0 check was reused and fresh numerical references replayed, but that is not a behavioral calibration of these new thinking-mode roles.', '',
        '**No human behavioral labels or external judge results have been fabricated.** The review packet is unannotated. Geometry alone cannot establish role adoption, neutrality, a separate agent, or equivalence. Role instruction adherence, refusals, task accuracy and surface style are important alternative explanations.', '',
        'All role, seed, readout and answer-token sensitivity estimates are in `effects.csv`; prompt-level paired observations are in `paired_question_effects.csv`. Full exact generations, token scalars, calibration provenance and segment mean vectors are saved.', '',
        '## Attrition','', '| Arm | Reason | Count |','|---|---|---:|']
    for r in summary['attrition']:lines.append(f"| {r['arm']} | {r['exclusion_reason']} | {r['n']} |")
    lines+=['','## Figures','']
    for name in summary['figures']:lines.extend([f'![{Path(name).stem}]({name})',''])
    (root/'results.md').write_text('\n'.join(lines))
