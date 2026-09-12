"""Paper figures and prompt-level reanalysis of frozen per-token projections.

No inference; original run stays immutable. Main E1 uses the original answer
exclusion. E2 compares exact content in each wrapper against its plain context.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from persona_dynamics.analysis import bootstrap_summary
from persona_dynamics.io import file_hash

DIRS=['assistant','ctrl_1','ctrl_2','null','random']
LABELS=['Assistant','Skeptic','Judge','Structured null','Random']
COLORS=['#245A91','#B18325','#C96636','#768339','#A65586']
DOMAINS=['advice','math']
SEED=2026
KEYS=['seed','domain','prompt_id']
CAPTIONS={}


def write_table(table,path):
    """Avoid the literal CSV label 'null', parsed as missing by common readers."""
    table=table.copy()
    for column in ['direction','control']:
        if column in table:table[column]=table[column].replace({'null':'structured_null'})
    table.to_csv(path,index=False)


def validate_tokens(t):
    if t.model.nunique()!=1 or t.layer.nunique()!=1:
        raise ValueError('One model/layer per figure collection; do not pool incompatible rulers')
    if t.duplicated(['record_id','token_idx']).any():raise ValueError('Duplicate token observations')
    if not t.primary_include.isin([True,False]).all():raise ValueError('Invalid inclusion flags')
    cols=[f'{prefix}_{d}' for prefix in ['dot','cos'] for d in DIRS]+['resid_norm','centered_norm']
    if not np.isfinite(t[cols].to_numpy()).all():raise ValueError('Non-finite measurements')
    if set(t.domain)!=set(DOMAINS):raise ValueError('This figure layout requires advice and math')
    e2=t[t.experiment.eq('E2')]
    for content in ['cot','answer']:
        cells=[content+'_'+c for c in ['as_cot','as_answer','in_scratch','in_special']]
        sub=e2[e2.condition.isin(cells)]
        keys=KEYS+['source_record_id','segment_idx']
        grouped=sub.groupby(keys)
        if not grouped.condition.nunique().eq(4).all() or not grouped.size().eq(4).all():
            raise ValueError('Incomplete or duplicated E2 content/context matching')
        if not grouped.token_id.nunique().eq(1).all():raise ValueError('E2 content token IDs differ')
        if not sub.primary_include.all():raise ValueError('E2 must measure every matched content token')
    return {'token_rows':len(t),'records':int(t.record_id.nunique()),
            'matched_e2_content_positions':int(len(e2)//4)}


def segment_means(t, all_answers=False):
    mask=t.primary_include if not all_answers else t.segment.isin(['think','answer'])
    subset=t[mask & t.segment.isin(['think','answer'])]
    cols=[f'{p}_{d}' for p in ['dot','cos'] for d in DIRS]+['resid_norm','centered_norm']
    return subset.groupby(KEYS+['experiment','condition','segment'])[cols].mean().reset_index()


def e1_effects(means):
    natural=means[means.experiment.eq('E1') & means.condition.eq('natural_think')]
    rows=[]
    for metric,prefix in [('projection','dot'),('cosine','cos')]:
        for direction in DIRS:
            w=natural.pivot(index=KEYS,columns='segment',values=f'{prefix}_{direction}')
            if not {'think','answer'}.issubset(w) or w[['think','answer']].isna().any().any():
                raise ValueError('Incomplete natural segment pair')
            delta=(w.think-w.answer).rename('value').reset_index()
            delta=delta.groupby(['domain','prompt_id']).value.mean().reset_index()
            rows.append(delta.assign(metric=metric,direction=direction))
    return pd.concat(rows,ignore_index=True)


def e2_effects(means):
    e2=means[means.experiment.eq('E2')]
    rows=[]; factorial=[]
    for metric,prefix in [('projection','dot'),('cosine','cos')]:
        for direction in DIRS:
            w=e2.pivot(index=KEYS,columns='condition',values=f'{prefix}_{direction}')
            if w.isna().any().any():raise ValueError('Incomplete E2 pair')
            for content in ['cot','answer']:
                for context,condition in [('Thinking','as_cot'),('Scratch','in_scratch'),('Tool-response','in_special')]:
                    delta=(w[f'{content}_{condition}']-w[f'{content}_as_answer']).rename('value').reset_index()
                    delta=delta.groupby(['domain','prompt_id']).value.mean().reset_index()
                    rows.append(delta.assign(metric=metric,direction=direction,content=content,context=context))
            ct,cp,at,ap=(w[c] for c in ['cot_as_cot','cot_as_answer','answer_as_cot','answer_as_answer'])
            for contrast,value in [('context_main',((ct-cp)+(at-ap))/2),('content_main',((ct-at)+(cp-ap))/2),
                                   ('interaction',(ct-cp)-(at-ap))]:
                delta=value.rename('value').reset_index().groupby(['domain','prompt_id']).value.mean().reset_index()
                factorial.append(delta.assign(metric=metric,direction=direction,contrast=contrast))
    return pd.concat(rows,ignore_index=True),pd.concat(factorial,ignore_index=True)


def summarize(frame, keys, draws):
    output=[]
    for key,sub in frame.groupby(keys,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        sub=sub.sort_values('prompt_id')
        if sub.prompt_id.duplicated().any():raise ValueError('Repeated prompt in statistical sample')
        output.append(dict(zip(keys,key),**bootstrap_summary(sub.value,draws,SEED)))
    return pd.DataFrame(output)


def specificity(effects):
    rows=[]
    for (domain,metric),sub in effects.groupby(['domain','metric']):
        w=sub.pivot(index='prompt_id',columns='direction',values='value')
        for d in DIRS[1:]:
            delta=(w.assistant.abs()-w[d].abs()).rename('value').reset_index()
            rows.append(delta.assign(domain=domain,metric=metric,control=d))
    return pd.concat(rows,ignore_index=True)


def limits(values,pad=.08,include_zero=True):
    a=np.asarray(values,float);lo=float(a.min());hi=float(a.max())
    if include_zero:lo,hi=min(0,lo),max(0,hi)
    margin=max(hi-lo,1e-4)*pad
    return lo-margin,hi+margin


def setup_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#89929D','axes.labelcolor':'#27333E',
        'text.color':'#27333E','xtick.color':'#46525D','ytick.color':'#46525D','axes.axisbelow':True,
        'grid.color':'#E6E9ED','grid.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})


def decorate(ax):
    ax.grid(axis='y');ax.axhline(0,color='#67727D',lw=.8,zorder=1)


def dots_interval(ax,x,sub,summary,color):
    sub=sub.sort_values('prompt_id')
    jitter=np.random.default_rng(SEED).uniform(-.15,.15,len(sub))
    ax.scatter(x+jitter,sub.value,s=15,color=color,alpha=.38,linewidths=0,zorder=2,rasterized=True)
    estimate_x=x+.25
    ax.vlines(estimate_x,summary.ci_low,summary.ci_high,color='#202D39',lw=1.8,zorder=4)
    ax.hlines([summary.ci_low,summary.ci_high],estimate_x-.045,estimate_x+.045,color='#202D39',lw=1,zorder=4)
    ax.scatter([estimate_x],[summary['mean']],marker='D',s=22,facecolor=color,edgecolor='white',linewidth=.5,zorder=5)


def estimate_legend(fig):
    fig.legend(handles=[Line2D([0],[0],marker='o',ls='',color='#7F8D9A',alpha=.6,label='Individual prompt'),
                        Line2D([0],[0],marker='D',ls='-',color='#26343F',markersize=4,label='Mean and 95% prompt-bootstrap CI')],
               loc='outside lower center',ncol=2,frameon=False,fontsize=9)


def save(fig,out,name,title,caption,pdf):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    pdf.savefig(fig,bbox_inches='tight',facecolor='white')
    CAPTIONS[name]={'title':title,'caption':caption}
    plt.close(fig)


def main_e1(effects,stats,out,pdf):
    data=effects[effects.metric.eq('projection')];ys=limits(data.value)
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.7),sharey=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        for i,(d,color) in enumerate(zip(DIRS,COLORS)):
            sub=data[data.domain.eq(domain)&data.direction.eq(d)]
            s=stats[stats.metric.eq('projection')&stats.domain.eq(domain)&stats.direction.eq(d)].iloc[0]
            dots_interval(ax,i,sub,s,color)
        decorate(ax);ax.set(title=f'{domain.capitalize()} · {sub.prompt_id.nunique()} prompts',ylim=ys)
        ax.set_xticks(range(5),['Assistant','Skeptic','Judge','Structured\nnull','Random'])
    axes[0].set_ylabel('Thinking − answer linear projection')
    estimate_legend(fig)
    fig.suptitle('1  Natural thinking–answer differences',fontsize=15,weight='bold')
    save(fig,out,'01_natural_gaps','Natural thinking–answer differences',
      'Each dot is one retained prompt. Tokens are averaged within each segment; paired seed effects are averaged within prompt. '
      'Diamonds show means; vertical lines show 95% paired prompt-bootstrap intervals (10,000 resamples). '
      'Natural answer means exclude the first five content tokens, as in the original run. All five unit directions share one scale; '
      'the null and role controls are not independent persona labels. Prompt-level values: tables/e1_prompt_effects.csv.',pdf)


def main_e2(effects,stats,out,pdf):
    data=effects[effects.metric.eq('projection')];ys=limits(data.value)
    fig,axes=plt.subplots(5,2,figsize=(9.6,11),sharex=True,sharey=True,layout='constrained')
    contexts=['Thinking','Scratch','Tool-response']
    for i,(d,label,color) in enumerate(zip(DIRS,LABELS,COLORS)):
        for j,content in enumerate(['cot','answer']):
            ax=axes[i,j]
            for k,context in enumerate(contexts):
                sub=data[data.direction.eq(d)&data.content.eq(content)&data.context.eq(context)]
                s=stats[stats.metric.eq('projection')&stats.direction.eq(d)&stats.content.eq(content)&stats.context.eq(context)].iloc[0]
                dots_interval(ax,k,sub,s,color)
            decorate(ax);ax.set_ylim(ys);ax.text(.98,.90,label,transform=ax.transAxes,ha='right',va='top',color=color,weight='bold')
            ax.set_xticks(range(3),contexts)
            if i==0:ax.set_title('Original thinking content' if content=='cot' else 'Original answer content')
    fig.suptitle('2  Same content under different prefixes',fontsize=15,weight='bold')
    fig.supylabel('Context − plain-answer linear projection')
    estimate_legend(fig)
    save(fig,out,'02_matched_context_effects','Same content under different prefixes',
      'Every context is compared with the exact same token IDs in plain-answer context, within each of 40 advice prompts. '
      'Thinking and answer source content both come from the original thinking-mode generation. All copied content tokens are included. '
      'Scratch and tool-response blocks are in answer position after an empty thinking block. Dots are prompt-level differences; '
      'diamonds and intervals are means and 95% paired prompt-bootstrap CIs (10,000 draws). All ten panels share one scale. '
      'Tool-response denotes template-supported atomic added tokens, not a real tool invocation. Prefixes also change whitespace and position; '
      'this does not isolate a latent persona intervention. Source: tables/e2_prompt_effects.csv.',pdf)


def trajectory_data(t,draws):
    data=t[t.experiment.eq('E1')&t.condition.eq('natural_think')&t.primary_include&t.segment.isin(['think','answer'])].copy()
    data['bin']=np.minimum((data.norm_pos*20).astype(int),19)
    cols=['dot_'+d for d in DIRS]
    grain=['domain','prompt_id','segment','bin']
    per=data.groupby(['seed']+grain)[cols].mean().groupby(grain).mean().reset_index()
    long=per.melt(id_vars=grain,var_name='direction',value_name='value');long.direction=long.direction.str.removeprefix('dot_')
    curves=summarize(long,['domain','segment','bin','direction'],draws)
    return long,curves


def main_trajectory(t,means,out,pdf,draws):
    points,curves=trajectory_data(t,draws);write_table(points,out/'tables/trajectory_prompt_bins.csv')
    write_table(curves,out/'tables/trajectory_summary.csv')
    refs=means[means.experiment.eq('E1')&means.condition.eq('natural_nothink')&means.segment.eq('answer')]
    refs=refs.groupby(['domain','prompt_id'])[['dot_'+d for d in DIRS]].mean().reset_index()
    refs.to_csv(out/'tables/nothink_reference_levels.csv',index=False)
    fig,axes=plt.subplots(5,2,figsize=(10.2,11.2),sharex=True,sharey='row',layout='constrained')
    for i,(d,label,color) in enumerate(zip(DIRS,LABELS,COLORS)):
        ylim=limits(np.r_[points.loc[points.direction.eq(d),'value'],refs['dot_'+d]],include_zero=False)
        for j,domain in enumerate(DOMAINS):
            ax=axes[i,j]
            for segment,offset in [('think',0),('answer',1.15)]:
                sub=points[points.domain.eq(domain)&points.direction.eq(d)&points.segment.eq(segment)]
                for _,p in sub.groupby('prompt_id'):
                    ax.plot((p.bin+.5)/20+offset,p.value,color=color,alpha=.09,lw=.6,rasterized=True)
                c=curves[curves.domain.eq(domain)&curves.direction.eq(d)&curves.segment.eq(segment)].sort_values('bin')
                x=(c.bin.to_numpy()+.5)/20+offset
                ax.fill_between(x,c.ci_low,c.ci_high,color=color,alpha=.18,lw=0)
                ax.plot(x,c['mean'],color=color,lw=1.8)
            ax.axhline(refs[refs.domain.eq(domain)]['dot_'+d].mean(),color='#414B56',ls='--',lw=.9)
            ax.axvline(1.075,color='#89929D',lw=.8);ax.grid(axis='y');ax.set_ylim(ylim)
            ax.text(.025,.94,label,transform=ax.transAxes,va='top',color=color,weight='bold')
            if i==0:ax.set_title(domain.capitalize())
            ax.set_xticks([.025,.975,1.175,2.125],['0','1','0','1'])
            if i==4:ax.set_xlabel('Thinking progress        Answer progress')
    handles=[Line2D([0],[0],color='#4B5661',lw=1.8,label='Mean ± 95% pointwise CI'),
             Line2D([0],[0],color='#4B5661',lw=.7,alpha=.3,label='Individual prompt (20 bins)'),
             Line2D([0],[0],color='#414B56',ls='--',lw=.9,label='No-thinking answer mean')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    fig.suptitle('3  Projection through thinking and answering\nShared scales across domains; different scales between directions',fontsize=13,weight='bold')
    fig.supylabel('Linear projection onto the unit direction')
    save(fig,out,'03_token_trajectories','Projection through thinking and answering',
      'Each thin trace shows one prompt after within-segment binning into 20 equal normalized-position bins; it is not an unsmoothed token trace. '
      'Thick lines and bands are equal-prompt means and 95% pointwise prompt-bootstrap intervals (1,000 draws). '
      'The separator denotes the transition across </think>; its activation and formatting-only tokens are not plotted here. '
      'The dashed level is the same direction’s mean no-thinking answer projection on the paired cohort. '
      'Vertical scales are shared between domains within each direction, but differ between rows to reveal trajectories; '
      'compare magnitudes across directions in Figures 1 and 2, which use common scales. No data were rescaled or centered. '
      'Normalized positions align unequal lengths, not equal token counts. First-five exclusion can leave early answer bins empty; '
      'the n column in tables/trajectory_summary.csv records contributing prompts per bin. No line connects the two segments.',pdf)
    return curves


def appendix_e0(root,out,pdf):
    t=pd.read_csv(root/'e0_tokens.csv.gz',low_memory=False,float_precision='round_trip')
    m=t[t.primary_include & t.segment.eq('answer')].groupby(['seed','prompt_id','role']).dot_assistant.mean().groupby(['prompt_id','role']).mean().unstack()
    roles=sorted(m.columns.drop('default'));delta=m[roles].rsub(m['default'],axis=0)
    delta.to_csv(out/'tables/e0_raw_default_minus_role.csv')
    fig,ax=plt.subplots(figsize=(10.5,4.7),layout='constrained')
    for i,role in enumerate(roles):
        values=delta[role].to_numpy();ax.scatter(i+np.linspace(-.10,.10,len(values)),values,s=35,color=COLORS[0],alpha=.6)
        ax.scatter(i,values.mean(),marker='D',s=45,color='#26333D')
    decorate(ax);ax.set_xticks(range(len(roles)),[r.capitalize() for r in roles],rotation=25,ha='right')
    ax.set_ylabel('Default − role linear projection');ax.set_title('A1  Axis validation on held-out questions',loc='left',weight='bold')
    save(fig,out,'A1_axis_validation','Axis validation on held-out questions',
      'Each point is one of three held-out E0 questions, paired between default and the named role; diamonds are three-question means. '
      'These raw projections are a post hoc companion diagnostic. The original gate used centered cosine and passed before E1/E2. '
      'No new threshold or role selection was tuned here. No small-sample precision claim is inferred from three points.',pdf)


def appendix_metrics(effects,out,pdf):
    a=effects[effects.direction.eq('assistant')].pivot(index=['domain','prompt_id'],columns='metric',values='value').reset_index()
    a['sign_disagreement']=np.sign(a.projection)!=np.sign(a.cosine);a.to_csv(out/'tables/normalization_prompt_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.7),sharex=True,sharey=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        s=a[a.domain.eq(domain)];ax.scatter(s.projection,s.cosine,s=30,color=COLORS[0],alpha=.7,edgecolor='white',linewidth=.4)
        ax.axvline(0,color='#67727D',lw=.8);decorate(ax)
        ax.set(title=f'{domain.capitalize()} · {s.sign_disagreement.sum()}/{len(s)} signs differ',xlabel='Raw projection gap: thinking − answer')
    axes[0].set_ylabel('Centered cosine gap: thinking − answer')
    fig.suptitle('A2  Dependence on normalization',fontsize=15,weight='bold')
    save(fig,out,'A2_raw_vs_cosine','Dependence on normalization',
      'Each point is one prompt’s Assistant-axis gap under both metrics using identical segment inclusion and equal-seed weighting. '
      'Zero lines divide same-sign and opposite-sign quadrants. The axes have different units, so an identity line would be misleading. '
      'The raw metric is post hoc relative to the original cosine analysis; disagreements remain visible.',pdf)
    return a


def appendix_boundary(t,out,pdf,draws):
    data=t[t.experiment.eq('E1')&t.condition.eq('natural_think')&t.pos_from_boundary.between(-30,30)].copy()
    cols=['dot_'+d for d in DIRS];grain=['domain','prompt_id','pos_from_boundary']
    per=data.groupby(['seed']+grain)[cols].mean().groupby(grain).mean().reset_index()
    long=per.melt(id_vars=grain,var_name='direction',value_name='value');long.direction=long.direction.str.removeprefix('dot_')
    curve=summarize(long,['domain','pos_from_boundary','direction'],draws)
    write_table(curve,out/'tables/boundary_summary.csv');write_table(long,out/'tables/boundary_prompt_points.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.8),sharey=True,sharex=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        for d,label,color in zip(DIRS,LABELS,COLORS):
            s=curve[curve.domain.eq(domain)&curve.direction.eq(d)].sort_values('pos_from_boundary')
            ax.plot(s.pos_from_boundary,s['mean'],color=color,lw=2 if d=='assistant' else 1.2,label=label,
                    ls='-' if d=='assistant' else '--')
            ax.fill_between(s.pos_from_boundary,s.ci_low,s.ci_high,color=color,alpha=.1)
        first=data[data.domain.eq(domain)&data.segment.eq('answer')&data.segment_idx.lt(5)].pos_from_boundary
        ax.axvspan(first.min()-.5,first.max()+.5,color='#929AA3',alpha=.15)
        ax.axvline(0,color='#313B44',lw=.9);ax.grid(axis='y');ax.set(title=domain.capitalize(),xlabel='Token offset from </think>',xlim=(-30,30))
    axes[0].set_ylabel('Linear projection')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle('A3  Boundary and first answer tokens',fontsize=15,weight='bold')
    save(fig,out,'A3_boundary_detail','Boundary and first answer tokens',
      'Offset zero is the closing thinking token. Shading covers the first five answer-content token offsets (here +2 through +6); '
      'formatting-only positions without saved activations are not observations, and line segments simply join observed positions. '
      'All saved boundary/first-five rows are included in this diagnostic regardless of primary exclusion. '
      'Lines and bands show equal-prompt means and 95% pointwise intervals (1,000 resamples). All five controls share the same scale.',pdf)


def appendix_length(t,effects,out,pdf):
    source=t[t.experiment.eq('E1')&t.condition.eq('natural_think')]
    lengths=source[source.segment.eq('think')].groupby(KEYS).size().rename('thinking_tokens').groupby(['domain','prompt_id']).mean()
    norm=source[source.primary_include].groupby(KEYS+['segment']).resid_norm.mean().unstack()
    norm=(norm.think-norm.answer).rename('norm_gap').groupby(['domain','prompt_id']).mean()
    points=effects[effects.metric.eq('projection')&effects.direction.eq('assistant')].merge(lengths.reset_index()).merge(norm.reset_index())
    points.to_csv(out/'tables/length_norm_prompt_diagnostics.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10.2,7),sharey=True,sharex='row',layout='constrained')
    for j,domain in enumerate(DOMAINS):
        s=points[points.domain.eq(domain)]
        for i,(x,label) in enumerate([('thinking_tokens','Thinking length (tokens)'),('norm_gap','Mean residual norm: thinking − answer')]):
            ax=axes[i,j];ax.scatter(s[x],s.value,s=28,color=COLORS[0],alpha=.7,edgecolor='white',linewidth=.4)
            decorate(ax);ax.set_xlabel(label)
            if i==0:ax.set_title(domain.capitalize())
            if j==0:ax.set_ylabel('Assistant raw projection gap')
    fig.suptitle('A4  Length and activation-magnitude diagnostics',fontsize=15,weight='bold')
    save(fig,out,'A4_length_and_norm','Length and activation-magnitude diagnostics',
      'Each point is one retained prompt. Vertical coordinates are E1 raw Assistant gaps; horizontal coordinates are thinking content length '
      'or the difference in mean uncentered activation norm. These are descriptive associations on a completion-filtered sample, '
      'not adjusted effect estimates or causal explanations. All panels use the same vertical scale; horizontal units differ by row.',pdf)


def appendix_attrition(root,out,pdf):
    cohort=json.loads((root/'cohort.json').read_text());a=pd.read_csv(root/'attrition.csv');rows=[]
    fig,axes=plt.subplots(1,2,figsize=(9.3,4.4),sharey=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        c=next(c for c in cohort if c['domain']==domain);values=[c['candidates'],c['eligible_all_seeds'],c['selected']]
        ax.bar(range(3),values,color=['#C6CDD5','#7F9CB9',COLORS[0]],width=.6)
        for i,v in enumerate(values):ax.text(i,v+2,str(v),ha='center',weight='bold')
        ax.set_xticks(range(3),['Candidates','Eligible pairs','Retained']);ax.set(title=domain.capitalize(),ylim=(0,max(x['candidates'] for x in cohort)*1.16));ax.grid(axis='y')
        rows.append(dict(domain=domain,candidates=values[0],eligible_pairs=values[1],retained=values[2],
                         length_terminated=int(a[a.domain.eq(domain)&a.finish_reason.eq('length')].shape[0])))
    axes[0].set_ylabel('Prompts');fig.suptitle('A5  Cohort selection and attrition',fontsize=15,weight='bold')
    pd.DataFrame(rows).to_csv(out/'tables/attrition_counts.csv',index=False)
    save(fig,out,'A5_attrition','Cohort selection and attrition',
      'Advice: 50 candidates → 45 eligible pairs → first 40 retained. Math: 100 → 79 → first 40. '
      'Five advice and 21 math thinking generations hit the 2,048-token cap; 15 math truncations also had incomplete thinking blocks. '
      'Math was extended from 50 to 100 fixed-order candidates because the first 50 yielded only 36 pairs. No effect-size-based selection. '
      'All no-thinking candidates were eligible. Counts refer to prompts, not measured tokens or E2 transplant cells.',pdf)


def appendix_sensitivity(effects,all_effects,out,pdf,draws):
    data=pd.concat([effects.assign(policy='Exclude first five'),all_effects.assign(policy='Include all answer content')])
    data=data[data.metric.eq('projection')]
    stats=summarize(data,['domain','direction','policy'],draws);write_table(stats,out/'tables/answer_exclusion_sensitivity.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.7),sharey=True,layout='constrained')
    for ax,domain in zip(axes,DOMAINS):
        for j,(policy,marker,offset) in enumerate([('Exclude first five','D',-.1),('Include all answer content','o',.1)]):
            for i,(d,color) in enumerate(zip(DIRS,COLORS)):
                s=stats[stats.domain.eq(domain)&stats.direction.eq(d)&stats.policy.eq(policy)].iloc[0]
                ax.vlines(i+offset,s.ci_low,s.ci_high,color=color,lw=1.4)
                ax.scatter(i+offset,s['mean'],marker=marker,color=color if j==0 else 'white',edgecolor=color,s=36,zorder=3)
        decorate(ax);ax.set_title(domain.capitalize());ax.set_xticks(range(5),['Assistant','Skeptic','Judge','Structured\nnull','Random'])
    axes[0].set_ylabel('Thinking − answer linear projection')
    fig.legend(handles=[Line2D([0],[0],marker='D',color='#475666',ls='',label='Exclude first five'),
                        Line2D([0],[0],marker='o',color='#475666',mfc='white',ls='',label='Include all answer content')],
               loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('A6  Sensitivity to answer-token exclusion',fontsize=15,weight='bold')
    save(fig,out,'A6_answer_exclusion','Sensitivity to answer-token exclusion',
      'Paired prompt means and 95% prompt-bootstrap intervals (10,000 draws) with the first-five rule applied or removed. '
      'This reuses exactly the retained cohort; it is not a rerun of candidate eligibility under a different rule. '
      'The thinking boundary and formatting-only tokens remain excluded in both policies.',pdf)


def build(root,out,draws=10000,curve_draws=1000):
    root,out=Path(root).resolve(),Path(out).resolve()
    if out==root or root in out.parents:raise ValueError('Output must be outside frozen run')
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'tables').mkdir()
    t=pd.read_csv(root/'tokens.csv.gz',low_memory=False,float_precision='round_trip')
    checks=validate_tokens(t)
    means=segment_means(t);means.to_csv(out/'tables/segment_readouts.csv',index=False)
    e1=e1_effects(means);e2,factorial=e2_effects(means);spec=specificity(e1)
    e1s=summarize(e1,['domain','metric','direction'],draws)
    e2s=summarize(e2,['domain','metric','direction','content','context'],draws)
    fs=summarize(factorial,['domain','metric','direction','contrast'],draws)
    ss=summarize(spec,['domain','metric','control'],draws)
    for name,table in [('e1_prompt_effects',e1),('e1_effects',e1s),('e2_prompt_effects',e2),('e2_effects',e2s),
                       ('e2_factorial_prompt_effects',factorial),('e2_factorial_effects',fs),('specificity_prompt_effects',spec),('specificity_effects',ss)]:
        write_table(table,out/'tables'/f'{name}.csv')
    setup_style()
    with PdfPages(out/'paper_figures.pdf') as pdf:
        main_e1(e1,e1s,out,pdf);main_e2(e2,e2s,out,pdf)
        curves=main_trajectory(t,means,out,pdf,curve_draws)
        appendix_e0(root,out,pdf);comparison=appendix_metrics(e1,out,pdf)
        appendix_boundary(t,out,pdf,curve_draws);appendix_length(t,e1,out,pdf);appendix_attrition(root,out,pdf)
        appendix_sensitivity(e1,e1_effects(segment_means(t,all_answers=True)),out,pdf,draws)
    # Independent arithmetic against saved token values; no reuse of means or effect helpers.
    errors=[]
    for row in e1.itertuples():
        p=t[t.prompt_id.eq(row.prompt_id)&t.condition.eq('natural_think')&t.experiment.eq('E1')&t.primary_include]
        col=('dot_' if row.metric=='projection' else 'cos_')+row.direction
        values=[]
        for _,sub in p.groupby('seed'):
            values.append(np.mean(sub[sub.segment.eq('think')][col].to_numpy())-np.mean(sub[sub.segment.eq('answer')][col].to_numpy()))
        errors.append(abs(np.mean(values)-row.value))
    assert max(errors)<1e-12
    raw=e1s[e1s.metric.eq('projection')&e1s.direction.eq('assistant')]
    # Independent multinomial bootstrap equivalent checks point estimates and CIs
    # within Monte Carlo tolerance without invoking the production bootstrap.
    bootstrap_checks=[]
    for domain in DOMAINS:
        x=e1[e1.metric.eq('projection')&e1.direction.eq('assistant')&e1.domain.eq(domain)].sort_values('prompt_id').value.to_numpy()
        w=np.random.default_rng(91).multinomial(len(x),np.ones(len(x))/len(x),size=50000)
        ci=np.quantile(w@x/len(x),[.025,.975]);saved=raw[raw.domain.eq(domain)].iloc[0]
        tolerance=.12*max(float(np.std(x,ddof=1)/np.sqrt(len(x))),1e-12)
        assert max(abs(ci-np.array([saved.ci_low,saved.ci_high])))<tolerance
        bootstrap_checks.append({'domain':domain,'independent_50000_ci':ci.tolist(),'tolerance':tolerance})
    facts={'source_run':str(root),'source_run_id':json.loads((root/'manifest.json').read_text())['run_id'],
           'source_hashes':{n:file_hash(root/n) for n in ['tokens.csv.gz','directions.npz','cohort.json','attrition.csv','e0_tokens.csv.gz']},
           'script_sha256':file_hash(__file__),'bootstrap_source_sha256':file_hash(Path(__import__('persona_dynamics.analysis',fromlist=['']).__file__)),
           'bootstrap_draws':draws,'curve_draws':curve_draws,'seed':SEED,'seeds':sorted(t.seed.unique().tolist()),
           'table_direction_names':{'structured_null':'dot_null / cos_null in the original token data'},
           'model':t.model.iloc[0],'layer':int(t.layer.iloc[0]),'checks':checks,'e1_independent_max_error':max(errors),
           'independent_bootstrap_checks':bootstrap_checks,'figures':list(CAPTIONS),'post_hoc':True,
           'trajectory_prompt_coverage_range':[int(curves.n.min()),int(curves.n.max())],
           'normalization_sign_disagreement':comparison.groupby('domain').sign_disagreement.sum().to_dict(),
           'limitations':['One generation seed in this run; CIs concern prompts, not seed/model populations.',
               'Completion and minimum-thinking-length filtered convenience sample; user clustering unmodeled.',
               'Pointwise exploratory intervals, no multiple-comparison adjustment.',
               'Raw projection chosen after original cosine results; both measures retained.',
               'E3/E4 and calibrated behavioral validation unavailable.']}
    (out/'manifest.json').write_text(json.dumps(facts,indent=2)+'\n')
    (out/'captions.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n')
    write_report(out,raw,e2s,ss,fs,facts)
    print(raw.to_string(index=False));print('Created',len(CAPTIONS),'figures in',out)
    return facts


def write_report(out,raw,e2s,ss,fs,facts):
    lines=['# Raw-projection figures and reanalysis','',
      'Post hoc reanalysis of frozen Qwen3-32B E0/E1/E2 measurements. No model inference was rerun. '
      'The original cosine report and raw token files remain unchanged. Raw projection means h_t · a for a fixed unit direction; '
      'there is no centering or activation-norm division. Aggregation occurs only here in post-processing.','',
      '[Download all nine figures as one PDF](paper_figures.pdf). Individual PDFs, PNGs, captions and source tables are below.','',
      '## Results','', '| Domain | Assistant thinking − answer projection | 95% CI | Prompts |', '|---|---:|---|---:|']
    for r in raw.itertuples():lines.append(f'| {r.domain.capitalize()} | {r.mean:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.n} |')
    lines += ['', 'These scores are activation coordinates, not probabilities or calibrated persona quantities. '
              'Specificity is evaluated by the paired per-prompt difference |Assistant gap| − |control gap|:','',
              '| Domain | Control | Assistant minus control absolute-gap advantage | 95% CI |', '|---|---|---:|---|']
    for r in ss[ss.metric.eq('projection')].itertuples():
        lines.append(f'| {r.domain.capitalize()} | {LABELS[DIRS.index(r.control)]} | {r.mean:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |')
    lines += ['', 'Negative entries favor the control’s absolute gap. Small random-direction effects do not establish specificity.','',
              '| Original content | Assistant think-prefix − plain-prefix effect | 95% CI |', '|---|---:|---|']
    for r in e2s[e2s.metric.eq('projection')&e2s.direction.eq('assistant')&e2s.context.eq('Thinking')].itertuples():
        lines.append(f'| {r.content} | {r.mean:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |')
    lines += ['', 'The raw E1 and E2 effects and factorial contrasts for every control, as well as the companion cosine estimates, '
              'are in `tables/`. The original cosine E0 gate is not replaced by a new gate.','',
              '## Main figures and appendix','']
    for name,item in CAPTIONS.items():
        lines += [f'### {item["title"]}','',f'![{item["title"]}](figures/{name}.png)','',item['caption'],'',f'[Vector PDF](figures/{name}.pdf)','']
    lines += ['## Interpretation limits','']+['- '+x for x in facts['limitations']]
    lines += ['', f'Trajectory bins contain between {facts["trajectory_prompt_coverage_range"][0]} and {facts["trajectory_prompt_coverage_range"][1]} prompts; exact counts are saved. '
              'The deficit is from early answer positions removed by the original first-five rule, not omitted observations hidden by a smoother.',
              '', '## Reproduce','', '```bash',f'python scripts/build_paper_figures.py {facts["source_run"]} --output artifacts/new-paper-figures','```','',
              '`manifest.json` records source hashes, parameters and arithmetic/bootstrap checks. All errors and uncertainty are explicit; '
              'the retained data do not identify H1/H2/H3 or prove a distinct persona.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir');p.add_argument('--output',required=True)
    args=p.parse_args();build(args.run_dir,args.output)
