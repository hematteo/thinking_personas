"""Research plots from frozen token measurements; no model inference.

E2 pairs exact source token IDs by source record and source-content offset.
E1 uses the original per-prompt segment inclusion policy. Full point tables are
saved separately from figures; no across-prompt averaging occurs in these plots.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from persona_dynamics.io import file_hash

DIRECTIONS=['assistant','ctrl_1','ctrl_2','null','random']
LABELS=['Assistant','Skeptic control','Judge control','Structured null','Random control']
COLORS=['#245A91','#B18325','#C96636','#768339','#A65586']


def paired_e2(tokens):
    e2=tokens.loc[tokens.experiment.eq('E2')]
    keys=['run_id','seed','model','prompt_id','source_record_id','segment_idx']
    pieces=[]
    for content in ['cot','answer']:
        left=e2.loc[e2.condition.eq(content+'_as_cot')]
        right=e2.loc[e2.condition.eq(content+'_as_answer')]
        pairs=left.merge(right,on=keys,how='outer',suffixes=('_think','_plain'),validate='one_to_one',indicator=True)
        if not pairs['_merge'].eq('both').all():raise ValueError('Unmatched E2 source content positions')
        if not pairs.token_id_think.eq(pairs.token_id_plain).all():raise ValueError('E2 source token IDs differ')
        piece=pairs[keys].copy();piece['content']=content;piece['token_id']=pairs.token_id_think
        for name in DIRECTIONS:piece['delta_'+name]=pairs['dot_'+name+'_think']-pairs['dot_'+name+'_plain']
        pieces.append(piece)
    return pd.concat(pieces,ignore_index=True)


def e1_gaps(tokens):
    e1=tokens.loc[tokens.experiment.eq('E1')&tokens.condition.eq('natural_think')&tokens.primary_include&tokens.segment.isin(['think','answer'])]
    keys=['run_id','seed','model','domain','prompt_id']
    columns=['dot_'+n for n in DIRECTIONS]+['cos_assistant']
    means=e1.groupby(keys+['segment'])[columns].mean()
    left=means.xs('think',level='segment');right=means.xs('answer',level='segment')
    if not left.index.equals(right.index):raise ValueError('Incomplete E1 paired segments')
    return (left-right).reset_index()


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=160,facecolor='white')
    fig.savefig(out/(name+'.pdf'),facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir');parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.run_dir).resolve();out=Path(args.output).resolve()
    if out==root or root in out.parents:raise ValueError('Use an output directory outside the frozen run')
    out.mkdir(parents=True,exist_ok=True)
    tokens=pd.read_csv(root/'tokens.csv.gz',low_memory=False,float_precision='round_trip')
    if tokens.seed.nunique()!=1 or tokens.model.nunique()!=1:raise ValueError('These figures require a single seed/model; facet new runs explicitly')
    if tokens.duplicated(['record_id','token_idx']).any():raise ValueError('Duplicate token records')
    columns=['dot_'+n for n in DIRECTIONS]+['cos_assistant']
    if not np.isfinite(tokens[columns].to_numpy()).all():raise ValueError('Non-finite readouts')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#28333F','text.color':'#28333F','axes.titleweight':'bold','grid.color':'#E3E7EB',
        'axes.edgecolor':'#A5ACB4','pdf.fonttype':42})
    pair=paired_e2(tokens);pair.to_csv(out/'e2_exact_token_differences.csv.gz',index=False)
    gaps=e1_gaps(tokens);gaps.to_csv(out/'e1_prompt_gaps.csv',index=False)
    cohort=json.loads((root/'cohort.json').read_text());order={c['domain']:c['selected_prompt_ids'] for c in cohort}
    assert len(gaps)==80 and all(gaps.groupby('domain').size()==40)
    assert all(pair.groupby('content').prompt_id.nunique()==40)
    highlight=order['advice'][0]
    # Every curve is a complete, unsmoothed token sequence. Highlight the first
    # retained prompt in frozen order, not an outcome-selected extreme.
    for zoom,name in [(None,'01_exact_content_context_shift'),(160,'01b_exact_content_first160')]:
        fig,axes=plt.subplots(5,2,figsize=(12,12),sharex=True,sharey=True)
        values=pair[['delta_'+n for n in DIRECTIONS]].to_numpy()
        limit=float(np.max(np.abs(values)))*1.04
        for i,(direction,label,color) in enumerate(zip(DIRECTIONS,LABELS,COLORS)):
            for j,content in enumerate(['cot','answer']):
                ax=axes[i,j];sub=pair.loc[pair.content.eq(content)]
                if zoom:sub=sub.loc[sub.segment_idx<zoom]
                for pid,trace in sub.groupby('prompt_id',sort=False):
                    trace=trace.sort_values('segment_idx')
                    if pid!=highlight:ax.plot(trace.segment_idx+1,trace['delta_'+direction],color=color,lw=.45,alpha=.09,rasterized=True)
                trace=sub.loc[sub.prompt_id.eq(highlight)].sort_values('segment_idx')
                ax.plot(trace.segment_idx+1,trace['delta_'+direction],color=color,lw=.8,alpha=.95,rasterized=True)
                ax.axhline(0,color='#4A515A',lw=.7,ls='--');ax.grid(axis='y',alpha=.5)
                ax.set_ylim(-limit,limit)
                ax.text(.985,.93,label,transform=ax.transAxes,ha='right',va='top',fontsize=10,color=color,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
                if i==0:ax.set_title('Original thinking content' if content=='cot' else 'Original answer content',pad=10)
                if i==4:ax.set_xlabel('Position within identical source content (tokens)')
                if zoom:ax.set_xlim(1,zoom)
        fig.suptitle('Same tokens, different context',x=.07,ha='left',y=.987,fontsize=20,weight='bold')
        fig.text(.07,.957,'Thinking-prefix projection − plain-answer-prefix projection · 40 advice prompts · no averaging or smoothing',fontsize=10)
        fig.supylabel('Raw projection difference',x=.012)
        fig.text(.07,.022,'One trace per prompt. Dark trace: first retained advice prompt. Shared scale across all directions; zero means no context shift.',fontsize=9)
        fig.text(.07,.007,'Highlight: '+highlight+(' · First 160 source tokens shown; full sequences in companion figure.' if zoom else ' · Each trace ends at its own content length.'),fontsize=9)
        fig.subplots_adjust(left=.08,right=.98,top=.92,bottom=.08,hspace=.19,wspace=.10)
        save(fig,out,name)
    # Paired per-prompt gaps, identical inclusion for dot and cosine.
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.6),sharex=True,sharey=True)
    xlim=max(abs(gaps.dot_assistant))*1.15;ylim=max(abs(gaps.cos_assistant))*1.20
    agreement={}
    for ax,domain,color,marker in zip(axes,['advice','math'],['#245A91','#B18325'],['o','D']):
        sub=gaps.loc[gaps.domain.eq(domain)]
        ax.scatter(sub.dot_assistant,sub.cos_assistant,s=43,c=color,marker=marker,alpha=.8,edgecolors='white',linewidth=.6)
        ax.axhline(0,c='#777F89',lw=.8);ax.axvline(0,c='#777F89',lw=.8);ax.grid(alpha=.5);ax.set_axisbelow(True)
        n=int((np.sign(sub.dot_assistant)!=np.sign(sub.cos_assistant)).sum());agreement[domain]=dict(prompts=len(sub),opposite_sign=n)
        ax.set(title=domain.capitalize(),xlabel='Raw projection gap: thinking − answer',xlim=(-xlim,xlim),ylim=(-ylim,ylim))
        ax.text(.03,.96,f'{n}/{len(sub)} prompts change sign',transform=ax.transAxes,va='top',fontsize=10)
    axes[0].set_ylabel('Centered cosine gap: thinking − answer')
    fig.suptitle('Does normalization change the direction of the effect?',x=.075,ha='left',fontsize=17,weight='bold')
    fig.text(.075,.895,'Each point is one prompt · original segment means and first-five answer exclusion · one generation seed',fontsize=9.5)
    fig.text(.075,.028,'Opposite-sign quadrants indicate disagreement. Axes have different units; there is intentionally no identity line.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.80,wspace=.10)
    save(fig,out,'02_raw_vs_cosine')
    # Shared diverging scale; preserve frozen cohort order, do not cluster on outcomes.
    fig,axes=plt.subplots(1,2,figsize=(10.5,10.5),sharey=True)
    limit=float(abs(gaps[['dot_'+n for n in DIRECTIONS]]).max().max())
    mapped=[]
    for ax,domain in zip(axes,['advice','math']):
        sub=gaps.loc[gaps.domain.eq(domain)].set_index('prompt_id').loc[order[domain]]
        matrix=sub[['dot_'+n for n in DIRECTIONS]].to_numpy()
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_xticks(range(5),['Assistant','Skeptic','Judge','Null','Random'],rotation=35,ha='right')
        ax.set_yticks(np.arange(0,40,4),np.arange(1,41,4));ax.set_title(domain.capitalize(),pad=12)
        for rank,pid in enumerate(order[domain],1):mapped.append(dict(domain=domain,display_row=rank,prompt_id=pid))
    axes[0].set_ylabel('Prompt in frozen selection order')
    fig.suptitle('Which directions move, and on which prompts?',x=.09,ha='left',fontsize=18,weight='bold',y=.98)
    fig.text(.09,.948,'Each cell: one prompt’s mean thinking − answer raw projection · 40 prompts per domain',fontsize=10)
    fig.subplots_adjust(left=.09,right=.97,top=.90,bottom=.22,wspace=.12)
    cax=fig.add_axes([.27,.115,.50,.018]);bar=fig.colorbar(im,cax=cax,orientation='horizontal')
    bar.set_label('Raw projection gap · blue: thinking lower · red: thinking higher',fontsize=9)
    fig.text(.09,.045,'Same unit-vector scale for every column; no column-wise normalization, sorting, or clustering.',fontsize=9)
    fig.text(.09,.027,'Segment summaries use the original first-five answer exclusion. Full prompt-ID mapping and values are saved alongside the figures.',fontsize=9)
    save(fig,out,'03_direction_heatmap')
    pd.DataFrame(mapped).to_csv(out/'heatmap_prompt_order.csv',index=False)
    result={'source':str(root),'tokens_sha256':file_hash(root/'tokens.csv.gz'),'script_sha256':file_hash(__file__),
      'e1_prompts':len(gaps),'e2_paired_content_tokens':len(pair),'e2_prompts':40,'highlight_prompt':highlight,
      'sign_disagreement':agreement,'e1_policy':'Original primary inclusion; per-prompt mean thinking minus mean answer, seed 0.',
      'e2_policy':'All exact source token pairs; no filtering, smoothing or averaging; traces aligned by source-content index.',
      'scope':'Exploratory post hoc raw-projection figures; pointwise observations, no new hypothesis confidence intervals.',
      'figures':['01_exact_content_context_shift','01b_exact_content_first160','02_raw_vs_cosine','03_direction_heatmap']}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
