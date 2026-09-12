"""Export observed low/middle/high E1 examples; never alter the frozen run."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('runs/a40-two-hour-reserves'); out=Path('artifacts/model-output-examples')
rows=pd.read_csv(root/'tokens.csv.gz',low_memory=False)
gaps=pd.read_csv(root/'e1_prompt_contrasts.csv')
gaps=gaps[gaps.direction.eq('cos_assistant') & gaps.contrast.eq('think_minus_answer')]
records=[json.loads(p.read_text()) for p in (root/'transcripts').glob('*.json')]
by_key={(r['prompt_id'],r['condition']):r for r in records if r['experiment']=='E1'}
keys=['cos_assistant','cos_ctrl_1','cos_ctrl_2','cos_null','cos_random']
labels=['Assistant','Skeptic control','Judge control','Structured null','Random']
colors=['#2962a3','#b58a28','#c76536','#73823f','#a35782']
examples=[]
for domain in ['advice','math']:
    ordered=gaps[gaps.domain.eq(domain)].sort_values('value')
    for label,position in [('Lowest gap',0),('Middle gap',len(ordered)//2),('Highest gap',len(ordered)-1)]:
        row=ordered.iloc[position]; natural=by_key[(row.prompt_id,'natural_think')]; plain=by_key[(row.prompt_id,'natural_nothink')]
        selected=rows[rows.record_id.eq(natural['record_id']) & rows.primary_include].sort_values('token_idx')
        points=[]
        for seg in ['think','answer']:
            part=selected[selected.segment.eq(seg)].copy()
            for _,b in part.groupby(part.segment_idx//20):
                points.append({'x':float(b.token_idx.mean()),'v':[round(float(b[k].mean()),7) for k in keys],
                               'segment':seg,'n':len(b),'start':int(b.token_idx.min()),'end':int(b.token_idx.max())})
        item={'id':row.prompt_id,'domain':domain,'selection':label,'gap':float(row.value),
              'prompt':natural['messages'][-1]['content'],'system':natural['messages'][0]['content'],
              'think':natural['think_text'],'answer':natural['answer_text'],'nothink':plain['answer_text'],
              'boundary':natural['boundary_idx'],'think_tokens':natural['think_token_count'],
              'answer_tokens':natural['answer_token_count'],'nothink_tokens':plain['answer_token_count'],
              'record_id':natural['record_id'],'nothink_record_id':plain['record_id'],'points':points}
        examples.append(item)
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for i,(name,color) in enumerate(zip(labels,colors)):
            for j,seg in enumerate(['think','answer']):
                pts=[p for p in points if p['segment']==seg]
                ax.plot([p['x'] for p in pts],[p['v'][i] for p in pts],color=color,
                        lw=1.8 if i==0 else 1.1,ls='-' if i==0 else '--',label=name if j==0 else None)
        ax.axvline(item['boundary'],color='#666666',lw=1,label='</think> boundary')
        ax.axhline(0,color='#777777',lw=.7,alpha=.5)
        ax.set(title=f"{domain.capitalize()} · {label.lower()} · gap {item['gap']:+.4f}",
               xlabel='Absolute token position (prompt included)',ylabel='Centered activation cosine')
        ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=3,loc='best')
        fig.text(.5,-.025,'Up to 20 tokens per bin; segments separate; first 5 answer tokens excluded; no uncertainty intervals.',ha='center',fontsize=8)
        for ext in ['png','pdf']:fig.savefig(out/(item['id']+'.'+ext),bbox_inches='tight',dpi=160)
        plt.close(fig)
        text=f"# {domain.capitalize()}: {label.lower()}\n\nPrompt ID: `{item['id']}`. Assistant thinking-minus-answer gap: {item['gap']:+.6f}.\n\n"
        text+='Selection: lowest, upper-middle or highest observed Assistant gap within this domain; illustrative, not random.\n\n'
        text+=f"![Activation traces]({item['id']}.png)\n\n"
        for heading,content in [('System prompt',item['system']),('User prompt',item['prompt']),('Full model thinking',item['think']),('Full final answer — thinking enabled',item['answer']),('Full answer — thinking disabled',item['nothink'])]:
            text+=f'## {heading}\n\n'+''.join('> '+line+'\n' for line in content.splitlines())+'\n'
        text+=f"Original thinking record: `{item['record_id']}`; no-thinking record: `{item['nothink_record_id']}`.\n"
        (out/(item['id']+'.md')).write_text(text)
(out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
(out/'README.md').write_text('# Saved model outputs and individual activation traces\n\nQwen3-32B, frozen E1 run `96325204a0eeebe0`, seed 0, layer 32. Full original text is preserved. These are model-generated research outputs, not human annotations. Examples span the lowest, middle and highest observed Assistant gap per domain; selection is illustrative and outcome-based.\n\n'+''.join(f"- [{e['domain'].capitalize()} — {e['selection'].lower()} ({e['gap']:+.4f})]({e['id']}.md)\n" for e in examples)+'\nPlots use non-overlapping bins of up to 20 tokens within each segment, excluding the first five answer tokens. There are no uncertainty intervals on these single-generation traces. Unsummarized scalar measurements remain in `runs/a40-two-hour-reserves/tokens.csv.gz`.\n')
print('Exported',len(examples),'paired examples with full outputs and PNG/PDF traces')
