"""Export descriptive real-input figures, not classifier results."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with gzip.open(args.geometry/'connection_geometry.jsonl.gz','rt',encoding='utf8') as f:
        rows=[json.loads(l) for l in f]
    sources=['ait_santos_intranet_server','ait_fox_intranet_server','unraveled_target']
    labels=['AIT santos','AIT fox','Unraveled']
    colors=['#177E89','#E58F22','#5252A3']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),gridspec_kw={'width_ratios':[1,1.1,1]})
    for i,(s,label,color) in enumerate(zip(sources,labels,colors)):
        attack=[r for r in rows if r['source']==s and r['target_kind']=='control' and r['raw8']['resp_port']==443]
        normal=[r for r in rows if r['source']==s and r['label']==0 and r['raw8']['resp_port']==443]
        if s=='unraveled_target':
            lo=min(r['connection_start'] for r in attack);hi=max(r['observed_end'] for r in attack)
            normal=[r for r in normal if lo<=r['connection_start']<=hi]
        axes[0].boxplot([[r['ack_segment_count'] for r in attack],[r['ack_segment_count'] for r in normal]],
                        positions=[i*3+1,i*3+2],widths=.6,showfliers=False,
                        patch_artist=True,boxprops={'facecolor':color,'alpha':.35},medianprops={'color':'black'})
        vals=sorted(r['first20_unique_byte_fraction']*100 for r in attack)
        axes[1].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=color,label=f'{label} (n={len(vals)})')
    axes[0].set_xticks([1,2,4,5,7,8],['C2','Normal']*3)
    axes[0].tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('ACK segments (quartiles; outliers omitted)')
    axes[0].set_title('A  Few stages in both classes',loc='left',fontweight='bold')
    for i,label in enumerate(labels):axes[0].text(i*3+1.5,-.23,label,ha='center',transform=axes[0].get_xaxis_transform())
    axes[1].set_xlabel('Unique TCP bytes observed in first 20 records (%)')
    axes[1].set_ylabel('Fraction of control connections')
    axes[1].set_xlim(0,101);axes[1].set_ylim(0,1.04)
    axes[1].legend(fontsize=8,loc='lower right')
    axes[1].set_title('B  Packet count is not byte coverage',loc='left',fontweight='bold')
    ax=axes[2]
    ax.set_title('C  Same normalized two-stage topology',loc='left',fontweight='bold')
    for y,color,name in [(2.1,'#B4454D','2 attack downloads'),(.7,'#177E89','329 normal HTTP response flows')]:
        ax.plot([.1,.45,.45,.9],[y,y,y-.45,y-.45],color=color,lw=3)
        ax.annotate('',xy=(.9,y-.45),xytext=(.48,y-.45),arrowprops={'arrowstyle':'->','color':color,'lw':2})
        ax.text(.1,y+.15,name,fontweight='bold',color=color,fontsize=10)
        ax.text(.1,y-.2,'request',fontsize=9)
        ax.text(.58,y-.65,'response',fontsize=9)
    ax.text(.1,-.35,'Normal response bodies are small (up to 3,752 B).\nThis is a topology counterexample, not a matched\nlarge-download comparison.',fontsize=9,va='top')
    ax.set_xlim(0,1);ax.set_ylim(-1,2.6);ax.axis('off')
    fig.suptitle('Existing real packets: structure evidence, no fitted models',fontweight='bold',fontsize=14)
    fig.subplots_adjust(bottom=.24,top=.83,wspace=.38)
    args.output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(args.output/('packet_evidence.'+ext),dpi=180,bbox_inches='tight')


if __name__=='__main__':main()
