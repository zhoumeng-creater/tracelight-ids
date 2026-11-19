"""Plot fixed descriptive representations; no model, threshold or significance test."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties, fontManager
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent/'representation/query_representations.jsonl.gz')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'order_real_evidence')
    args = parser.parse_args()
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False, 'font.size':10, 'pdf.fonttype':42})

    data = []
    with gzip.open(args.input, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if (r['source_family']=='Unraveled' and r['host']=='10.1.3.8'
                    and r['protocol']==6 and r['response_port']==443 and r['label'] in (0,1)):
                data.append((float(r['time']), int(r['label']), float(np.linalg.norm(r['direction']))))
    data.sort()
    t, y, magnitude = np.asarray(data).T
    attack = y==1
    start, end = t[attack].min(), t[attack].max()
    during = (t>=start)&(t<=end)
    normals_during = magnitude[during & ~attack]
    controls_during = magnitude[during & attack]
    assert len(controls_during)==473 and len(normals_during)==769
    times = [datetime.fromtimestamp(v,timezone.utc) for v in t]
    colors = {0:'#3976A8',1:'#C45D24'}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13.2,5.3),gridspec_kw={'width_ratios':[1.8,1]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.81,wspace=.28)
    ax.axvspan(datetime.fromtimestamp(start,timezone.utc),datetime.fromtimestamp(end,timezone.utc),
               color='#E9C99F',alpha=.22,zorder=0)
    for label,name,size,alpha in [(0,'可信正常（5081 条）',7,.35),(1,'C2（473 条）',10,.72)]:
        mask=y==label
        ax.scatter(np.asarray(times)[mask],magnitude[mask],s=size,c=colors[label],alpha=alpha,
                   linewidths=0,label=name,rasterized=True)
    ax.set_title('完整捕获中的 TCP/443 连接',loc='left',fontsize=11)
    ax.set_ylabel(r'方向向量范数  $\|Rg\|_2$')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M',tz=timezone.utc))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=7))
    ax.set_xlabel('完整流导出时刻（UTC）')
    ax.legend(loc='upper left',fontsize=9,frameon=False)
    ax.text(.99,.97,'浅色区域：首条至末条 C2 导出时段',ha='right',va='top',transform=ax.transAxes,fontsize=8.5)

    series=[normals_during,controls_during]
    boxes=bx.boxplot(series,positions=[1,2],widths=.42,showfliers=False,patch_artist=True,
                     medianprops={'color':'black','linewidth':1.2},
                     boxprops={'linewidth':1},whiskerprops={'linewidth':1},capprops={'linewidth':1})
    for i,(values,box) in enumerate(zip(series,boxes['boxes'])):
        box.set_facecolor(colors[i]);box.set_alpha(.2)
        # Deterministic display offsets only, no resampling or observations omitted.
        offsets=.14*np.sin(np.arange(len(values))*2.399963229728653)
        bx.scatter((i+1)+offsets,values,s=7,color=colors[i],alpha=.25,linewidths=0,rasterized=True)
    bx.set_xticks([1,2],['可信正常\nn=769','C2\nn=473'])
    bx.set_title('同一 C2 时段内的分布',loc='left',fontsize=11)
    bx.set_ylabel(r'方向向量范数  $\|Rg\|_2$')
    for axis in (ax,bx):
        axis.grid(axis='y',alpha=.18)
        axis.spines[['top','right']].set_visible(False)

    span='{}—{}'.format(datetime.fromtimestamp(start,timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                       datetime.fromtimestamp(end,timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    fig.suptitle('Unraveled：真实历史的方向表示与正常业务',x=.075,ha='left',y=.96,fontsize=15)
    fig.text(.075,.885,'表示量，不是攻击评分  |  固定 13 维输入，λ=0  |  同监控主机 10.1.3.8、同 TCP/443',fontsize=10)
    fig.text(.075,.105,f'C2 导出时段（UTC）：{span}。右图保留时段内全部 473 条 C2 与 769 条可信正常连接。',fontsize=9)
    fig.text(.075,.055,'图中没有告警阈值、分类器或显著性检验；一维分布重叠不证明完整表示或所有分类器无效。',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    print(json.dumps({'png':str(args.output.with_suffix('.png')),'pdf':str(args.output.with_suffix('.pdf')),
                      'whole_capture_classes':dict(Counter(int(v) for v in y)),
                      'normal_during':len(normals_during),'control_during':len(controls_during),
                      'real_model_calls':0,'thresholds':0},ensure_ascii=True))


if __name__=='__main__':main()
