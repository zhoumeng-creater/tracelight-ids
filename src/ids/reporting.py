"""Generate a self-contained inspection artifact from verified validation runs."""
from __future__ import annotations

import json
from pathlib import Path

from .common import read_json, utc_now
from .runner import verify_run


def build_report(run_root: str | Path, output: str | Path) -> Path:
    root, output = Path(run_root), Path(output)
    directories = [root] if (root / "run.json").is_file() else sorted(p.parent for p in root.glob("*/run.json"))
    if not directories:
        raise ValueError("No completed run manifests found")
    runs, datasets = [], set()
    for directory in directories:
        meta = verify_run(directory)
        datasets.add(meta["dataset_manifest_sha256"])
        runs.append({"id": meta["run_id"], "model": meta["model_name"],
                     "data_kind": meta["data_kind"], "manifest_sha256": meta["dataset_manifest_sha256"],
                     "metrics": read_json(directory / "validation_metrics.json"),
                     "threshold": read_json(directory / "validation_threshold.json"),
                     "examples": read_json(directory / "examples.json"),
                     "features": read_json(directory / "selected_features.json"),
                     "config": read_json(directory / "config.json")})
    if len(datasets) != 1:
        raise ValueError("Comparison requires identical prepared dataset manifests")
    payload = json.dumps({"generated_at": utc_now(), "runs": runs}, ensure_ascii=False, allow_nan=False)
    # Data remains inert even when CSV category strings contain HTML/script tags.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    document = _HTML.replace("__RUN_DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>TraceLight · Validation Evidence</title>
<style>
:root{color-scheme:dark;--bg:#0b111a;--panel:#121d29;--ink:#eef4f8;--muted:#a1b3c4;--cyan:#6ee7df;--orange:#ffb870;--line:#2b3b4b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1380px;margin:auto;padding:40px 5vw}h1{font-size:clamp(32px,5vw,60px);line-height:1.08;letter-spacing:-2px;margin:16px 0}h2{font-size:21px;margin:0 0 16px}p{margin:8px 0}.eyebrow{color:var(--cyan);letter-spacing:3px;font-size:12px}.muted,small{color:var(--muted)}.notice{border-left:3px solid var(--orange);padding:12px 18px;background:#241e18;margin:24px 0;color:#ffe1ba}.toolbar{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:24px 0}select,button{border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);padding:10px;font:inherit}select:focus,button:focus{outline:2px solid var(--cyan)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:24px;margin:24px 0 36px}.metric{border-top:1px solid var(--line);padding-top:12px}.metric strong{display:block;font-size:34px;font-weight:500;font-variant-numeric:tabular-nums}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:32px}section{min-width:0}.chart{width:100%;height:auto;display:block;background:var(--panel);border-radius:8px}.legend{display:flex;gap:24px;margin:12px 0;font-size:13px}.normal{color:var(--cyan)}.attack{color:var(--orange)}.scroll{overflow:auto;max-height:480px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;text-align:left;font-size:13px}th{color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--bg)}td,th{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}tr.selected td{background:#173444}tbody tr[data-id]{cursor:pointer}button.link{border:0;padding:0;background:transparent;color:var(--cyan);text-decoration:underline;text-align:left}.below{margin-top:38px}.details{font-family:ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:430px;overflow:auto;background:var(--panel);padding:18px;border-radius:8px;font-size:12px}.key{color:var(--cyan)}footer{margin-top:40px;border-top:1px solid var(--line);padding-top:16px;font-size:12px;overflow-wrap:anywhere}svg text{font:12px system-ui;fill:#a1b3c4}@media(max-width:850px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}main{padding:24px 5vw}.metric strong{font-size:28px}}@media print{body{background:white;color:black}.notice{color:black}select,button{display:none}th{background:white}.details,.chart{background:#eee}.muted,small{color:#555}}
</style></head><body><main>
<div class="eyebrow">TRACELIGHT / RESEARCH WORKBENCH</div>
<h1>每一次判断，都能追溯。</h1>
<p class="muted">轻量化网络入侵检测 · 验证集实验记录与逐样本核查</p>
<div class="notice" id="notice"></div>
<div class="toolbar"><label>模型运行 <select id="run"></select></label><span id="scope" class="muted"></span></div>
<div class="metrics" id="metrics"></div>
<div class="grid"><section><h2>攻击分数与决策阈值</h2><svg class="chart" id="chart" viewBox="0 0 680 280" role="img" aria-label="验证样本攻击分数及阈值散点图"></svg><div class="legend"><span class="normal">● 正常真值</span><span class="attack">● 攻击真值</span><span>虚线：开发阶段阈值</span></div><small id="sampling"></small></section>
<section><h2>相同划分下的基线对比</h2><div class="scroll"><table><thead><tr><th>运行</th><th>攻击召回</th><th>正常误报</th><th>AP</th><th>原始字段</th></tr></thead><tbody id="comparison"></tbody></table></div><p class="muted">阈值使用验证集正常样本确定，因此表中误报率是开发记录；最终结论需要独立校准与测试。</p></section></div>
<div class="grid below"><section><h2>逐样本核查</h2><div class="toolbar"><label>查看 <select id="filter"><option value="all">全部展示样本</option><option value="errors">误报与漏报</option><option value="alert">产生告警</option></select></label><span id="count" class="muted"></span></div><div class="scroll"><table><thead><tr><th>样本 ID</th><th>分数</th><th>真值</th><th>结果</th></tr></thead><tbody id="events"></tbody></table></div></section>
<section><h2>证据明细</h2><p class="muted">点击记录查看原始输入、分数和阈值。分数不是经过概率校准的攻击概率。</p><pre id="detail" class="details">选择一条记录</pre><p id="fields" class="muted"></p></section></div>
<footer id="provenance"></footer>
</main><script id="run-data" type="application/json">__RUN_DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('run-data').textContent), $=id=>document.getElementById(id);
const pct=x=>x==null?'未定义':(100*x).toFixed(2)+'%', num=x=>x==null?'未定义':Number(x).toFixed(4);
let active=data.runs[0];
function cell(row,value){const el=document.createElement('td');el.textContent=value;row.append(el);return el;}
function label(e){return e.predicted_label===e.true_label?'正确':e.predicted_label===1?'误报':'漏报';}
data.runs.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;$('run').append(o);const tr=document.createElement('tr');[r.id,pct(r.metrics.attack_recall),pct(r.metrics.normal_fpr),num(r.metrics.average_precision),r.metrics.n_raw_features].forEach(v=>cell(tr,v));$('comparison').append(tr);});
function detail(e){$('detail').textContent=JSON.stringify({sample_id:e.sample_id,split:e.split,data_kind:e.data_kind,score:e.score,threshold:e.threshold,predicted_label:e.predicted_label,true_label:e.true_label,attack_category:e.attack_category,features:e.features},null,2);}
function rows(){const mode=$('filter').value,events=active.examples.events.filter(e=>mode==='all'||(mode==='errors'?e.predicted_label!==e.true_label:e.predicted_label===1));$('events').replaceChildren();events.forEach(e=>{const tr=document.createElement('tr');tr.dataset.id=e.sample_id;const td=cell(tr,'');const b=document.createElement('button');b.className='link';b.textContent=e.sample_id.slice(-18);b.title=e.sample_id;b.onclick=()=>detail(e);td.append(b);[num(e.score),e.attack_category,label(e)].forEach(v=>cell(tr,v));tr.onclick=()=>detail(e);$('events').append(tr);});$('count').textContent=events.length+' / '+active.examples.events.length+' 条展示记录';if(events.length)detail(events[0]);else $('detail').textContent='该筛选下没有展示样本。';}
function svg(tag,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!=null)el.textContent=text;return el;}
function chart(){const el=$('chart');el.replaceChildren();const es=active.examples.events;for(let i=0;i<=4;i++){const y=232-i*48;el.append(svg('line',{x1:48,x2:650,y1:y,y2:y,stroke:'#2b3b4b'}),svg('text',{x:12,y:y+4},(i/4).toFixed(2)));}es.forEach((e,i)=>{const dot=svg('circle',{cx:50+i*598/Math.max(1,es.length-1),cy:232-e.score*192,r:3,fill:e.true_label?'#ffb870':'#6ee7df',opacity:.8});dot.append(svg('title',{},e.sample_id+' · '+label(e)+' · '+num(e.score)));el.append(dot);});const t=Math.min(1,active.threshold.threshold),y=232-t*192;el.append(svg('line',{x1:48,x2:650,y1:y,y2:y,stroke:'#ffffff','stroke-dasharray':'6 5'}),svg('text',{x:50,y:20},'阈值 '+num(active.threshold.threshold)+(active.threshold.threshold>1?'（全部拒绝告警）':'')),svg('text',{x:48,y:266},'按稳定样本 ID 排列；横轴不是时间'));}
function render(){const r=active,m=r.metrics;const kind=r.data_kind==='synthetic'?'合成工程测试数据：不能用作论文性能证据。':r.data_kind==='mirror'?'公开镜像数据：已记录来源与哈希，尚未与官方文件逐字节比对。':'官方来源数据：来源与文件哈希见数据清单。';$('notice').textContent=kind+' 当前展示 validation 开发记录，未展示最终测试结果。';$('scope').textContent='验证集 '+m.n_samples.toLocaleString()+' 行 · '+m.n_raw_features+' 个原始字段 → '+m.encoded_dim+' 维编码';$('metrics').replaceChildren();[['攻击召回率',pct(m.attack_recall)],['正常流量误报率',pct(m.normal_fpr)],['PR 平均精度 AP',num(m.average_precision)],['Macro-F1',num(m.macro_f1)]].forEach(([k,v])=>{const d=document.createElement('div');d.className='metric';const s=document.createElement('span');s.textContent=k;s.className='muted';const strong=document.createElement('strong');strong.textContent=v;d.append(s,strong);$('metrics').append(d);});$('sampling').textContent='图与明细固定展示稳定样本 ID 排序后的前 '+r.examples.events.length+' 条；指标使用完整验证集。没有伪造 IP、时间戳或实时流量。';$('fields').textContent='本次模型使用：'+r.features.join(', ');$('provenance').textContent='数据清单 SHA-256: '+r.manifest_sha256+' | 生成时间: '+data.generated_at+' | 无外部服务、无数据上传；此报告是可重建的实验工件。';chart();rows();}
$('run').onchange=()=>{active=data.runs[Number($('run').value)];render();};$('filter').onchange=rows;render();
</script></body></html>'''
