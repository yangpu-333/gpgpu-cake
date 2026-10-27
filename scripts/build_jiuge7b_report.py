"""Build a self-contained Markdown homepage and offline HTML research report.

Public evidence is snapshotted byte-for-byte. Large raw traces, tensors and
machine access details are not part of the publication bundle.
"""
from pathlib import Path
import hashlib,html,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/jiuge_training'
DOC=ROOT/'docs/jiuge7b';EVID=DOC/'evidence'
SOURCES=[
 ('cpt-formal','results/pinning-driver-20261016-05/formal-independent-review.json'),
 ('sft-formal','results/sft-pinning-independent-review-20261020-continuation-v2.json'),
 ('bi-correctness','results/datathinking-short-loop-correctness-independent-review-20261026-05.json'),
 ('bi-trace','results/datathinking-ssh-20261025-01/NUMA-deliverable-trace-bottleneck-summary-20261027-59.json'),
 ('bi-numeric-failure','results/datathinking-NUMA-followup-failed-native-independent-review-20261027-01.json'),
 ('operator-validation','results/resource-minimal-attention-independent-review-20261027.json'),
 ('peer-review','results/peer-reference-review-20261027.json')]
ANCHORS={'cpt-formal':'appendix-cpt','sft-formal':'appendix-sft','bi-correctness':'appendix-bi','bi-trace':'appendix-trace','bi-numeric-failure':'appendix-failures','operator-validation':'appendix-operator','peer-review':'appendix-peer'}
EVID.mkdir(parents=True,exist_ok=True)
(EVID/'.gitattributes').write_text('*.json -text\n',encoding='utf8')
data={};manifest=[]
for name,path in SOURCES:
 target=EVID/(name+'.json');source=BASE/path
 if source.exists():raw=source.read_bytes();target.write_bytes(raw)
 else:raw=target.read_bytes()
 obj=json.loads(raw);data[name]=obj
 # Reject access details and credentials before anything is published.
 text=raw.decode('utf8')
 assert not re.search(r'(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)|BEGIN [A-Z ]*PRIVATE KEY|id_ed25519|known_hosts|(?:api[_-]?key|password)\s*[=:]\s*["\x27][^*]',text,re.I),name
 manifest.append({'id':name,'source':path,'published_path':'docs/jiuge7b/evidence/'+name+'.json','SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
assert data['cpt-formal']['formal_performance_passed'] and data['sft-formal']['formal_performance_passed']
assert data['bi-correctness']['full_model_correctness_and_native_resume_passed'] and not data['operator-validation']['full_model_training_accepted']
assert len(data['bi-trace']['trace_windows'])==12 and data['operator-validation']['closed_cases']==8
brief=(BASE/'BRIEF.md').read_text(encoding='utf8')
# Keep the original compact BRIEF text, replacing machine-local links with anchors.
brief=brief[:brief.rfind('\n[算子基准复核]')].rstrip()
intro='<a id="report-top"></a>\n\n'+brief
intro+='\n\n[算子基准复核](#appendix-operator) · [详细计划与验收标准](#appendix-plan) · [CPT正式结果](#appendix-cpt) · [SFT正式结果](#appendix-sft)\n\n[下载离线单文件HTML](docs/jiuge7b/report.html) · [其他项目历史概览](docs/project-overview-20261027.md)\n'
parts=[intro,'\n---\n\n## 完整内容与复核证据\n\n以下内容均在本文件内。表格用于阅读，折叠区保留完整关键复核JSON；也可从GitHub下载对应JSON。\n']
def section(anchor,title,text):parts.append(f'\n<a id="{anchor}"></a>\n\n## {title}\n\n{text.rstrip()}\n')
def evidence(name):
 return '\n[独立JSON文件](docs/jiuge7b/evidence/'+name+'.json)\n\n<details>\n<summary>展开完整复核数据</summary>\n\n```json\n'+json.dumps(data[name],ensure_ascii=False,indent=2)+'\n```\n\n</details>\n'
plan='''| 顺序 | 要做什么 | 交付与准入 |
|---|---|---|
| P0 数值与内存 | 固定初始权重、实际输入和RNG，沿embedding梯度→首步更新→次步前向定位跨独立进程差异；为新增自有审计pack做有界写回/退缓存 | 可复现的分歧证据；内容SHA、inode和驻留检查。保留旧参考、原容差与内存保护 |
| P1 同口径分析 | 复用runtime关联、exclusive归因和区间并集方法分析现有trace；必要时用ixSYS采短CUDA/NVTX窗口 | 算子/shape热点、搬运及同步依赖表，明确分母与未归因项 |
| P2 单一候选 | 端到端优先研究offload回传/批次/同步；GPU方向验证7B真实shape的mask/scale/softmax前后向融合 | 保留mask、cast、缩放舍入、重计算和梯度语义；先通过边界正确性与隔离微基准 |
| P3 模型验收 | 三臂完整数值/SFT、原生恢复通过后，补必要的新trace与无profiler正式对照 | 分别报告step延迟、有效输入tokens/s及答案targets/s |

真实7B配置：官方9G7B_MHA，7,668,831,744参数、28层、hidden=3584、FFN=18944、28个MHA头；TP2、BF16、S2048、micro/global batch=1、CPU AdamW offload。数据为48,818条中文Alpaca；CPT监督全文，SFT只监督答案及结束标记。

attention目标为每rank `[1,14,S,S]`，覆盖S2048/S4096。GEMM先对FC1 `[S,3584]×[3584,18944]`、FC2 `[S,9472]×[9472,3584]`及对应反向建基准，再决定是否优化。

验收标准：

1. 三臂各3次真实S2048 BF16更新，每rank199组、3585字段；原FullAudit逐块有限性/CPU float64比较，`atol=3e-4, rtol=1e-3`。覆盖source、answer mask、逐token loss、RNG、scheduler和TP；新增raw独立核验。
2. 原生iteration2保存，native与optimized恢复并真实执行step3；995项恢复、1195项下一步字段，以及HDO调用、398个moment恢复和原12项负例均须通过。
3. 三种子×三臂轮换，每例120步、剔除20步，GPU同步计时取两rank慢端，13条loss精确一致。native/control种子CV≥10%须完整九例确认；两baseline所有配对更快且各几何平均收益>1%，才接受收益。
4. GPU实验先核对资源与他人占用，保留NUMA/RSS/host保护。费用优先抵扣券；正式实验保留初始3小时，诊断与微基准不替代正式结果。'''
section('appendix-plan','详细研究计划与验收标准',plan)
cpt=data['cpt-formal'];table='| seed | arm | 秒/步 | 有效输入tokens/s | 保留步数 |\n|---|---|---:|---:|---:|\n'
for x in cpt['cases']:table+=f"| {x['seed']} | {x['arm']} | {x['mean_seconds_per_step']:.6f} | {x['effective_tokens_per_second']:.3f} | {x['retained_steps']} |\n"
section('appendix-cpt','CPT正式结果：双A100',table+'\n配对吞吐提升为84.4%。合并吞吐与逐种子几何平均是不同统计量；修复组种子均值CV为13.6%。该结果不外推到BI-V150。\n'+evidence('cpt-formal'))
sft=data['sft-formal'];table='| arm | 秒/步 | 输入tokens/s | 答案targets/s |\n|---|---:|---:|---:|\n'
for name,x in sft['rounds'][0]['arms'].items():table+=f"| {name} | {x['mean_seconds_per_step']:.6f} | {x['aggregate_input_token_slots_per_second']:.3f} | {x['aggregate_supervised_answer_targets_per_second']:.3f} |\n"
section('appendix-sft','SFT正式结果：双A100',table+'\n配对吞吐提升为92.9%，三种子均更快，修复组CV为9.24%。完整数值和原生恢复最大误差为0。复核JSON的round字段只记录当轮统计，最终验收以顶层`formal_performance_passed=true`为准。\n'+evidence('sft-formal'))
section('appendix-bi','BI-V150原执行配置的正确性与恢复','旧执行配置三臂各3次真实更新，每rank3585字段最大差0；原生保存、两条真实恢复路径与下一步比较通过。该结论只对其原执行路径有效，不能直接套用到新增NUMA布局和严格执行路径。\n'+evidence('bi-correctness'))
t=data['bi-trace'];table='| 臂 | 窗口耗时 | CPU AdamW | optimizer整体 | AdamW之外 | GPU计算并集 |\n|---|---:|---:|---:|---:|---:|\n'
for name,x in zip(('pinned基线','绑核对照','本地moment候选'),t['summaries']):
 vals=[x[k]for k in ('window_step_seconds_mean','CPUAdamW_seconds_per_rank_step','optimizer_seconds_per_rank_step','optimizer_outside_AdamW_seconds_per_rank_step','GPU_compute_seconds_per_rank_step')]
 table+='| '+name+' | '+' | '.join(f'{y:.3f}'for y in vals)+' |\n'
section('appendix-trace','BI-V150十二窗口瓶颈分析',table+'\n单位为秒/rank/步。各臂35步，采第21–25和31–35步、两rank，共12个rank/window。它们不是独立种子；范围可重叠，不可相加为step耗时，不用于正式加速判断。12窗口CPU AdamW/HtoD记录重叠均为0。候选未记录GPU活动空档代理30.6%–33.8%，不等于硬件利用率；rank间collective时长差不能全归因为链路慢。\n'+evidence('bi-trace'))
o=data['operator-validation'];table='| GPU | S | 输入 | 无profiler中位毫秒 | kernel数 | 抽样前向最大差 | 抽样反向最大差 |\n|---|---:|---|---:|---:|---:|---:|\n'
for x in o['summary']:table+=f"| {x['device']} | {x['S']} | {x['dtype'].replace('torch.','')} | {x['median_ms']:.3f} | {x['kernels']} | {x['forward_error']:.3g} | {x['backward_error']:.3g} |\n"
section('appendix-operator','新增八例算子基准与GPU trace',table+'\n子链为BF16/FP32输入→FP32缩放→有限值-10000因果mask→FP32 softmax及反向；合成输入，shape为`[1,14,S,S]`。每例2次预热、7次无profiler同步计时，另采1次GPU trace。全输出/梯度有限，每例12行CPU float64抽样比较通过原容差。BF16输入路径记录13次kernel启动，FP32为9次，支持进一步研究cast和中间矩阵搬运；不能把两条精度路径的差异称为优化收益。未包含QK/PV矩阵乘、整模型更新或TP通信。\n'+evidence('operator-validation'))
failure='''| 问题 | 已确定的事实 | 目前结论 |
|---|---|---|
| NUMA工作集 | 两rank约28.57GiB延后分配的FP32梯度未完整纳入早期预算；新增审计pack也形成大量文件缓存 | 公共预分配与容量探针有证据；新增完整路径仍须通过原保护 |
| 连续更新差异 | 实际输入/RNG相同，已捕获embedding梯度微差；严格固定权重重复通过，真实连续更新对旧参考仍失败 | 不能指认具体kernel或声称长训练修好，保留原数值标准 |
| 审计布局与入口 | 出现过参考符号链接越界、整块缓冲峰值及模块路径错误 | 失败保留，CPU修复版本化；不把编排修复当训练优化 |
| 检查器断档 | Windows关机导致本机观察器中断；远端持久监督与故障终态探针随后通过 | 与NUMA拒绝、数值失败分别记录 |
| 资源准入 | 当前新增GPU配额为0，现有GPU本地节点余量不足整模型80GiB准入；ACT暂不可达 | 先在现有卡做独立算子研究；不动不明归属环境 |

最近严格native尝试在原NUMA16GiB保护下停止。首步受检字段在原容差内，第二步部分梯度和逐token loss超原容差；三臂完整数值、原生恢复、长loss与正式性能均未接受。预算结束和安全停止是执行边界，不替代数值失败的结论。
'''
section('appendix-failures','失败原因与未完成项',failure+evidence('bi-numeric-failure'))
peer='''| 参考内容 | 可用价值 | 边界 |
|---|---|---|
| 同组7B/8B trace | 原始日志、实际shape与算子分析，支持交叉分析 | S4096短采样，与我方S2048正式训练配置不同 |
| 4B分析器 | runtime关联、exclusive归因、并集/重叠计算、未归因保留 | 热点百分比不能直接外推到7B |
| ixSYS资料 | CPU API、GPU kernel、memcpy及同步依赖分析 | 提供PDF较旧；现有环境实查V4.4.0，以实际help与短探针为准 |

同组7B BF16短前向相对RMS约12.98%，未达其3%门槛；4B最终BF16对齐也未通过。8B仅达到相对RMS<3%、top-1≥90%的短前向标准，约1.88%/90.62%，不等于完整训练与恢复正确性。其checkpoint缺少optimizer/RNG，不能替代精确断点恢复。

我方独立核验172项文件哈希、重算四份SFT rank trace，主要热点排序一致；不同选步边界导致累计kernel时长差不超过约0.011%。分析器9项测试通过。7B S4096的GEMM/BMM约占设备kernel累计时间37.20%，copy/cast约24.59%；该分母不是端到端墙钟。

既有采集和实现归原作者；本报告新增贡献为独立核验与适用范围判断。截图的hidden=4096、FFN=16384与8B相符，不作为7B优化目标。
'''
section('appendix-peer','外部参考与研究归因',peer+evidence('peer-review'))
prov='模型权重、完整数值张量、检查点及全量原trace继续保留在实验原存储；本文件内嵌的是关键复核结果与SHA，不宣称这些大型原件已上传GitHub。\n\n| 证据 | 原复核文件SHA256 | 字节数 |\n|---|---|---:|\n'
for x in manifest:prov+=f"| {x['id']} | `{x['SHA256']}` | {x['bytes']} |\n"
section('appendix-provenance','来源与复核范围',prov)
md='\n'.join(parts).rstrip()+'\n';(ROOT/'README.md').write_text(md,encoding='utf8')
# GitHub links for the short project BRIEF now resolve without ignored local results.
footer='\n\n[算子基准复核](../../README.md#appendix-operator) · [详细计划与验收标准](../../README.md#appendix-plan) · [CPT正式结果](../../README.md#appendix-cpt) · [SFT正式结果](../../README.md#appendix-sft)\n'
(BASE/'BRIEF.md').write_text(brief+footer,encoding='utf8')
# Build a zero-dependency HTML rendition; all research links stay inside one file.
linkmap={'docs/jiuge7b/evidence/'+n+'.json':'#'+ANCHORS[n]for n,_ in SOURCES}
linkmap['docs/jiuge7b/report.html']='#report-top';linkmap['docs/project-overview-20261027.md']='https://github.com/yangpu-333/gpgpu-cake/blob/main/docs/project-overview-20261027.md'
def inline(s):
 tokens={}
 def keep(value):
  key=f'ZZTOKEN{len(tokens)}ZZ';tokens[key]=value;return key
 s=re.sub(r'`([^`]+)`',lambda m:keep('<code>'+html.escape(m[1])+'</code>'),s)
 s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:keep('<a href="'+html.escape(linkmap.get(m[2],m[2]),quote=True)+'">'+html.escape(m[1])+'</a>'),s)
 s=html.escape(s);s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
 for k,v in tokens.items():s=s.replace(k,v)
 return s
out=[];lines=md.splitlines();i=0
while i<len(lines):
 line=lines[i]
 if not line.strip():i+=1;continue
 if line.startswith('```'):
  buf=[];i+=1
  while i<len(lines) and not lines[i].startswith('```'):buf.append(lines[i]);i+=1
  out.append('<pre><code>'+html.escape('\n'.join(buf))+'</code></pre>');i+=1;continue
 if re.fullmatch(r'<a id="[a-z0-9-]+"></a>',line)or line in ('<details>','</details>')or line.startswith('<summary>'):
  out.append(line);i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines)and lines[i].startswith('|'):rows.append([x.strip()for x in lines[i].strip().strip('|').split('|')]);i+=1
  out.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>'for x in rows[0])+'</tr></thead><tbody>')
  for row in rows[2:]:out.append('<tr>'+''.join('<td>'+inline(x)+'</td>'for x in row)+'</tr>')
  out.append('</tbody></table></div>');continue
 h=re.match(r'^(#{1,6}) (.*)',line)
 if h:out.append(f'<h{len(h[1])}>'+inline(h[2])+f'</h{len(h[1])}>')
 elif line=='---':out.append('<hr>')
 else:out.append('<p>'+inline(line)+'</p>')
 i+=1
css='''html{scroll-behavior:smooth}body{margin:0;background:#f6f7f9;color:#18232f;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:24px auto;padding:36px 44px;background:#fff;border:1px solid #e5e9ed;border-radius:12px}h1{font-size:30px;line-height:1.35}h2{font-size:22px;margin-top:36px;border-bottom:1px solid #e5e9ed;padding-bottom:8px}a{color:#145ac7}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 12px;border:1px solid #dbe2ea;text-align:left}th{background:#eef3f8}.table-wrap{overflow-x:auto;margin:18px 0}code{background:#f1f4f7;padding:2px 4px;font-size:13px}pre{padding:18px;background:#101923;color:#d9e5f2;overflow:auto;max-height:640px}pre code{background:transparent;padding:0;color:inherit}summary{cursor:pointer;font-weight:600;color:#145ac7}details{margin:14px 0;padding:12px 16px;border:1px solid #dbe2ea;border-radius:7px}hr{border:0;border-top:1px solid #dbe2ea;margin:32px 0}@media(max-width:700px){main{margin:0;padding:20px 16px;border:0;border-radius:0}h1{font-size:25px}}'''
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>九格7B研究报告</title><style>'+css+'</style></head><body><main>'+''.join(out)+'</main></body></html>'
(DOC/'report.html').write_text(page,encoding='utf8')
manifest_record={'schema':'jiuge7b-public-report-evidence-v1','date':'2026-10-27','files':manifest,'original_evidence_byte_identical':True,'raw_large_files_published':False}
(DOC/'manifest.json').write_text(json.dumps(manifest_record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
# Verify every report link and anchor and enforce standalone HTML assets.
anchors=set(re.findall(r'<a id="([a-z0-9-]+)"></a>',md))
for label,link in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',md):
 if link.startswith('#'):assert link[1:]in anchors,(label,link)
 elif not link.startswith('http'):assert (ROOT/link).is_file(),link
assert not re.search(r'<(?:script|link|img)\b',page,re.I)
for href in re.findall(r'href="([^"]+)"',page):assert href.startswith(('#','https://')),href
for x in manifest:assert hashlib.sha256((ROOT/x['published_path']).read_bytes()).hexdigest()==x['SHA256']
publish=['README.md','experiments/jiuge_training/BRIEF.md','docs/project-overview-20261027.md','docs/jiuge7b/report.html','docs/jiuge7b/manifest.json','docs/jiuge7b/evidence/.gitattributes','scripts/build_jiuge7b_report.py']+[x['published_path']for x in manifest]
qa={'passed':True,'self_contained_Markdown':'README.md','self_contained_HTML':'docs/jiuge7b/report.html','research_anchor_count':len(anchors),'embedded_evidence_records':len(manifest),'all_report_links_resolve':True,'evidence_copies_SHA_equal':True,'HTML_external_assets':0,'publication_paths':publish,'README_bytes':len(md.encode()),'HTML_bytes':len(page.encode())}
(BASE/'results').mkdir(exist_ok=True)
(BASE/'results/public-report-build-review-20261027.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
if '--publication-json'in sys.argv:
 print(json.dumps({'files':[{'path':p,'mode':'100644','type':'blob','content':(ROOT/p).read_text(encoding='utf8')}for p in publish]},ensure_ascii=False))
else:print(json.dumps(qa,ensure_ascii=False))
