# 六轨迹训练与 96 个事件审计

本次固定训练与选轮合同下，96 个记录候选中没有合格改进；这不能推出合同在参数空间中不可满足。48 个 transport 事件的 H 均低于各自 baseline，说明正常子群尾部指标被改变；现有证据没有证明这些变化可在保护原检出能力的同时带来 C/E 改善。

## 执行与拒绝

六轨迹均完成 2,224 次更新和 16 次选轮，总计 13,344 次更新；无 P 重拟合，六份交付工件均逐字节回退原同 seed R30。配对的初始化、预处理、F/V 身份、采样及 dropout 流在全部 16 个事件一致。

| 轨迹 | 全守卫通过 | H 严格下降 | H0 → 记录最小H（步） | 去掉V逐正例守卫后其余条件通过 |
|---|---:|---:|---|---|
| identity_target_s17 | 0/16 | 10/16 | 0.637350506 → 0.609291628（1251） | 无 |
| identity_target_s29 | 1/16 | 1/16 | 0.708647654 → 0.707727691（278） | 无 |
| identity_target_s43 | 0/16 | 1/16 | 0.662666667 → 0.661333333（1251） | 无 |
| transport_target_s17 | 0/16 | 16/16 | 0.637350506 → 0.438178473（2224） | 1807,2085 |
| transport_target_s29 | 0/16 | 16/16 | 0.708647654 → 0.382060718（1946） | 无 |
| transport_target_s43 | 0/16 | 16/16 | 0.662666667 → 0.396780129（2224） | 139 |

transport 在全部 48 个同 seed、同步数的配对事件中，H 也均低于 identity 对照。唯一全守卫通过事件为 identity s29 的 step 417；其 H 未严格下降，因此没有入选。原报告标题“守卫从未通过”与正文矛盾。

仅删除 V 逐正例逆序保护后，剩余条件和 H 严格改善同时成立的事件为 transport s17 的 1807、2085，以及 s43 的 139，共 3/96。s29 仍为 0/16；该轨迹所有事件的正常并列对数均超过 baseline。因此不能把失败归结为一条守卫，也不能据此推出放宽该守卫会让三 seed 同时可用。这里仅做历史事件的逻辑复核，不重新选轮、采用候选或评分 C/E。

| 守卫 | 失败事件数/96 |
|---|---:|
| V_each_positive_inversions_not_increased | 89 |
| V_pooled_pauc_not_lower | 38 |
| each_H_q_not_increased | 36 |
| each_source_H_q_not_increased | 31 |
| normal_tied_pairs_not_increased | 36 |
| F_each_positive_inversions_not_increased | 1 |
| V_panel_strict_separation | 4 |
| V_panel_pauc_not_lower | 4 |
| normal_max_tie_block_not_increased | 0 |

## 每条轨迹的联合障碍与逆序幅度

### identity_target_s17

最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：1。对应事件：139: V_pooled_pauc_not_lower; 556: each_H_q_not_increased; 973: each_H_q_not_increased; 1112: each_H_q_not_increased; 1390: each_H_q_not_increased; 1668: each_H_q_not_increased; 1946: each_H_q_not_increased; 2085: each_H_q_not_increased。

F：16 个事件均未出现逐正例逆序数增加。
V：全程曾变差的不同正例 27 条（B 27条）；每事件变差人数范围 15–27，中位数 18。变差正例-事件的逆序增量中位数 1，p95 7，最大值 11。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。

联合失败组合：1次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower]; 7次[V_each_positive_inversions_not_increased,each_H_q_not_increased,each_source_H_q_not_increased]; 7次[V_each_positive_inversions_not_increased,each_H_q_not_increased]; 1次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,each_H_q_not_increased,each_source_H_q_not_increased]。

### identity_target_s29

最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：1。对应事件：278: each_source_H_q_not_increased。

F：16 个事件均未出现逐正例逆序数增加。
V：全程曾变差的不同正例 16 条（B 16条）；每事件变差人数范围 0–10，中位数 3。变差正例-事件的逆序增量中位数 2，p95 4.05，最大值 10。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。

联合失败组合：7次[V_each_positive_inversions_not_increased,each_source_H_q_not_increased]; 5次[each_source_H_q_not_increased]; 1次[全部通过]; 2次[V_each_positive_inversions_not_increased]; 1次[V_each_positive_inversions_not_increased,each_H_q_not_increased]。

### identity_target_s43

最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：1。对应事件：1251: each_H_q_not_increased。

F：16 个事件均未出现逐正例逆序数增加。
V：全程曾变差的不同正例 21 条（B 21条）；每事件变差人数范围 6–18，中位数 12。变差正例-事件的逆序增量中位数 1，p95 3，最大值 9。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。

联合失败组合：1次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,each_source_H_q_not_increased]; 4次[V_each_positive_inversions_not_increased,each_source_H_q_not_increased]; 2次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,each_H_q_not_increased,each_source_H_q_not_increased]; 2次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower]; 1次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,normal_tied_pairs_not_increased,each_H_q_not_increased,each_source_H_q_not_increased]; 1次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,each_H_q_not_increased]; 1次[V_each_positive_inversions_not_increased,each_H_q_not_increased]; 3次[V_each_positive_inversions_not_increased,each_H_q_not_increased,each_source_H_q_not_increased]; 1次[V_each_positive_inversions_not_increased]。

### transport_target_s17

最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：0。对应事件：1807: 无; 2085: 无。

F：全程曾变差的不同正例 5 条（A 5条）；每事件变差人数范围 0–5，中位数 0。变差正例-事件的逆序增量中位数 1，p95 1，最大值 1。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。
V：全程曾变差的不同正例 39 条（B 39条）；每事件变差人数范围 9–33，中位数 18。变差正例-事件的逆序增量中位数 2，p95 3，最大值 4。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。

联合失败组合：1次[F_each_positive_inversions_not_increased,V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,normal_tied_pairs_not_increased,each_H_q_not_increased]; 8次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,each_H_q_not_increased]; 3次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,normal_tied_pairs_not_increased,each_H_q_not_increased]; 2次[V_each_positive_inversions_not_increased]; 1次[V_each_positive_inversions_not_increased,normal_tied_pairs_not_increased]; 1次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,normal_tied_pairs_not_increased]。

### transport_target_s29

最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：1。对应事件：139: normal_tied_pairs_not_increased; 278: normal_tied_pairs_not_increased; 417: normal_tied_pairs_not_increased; 1529: normal_tied_pairs_not_increased; 1668: normal_tied_pairs_not_increased; 2085: normal_tied_pairs_not_increased。

F：16 个事件均未出现逐正例逆序数增加。
V：全程曾变差的不同正例 39 条（B 39条）；每事件变差人数范围 0–35，中位数 20。变差正例-事件的逆序增量中位数 5，p95 11，最大值 29。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。

联合失败组合：5次[V_each_positive_inversions_not_increased,normal_tied_pairs_not_increased]; 1次[normal_tied_pairs_not_increased]; 7次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,normal_tied_pairs_not_increased]; 1次[V_each_positive_inversions_not_increased,V_panel_strict_separation,V_pooled_pauc_not_lower,V_panel_pauc_not_lower,normal_tied_pairs_not_increased]; 2次[V_each_positive_inversions_not_increased,V_panel_strict_separation,V_panel_pauc_not_lower,normal_tied_pairs_not_increased]。

### transport_target_s43

最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：0。对应事件：139: 无。

F：16 个事件均未出现逐正例逆序数增加。
V：全程曾变差的不同正例 36 条（B 36条）；每事件变差人数范围 3–36，中位数 15。变差正例-事件的逆序增量中位数 2，p95 16，最大值 28。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。

联合失败组合：1次[V_each_positive_inversions_not_increased]; 2次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower]; 6次[V_each_positive_inversions_not_increased,V_pooled_pauc_not_lower,normal_tied_pairs_not_increased]; 6次[V_each_positive_inversions_not_increased,normal_tied_pairs_not_increased]; 1次[V_each_positive_inversions_not_increased,V_panel_strict_separation,V_panel_pauc_not_lower,normal_tied_pairs_not_increased]。

## 高尾分位与并列变化

transport 的 48 个事件均通过 A/B 来源内各 Hq 不增检查。s17 的池化 H 虽全部下降，池化 top1% 的 Hq 却在前 12 个记录事件超过 baseline；这正是只看最大 H 会漏掉的局部退化。s29、s43 的池化三个 Hq 均未超过各自 baseline。以下向量依次为 top10%、top5%、top1%。

| transport轨迹 | 来源 | Hq baseline → 最后事件2224 |
|---|---|---|
| transport_target_s17 | A | (0.665501, 0.586201, 0.118712) → (0.440018, 0.324011, 0.058914) |
| transport_target_s17 | B | (0.295498, 0.258682, 0.120354) → (0.204636, 0.155788, 0.022000) |
| transport_target_s29 | A | (0.710488, 0.567295, 0.116872) → (0.409660, 0.227415, 0.047821) |
| transport_target_s29 | B | (0.420900, 0.386334, 0.097846) → (0.202251, 0.079032, 0.035333) |
| transport_target_s43 | A | (0.545814, 0.490018, 0.106753) → (0.399540, 0.203149, 0.060529) |
| transport_target_s43 | B | (0.599733, 0.263867, 0.089678) → (0.160450, 0.110215, 0.043333) |

transport s29 的正常并列对数相对 baseline 4020 增加 7–17，最大并列块始终为 84。step278 仅失败并列对数保护：4020→4034，其余守卫通过，H 从 0.708647654 降至 0.516375345。这说明存在不同的拒绝路径，不能把所有失败归入逐正例保护，也不能把少量并列增加直接解释成分数塌缩。冻结合同仍然拒绝该事件，本审计不据结果修改规则。

删除 V 逐正例守卫后的三个逻辑通过事件，其原失败规模为：s17 step1807 有12个B正例各增1–3个正常逆序，step2085 有9个B正例各增1–2；s43 step139 有3个B正例各增1。全部六轨迹中，V逆序变差的身份均来自B。

## 损失、梯度与采样

以下损失是首个与最后一个 139 步窗口的路径均值，不是 step0 或终点损失。两臂 normal loss 的目标不同，不能直接把绝对数值大小当作方法优劣。

| 轨迹 | BCE 首→末 | normal 首→末 | positive 首→末 | 全程裁剪步数/最大裁剪前范数 |
|---|---|---|---|---|
| identity_target_s17 | 2.47839e-05 → 5.24612e-06 | 4.44881e-05 → 2.15174e-05 | 6.94752e-07 → 0 | 0 / 0.167796299 |
| identity_target_s29 | 7.80981e-05 → 1.7408e-05 | 3.76198e-05 → 1.15682e-05 | 5.78287e-07 → 4.12613e-07 | 0 / 0.203355893 |
| identity_target_s43 | 1.04392e-05 → 4.60639e-06 | 8.92934e-05 → 7.28417e-06 | 1.44255e-08 → 0 | 0 / 0.460386306 |
| transport_target_s17 | 0.000331641 → 2.7287e-05 | 0.0951516 → 0.035239 | 8.70998e-05 → 2.40308e-11 | 0 / 0.998627245 |
| transport_target_s29 | 0.000329817 → 6.23746e-05 | 0.0929691 → 0.0324159 | 0.000162408 → 3.9601e-07 | 0 / 0.862983108 |
| transport_target_s43 | 8.04215e-05 → 2.67667e-05 | 0.103787 → 0.0358977 | 8.66606e-05 → 4.00105e-07 | 1 / 1.298440695 |

六条轨迹每组均抽取 142,336 行次，最终累计完整循环：A0=16、A1=27、B0=24、B1=1581。B1 的 98 个完整循环对应首个 139 步事件，不能作为全程轮数。transport s43 的首个窗口有 1 步发生裁剪，最大裁剪前范数 1.2984406948；原报告“从未裁剪”不准确。裁剪次数本身不构成优化失稳证据。

## 对随包解释的判断

- “合同不可满足”超出证据：只能确认这六条轨迹的 96 个记录点没有同时满足保护与 H 改善。baseline 已满足保护，identity s29 step417 也满足保护；没有证明满足全部改进条件的参数状态不存在。
- “目标不是无效”应拆开：transport 的 H 有实际下降，不能说它完全没有作用；其完整方法未产生合格候选，不能据 loss/H 下降证明目标能够解决误报且保留检出。
- identity 不是严格保持原输出的无更新对照：它仍有 BCE、正常分数 MSE、正例单侧软锚和 AdamW 更新。其失败不足以证明任何微调都会破坏保护条件。
- “交集出现1次”不准确：1次仅指全守卫通过；全守卫通过且 H 严格改善的交集是0。
- transport s17 首事件 F 上15个正例绝对分数下降是报告项；真正逆序变差是5个A正例各增加1个正常逆序，二者不能混用。
- 拒绝事件提供了新增训练过程证据。最终 C/E 因回退只评价原工件，不能据该评价推出被拒绝候选在 C/E 上的实际效果。

## 可复核范围

审计只读取回传 JSON 和工件字节，对所读回传文件逐项核对 archive_manifest 的 SHA-256。H/Hq 与守卫由已保存的 Fraction、逆序数组和 pAUC 记录重新计算，未加载模型。被拒绝事件没有完整原始分数向量，故不能独立重建其 ROC、排序或部署判决。

training_audit.json 保存全部 96 事件的 H/Hq/来源 Hq、各守卫、联合失败、F/V 逐正例变差身份和增量分布；training_audit.py 可按同路径重放，生成同一结果。
