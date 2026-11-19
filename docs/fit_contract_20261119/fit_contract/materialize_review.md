# materialize.py 数据整合核阅

2026-11-19。只读核阅代码与已有数据；未运行整合入口、未修改实现、未训练。代码 SHA256：`8f8ab101717f1d4f0a035dbb5dae24cd9f52920e2e92f55bc200a7b8a33df915`。

## 需要修正：主统一表重复保留同一观察的旧、新身份

[materialize.py 第 226 行](<E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/.worktrees/fit-contract-20261119/scripts/fit_contract/materialize.py:226>) 将旧 observations 与全量新 observations 直接相加，随后旧 links 与新 links 也相加。C25 已证明旧、新 raw8 是同一观察，但旧/新 flow_id 不同，因此字符串唯一性检查会通过，主表仍出现重复物理观察。`measurement_aliases` 只是另外一张关系表，主表没有 canonical/active 标记；全部新 cloud 观察还与 v1 中其他 cloud 观察重合，不仅是 C25。读取统一主表或事件候选表的下游容易重复计流、重复命中。

`model_rows` 当前只从新 observations 生成，故这一问题没有让当前 F/V/C 拟合表重复；风险出在主统一消费表和事件评价。

最小修正：v1 原表保留原路径或单独历史副本；当前 observations/event_flow_links 输出一个可直接消费的 canonical 版本，别把版本历史并进样本表。若确需同文件保留版本，必须明确行类型与 canonical ID，并让默认读取器只返回 active 版本；对所有重导出观察建立身份关系，不能只给 C25 建 alias。不要通过删除原件解决。

## 已核实的部分

- C25 对新 cloud 提取逐条按协议、方向端点、端口、起点唯一匹配，25/25 raw8 完全相同。新来源 SHA 与旧 cloud 原件相同；新派生捕获表示另有 SHA。
- 当前 cloud 全轮转有 21,925 条观察，未按标签筛背景；仅 IPv4 本机地址可形成 20,511 个主机视角，1,414 条非本机观察单列保留。root 已改从作者 facts 读完整 IPv4/IPv6 地址，避免由 C 标签反推主机。源文件：`WORK/统一事件与方法落实_20261119/ait/source/capture_config/gather/cloud_share/facts.json`；地址为 `172.19.130.106`，ens3 还含 link-local IPv6。
- cloud 没有空 replay_export_time，没有重复 stable_flow_id；1,660 个负值均为上游 duration 缺失 sentinel。转为 None 而非零的方向正确；其余真实缺失继续由 HGB 原生支持。
- 主机视角由任一端点属于作者本机地址决定；peer 取另一端。全背景先构造历史，再选 supervised 行，动作标签没有筛历史。
- AIT 使用记录的离线导出 network_time；前缀模块禁止同刻互看并核查严格过去。它不是实际检测器接收时间。
- 动作、候选 links、流级 label 分别存储；unknown 不转正常。主分配 santos F、fox V、russell C，没有把 V 用来拟合。稳定身份跨 F/V/C 交叉检查存在，C 正例为零。
- v1 文件只读且有输入哈希，未被覆盖。事件数 254 的输入字段正在等 mapper 完整交付；目前只核阅已存在的 santos 映射，不声称全入口已运行。

本轮只发现上述主表版本重复这一项需要改变产物结构的问题；russell 地址依据已由 root 修正。
