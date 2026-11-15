"use strict";

(() => {
  const data = window.readinessEvidence;
  const element = (id) => document.getElementById("r25-" + id);
  const integer = (n) => new Intl.NumberFormat("zh-CN").format(n);
  const percent = (fp, n) => n ? (100 * fp / n).toFixed(4) + "%" : "无法核算";
  const names = { Scanning: "扫描", ExternalC2: "控制通信", Download: "下载通信", OtherAttack: "其他攻击" };
  const reasons = { original_hour_ready: "原一小时边界已具备足够参考", waited_for_samples: "等待严格先前记录达到有限序位要求", hour_not_elapsed: "记录结束时尚未满一小时", insufficient_strictly_prior_rows: "严格先前参考记录仍不足" };
  const roles = { forward_unopened: "原实验的前向确认日期", forward_mixed: "原实验的前向日期，含已开与未开事件", forward_same_testbed: "同一平台的后续时段", development: "开发日期", earlier_transfer: "较早迁移日期", retrospective: "历史回放日期" };
  let selected = "IoT-SFC-106-2025-04-12";
  let unopenedOnly = false;
  const choice = (name) => element(name).value;
  const budget = (source, policy, version = "repaired", phase = "all") => source.budgets[[choice("group"), policy, version, phase].join("|")];
  const eligible = (b) => b[0] > 0 ? b[1] <= b[2] : null;

  function render() {
    const available = data.sources.filter((source) => source.provider === choice("provider"));
    if (!available.some((source) => source.name === selected)) {
      selected = available.find((source) => source.name === "DEDALE-green-D17")?.name || available[0].name;
    }
    const source = available.find((item) => item.name === selected);
    const policy = choice("policy");
    element("days").replaceChildren();
    for (const day of available) {
      const b = budget(day, policy);
      const pass = eligible(b);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "day " + (pass === null ? "unknown" : pass ? "pass" : "fail");
      button.textContent = day.window.slice(5) + (day.provider === "DEDALE" ? " · " + day.name.split("-").at(-1) : "");
      button.setAttribute("aria-pressed", String(day.name === selected));
      button.setAttribute("aria-label", day.window + "，" + (pass === null ? "无正常标签" : pass ? "误报达标" : "误报超标") + "，" + percent(b[1], b[0]));
      button.title = day.name + " · " + percent(b[1], b[0]);
      button.addEventListener("click", () => { selected = day.name; render(); });
      element("days").append(button);
    }
    const current = budget(source, policy);
    const pass = eligible(current);
    element("date").textContent = source.name + " · " + source.window + " UTC";
    element("fpr").textContent = percent(current[1], current[0]);
    element("budget-flag").textContent = pass === null ? "没有正常标签，不能判定达标" : pass ? "本日误报在 1% 预算内" : "本日误报超预算 " + integer(current[1] - current[2]) + " 条";
    element("budget-flag").classList.toggle("over", pass === false);
    element("budget-flag").classList.toggle("unverified", pass === null);
    element("budget-counts").textContent = "误报 " + integer(current[1]) + " / 正常 " + integer(current[0]) + " · 上限 " + integer(current[2]);
    const comparison = choice("comparison");
    const baseline = budget(source, comparison === "same_policy" ? policy : comparison, comparison === "same_policy" ? "original" : "repaired");
    element("base-budget").textContent = "基线误报：" + percent(baseline[1], baseline[0]) + "（" + integer(baseline[1]) + " 条）";
    element("ready").textContent = "本地阈值参考：" + (source.ready === "ready" ? "已就绪" : "未就绪");
    element("ready-detail").textContent = reasons[source.reason] + (source.readyAt ? " · " + source.readyAt.replace("+00:00", " UTC") : "") + (source.referenceRows !== null ? " · 参考 " + integer(source.referenceRows) + " 条" : "");
    const phaseText = (phase, label) => {
      const b = budget(source, policy, "repaired", phase);
      return b ? label + "：误报 " + integer(b[1]) + " / 正常 " + integer(b[0]) + "（" + percent(b[1], b[0]) + "）" : label + "：没有记录";
    };
    element("phases").textContent = phaseText("startup", "修复后就绪边界之前") + "；" + phaseText("after", "边界之后") + "。两阶段均计入本日总误报。";
    element("all-attacks").setAttribute("aria-pressed", String(!unopenedOnly));
    element("unopened-attacks").setAttribute("aria-pressed", String(unopenedOnly));
    const families = (unopenedOnly ? source.unopenedPairs : source.pairs)[[choice("group"), policy, comparison].join("|")] || {};
    element("pairs").replaceChildren();
    let attackRows = 0;
    const keys = [...new Set([...Object.keys(names), ...Object.keys(families)])].filter((key) => !["Benign", "Unresolved"].includes(key));
    for (const family of keys) {
      const v = families[family];
      if (!v) continue;
      attackRows += v[0];
      const tr = document.createElement("tr");
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = names[family] || family;
      tr.append(label);
      for (const value of [v[0], v[1], v[2], v[3], v[4], v[0] - v[2], unopenedOnly ? "未分拆" : source.eventCounts[family] ?? "未登记"]) {
        const td = document.createElement("td");
        td.textContent = typeof value === "number" ? integer(value) : value;
        tr.append(td);
      }
      element("pairs").append(tr);
    }
    if (!attackRows) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 8;
      cell.textContent = unopenedOnly ? "本日没有该范围内的原未开攻击记录。" : "本日没有可按已确认攻击类别核算的记录。";
      row.append(cell);
      element("pairs").append(row);
    }
    element("evidence-note").textContent = (roles[source.role] || source.role) + "。" + (pass === false ? "当前策略超出预算，上表检出不能称为 1% 约束下的收益。" : pass === null ? "没有正常分母，上表不能用于证明低误报检测能力。" : "达标只表示本日正常误报合规；检出与剩余漏检仍须分别查看。") + (policy.endsWith("_old") ? "所选旧策略使用原固定阈值，本地就绪状态不改变其阈值。" : "样本就绪仅说明能够登记有限阈值，不证明参考背景具有代表性。") + " 未列出的类别表示本日没有对应已确认记录，不表示该类全部检出。";
    if (unopenedOnly) element("evidence-note").textContent += " 本表按原实验的 exposure 记录筛选未开攻击；事件 ID 汇总未按该子集拆分。";
    element("unknown").textContent = "未解决标签 " + integer(current[3]) + " 条，其中告警 " + integer(current[4]) + " 条；不计入正常误报或攻击召回分母。";
  }

  const r = data.resource;
  element("resources").textContent = r.workers + " 个工作进程；并行阶段平均 " + r.meanCores.toFixed(3) + " 个 CPU 核，峰值 " + r.peakCores.toFixed(3) + "；进程树 RSS 采样峰值 " + r.peakRssGiB.toFixed(3) + " GiB。完整主调用 " + r.seconds.toFixed(3) + " 秒，不含打包。本轮复用分数，训练 " + r.fits + " 次、模型前向 " + r.forwards + " 次；设备级 GPU 占用不能归因于本轮。";
  const term = document.createElement("dt");
  const value = document.createElement("dd");
  term.textContent = "冻结代码 " + data.release;
  value.textContent = data.commit;
  element("hashes").append(term, value);
  for (const [path, hash] of Object.entries(data.inputHashes).filter(([path]) => !path.includes("/evaluation/"))) {
    const label = document.createElement("dt");
    const code = document.createElement("dd");
    label.textContent = path;
    code.textContent = hash;
    element("hashes").append(label, code);
  }
  for (const name of ["provider", "group", "policy", "comparison"]) element(name).addEventListener("change", render);
  for (const [id, value] of [["all-attacks", false], ["unopened-attacks", true]]) element(id).addEventListener("click", () => { unopenedOnly = value; render(); });
  render();
})();
