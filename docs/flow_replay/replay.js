"use strict";

const evidence = window.replayEvidence;
const byId = (id) => document.getElementById(id);
const count = new Intl.NumberFormat("zh-CN");
const number = (value) => count.format(value);
const utc = (epoch) => new Date(epoch).toISOString().slice(11, 23);
const signed = (value) => (value > 0 ? "+" : "") + number(value);

const rows = evidence.rows;
const windowStart = rows[0].time;
const windowEnd = rows[rows.length - 1].time;
const width = 30_000;
const bins = Array.from({ length: Math.floor((windowEnd - windowStart) / width) + 1 }, () => []);
for (const [i, row] of rows.entries()) {
  bins[Math.floor((row.time - windowStart) / width)].push(i);
}
const tallest = Math.max(...bins.map((items) => items.length));
let actual = true;
let selected = rows.findIndex((row) => row.score > evidence.threshold);
const bars = [];
let playback = null;

function makeTimeline() {
  bins.forEach((indices, index) => {
    const button = document.createElement("button");
    button.className = "bucket";
    button.disabled = indices.length === 0;
    const alert = document.createElement("span");
    const other = document.createElement("span");
    alert.className = "bar alert";
    other.className = "bar other";
    button.append(alert, other);
    button.addEventListener("click", () => {
      stopPlayback();
      selected = indices[0];
      renderTimeline();
    });
    byId("plot").append(button);
    bars.push({ button, alert, other, index });
  });
  byId("start").textContent = utc(windowStart).slice(0, 8);
  byId("end").textContent = utc(windowEnd).slice(0, 8);
}

function renderTimeline() {
  const row = rows[selected];
  const currentBin = Math.floor((row.time - windowStart) / width);
  for (const { button, alert, other, index } of bars) {
    const ids = bins[index];
    const alarms = actual ? ids.filter((i) => rows[i].score > evidence.threshold).length : 0;
    const from = utc(windowStart + index * width).slice(0, 8);
    const hint = from + " UTC 起，" + ids.length + " 条正常流，" + alarms + " 条误报";
    button.setAttribute("aria-label", hint);
    button.title = hint;
    button.setAttribute("aria-pressed", String(currentBin === index));
    alert.style.height = (100 * alarms / tallest) + "%";
    other.style.height = (100 * (ids.length - alarms) / tallest) + "%";
  }

  byId("actual").setAttribute("aria-pressed", String(actual));
  byId("zero").setAttribute("aria-pressed", String(!actual));
  byId("alarmCount").textContent = (actual ? evidence.hostFP : evidence.zeroHostFP) + " / " + rows.length + " 条流误报";
  byId("rangeNote").textContent = "纵轴最高柱为 " + tallest + " 条。图中为该主机 " + evidence.hostCount +
    " 条校准记录中，落在首末误报之间的 " + rows.length + " 条。点击时间柱，或用箭头逐条查看。";
  byId("flowTime").textContent = utc(row.time);
  byId("flowId").textContent = "UQ CSV · 第 " + number(row.row) + " 行";
  const isAlarm = actual && row.score > evidence.threshold;
  const verdict = byId("verdict");
  verdict.textContent = isAlarm ? "误报" : "未告警";
  verdict.classList.toggle("alarm", isAlarm);
  const threshold = evidence.threshold.toFixed(9);
  if (actual) {
    const relation = row.score === evidence.threshold ? " = " : (isAlarm ? " > " : " < ");
    byId("score").textContent = "分数 " + row.score.toFixed(9) + relation + "阈值 " + threshold;
  } else {
    byId("score").textContent = "分数 ≤ 阈值 " + threshold;
  }
  byId("flows").textContent = number(actual ? row.flows : 0);
  byId("ips").textContent = number(actual ? row.ips : 0);
  byId("ports").textContent = number(actual ? row.ports : 0);
  byId("noReply").textContent = ((actual ? row.noReply : 0) * 100).toFixed(3) + "%";
  byId("position").textContent = (selected + 1) + " / " + rows.length;
  byId("prev").disabled = selected === 0;
  byId("next").disabled = selected === rows.length - 1;
}

function stopPlayback() {
  window.clearInterval(playback);
  playback = null;
  byId("play").setAttribute("aria-pressed", "false");
  byId("play").textContent = "逐条播放";
}

byId("play").addEventListener("click", () => {
  if (playback !== null) {
    stopPlayback();
    return;
  }
  if (selected === rows.length - 1) selected = 0;
  byId("play").setAttribute("aria-pressed", "true");
  byId("play").textContent = "暂停播放";
  renderTimeline();
  playback = window.setInterval(() => {
    selected += 1;
    renderTimeline();
    if (selected === rows.length - 1) stopPlayback();
  }, 600);
});

const notes = {
  "CIC2018": "旧窗口在本次方案设计前已打开。这里回放的是已抽样的 CIC2018 流。Bot 沿用原生标签。",
  "34-1": "新时段 34-1 用较早的 IoT-23 流训练和校准。纯 C&C 按原作者的指令控制通信标签统计，与 CIC2018 Bot 分开。所有扫描到达时，无标签参考池都不足 1,000 条。",
  "49-1": "新时段 49-1 用较早的 IoT-23 流训练和校准。纯 C&C 按原作者的指令控制通信标签统计。无标签更新生效时，参考池的攻击比例达到 99.923%–99.940%。",
};
const policyNotes = {
  fixed: "固定阈值在较早的正常流上登记，每个模型独立校准。",
  unlabeled: "每 15 分钟只用过去一小时的已见分数更新，不读取标签；不足 1,000 条参考时回退到原固定阈值。",
  recent_normal: "条件对照假设能在 15 分钟后获得可信的正常确认。数据中没有真实的确认到达日志。",
};
const names = { normal: "正常流误报", Scanning: "扫描", Bot: "Bot", "C&C": "纯 C&C", OtherAttack: "其他攻击" };

function renderComparison() {
  const scene = byId("scene").value;
  const mode = byId("comparison").value;
  const policySelect = byId("policy");
  policySelect.options[0].disabled = mode === "update";
  if (mode === "update" && policySelect.value === "fixed") policySelect.value = "unlabeled";
  const policy = policySelect.value;
  const groups = evidence.scenes[scene][mode][policy];
  byId("baseName").textContent = mode === "compare" ? "原字段" : "固定阈值";
  byId("candName").textContent = mode === "compare" ? "加入历史" : "更新阈值";
  const normal = groups.normal;
  const budget = Math.floor(normal.rows / 100);
  const pass = normal.candidate <= budget;
  byId("fpr").textContent = (100 * normal.candidate / normal.rows).toFixed(4) + "%";
  byId("budgetFlag").textContent = pass ? "误报在 1% 预算内" : "超出 1% 预算 " + number(normal.candidate - budget) + " 条";
  byId("budgetFlag").classList.toggle("over", !pass);
  byId("budgetCounts").textContent = "误报 " + number(normal.candidate) + " / 正常流 " + number(normal.rows) + " · 上限 " + number(budget);
  const tbody = byId("pairs");
  tbody.replaceChildren();
  for (const [name, label] of Object.entries(names)) {
    const g = groups[name];
    if (!g) continue;
    const tr = document.createElement("tr");
    const header = document.createElement("th");
    header.scope = "row";
    header.textContent = label;
    const total = document.createElement("small");
    total.textContent = "共 " + number(g.rows) + " 条";
    header.append(total);
    tr.append(header);
    for (const value of [number(g.baseline), number(g.candidate), number(g.added), number(g.lost), signed(g.candidate - g.baseline)]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    tbody.append(tr);
  }
  byId("selectionNote").textContent = notes[scene];
  byId("ruleNote").textContent = policyNotes[policy];
  if (policy === "recent_normal") {
    byId("ruleNote").textContent += scene === "CIC2018" ? "旧窗口使用距更新时刻 75 至 15 分钟的正常流。" : "新场景使用前 24 小时内已确认的正常流。";
  }
}

for (const [id, useActual] of [["actual", true], ["zero", false]]) {
  byId(id).addEventListener("click", () => {
    actual = useActual;
    renderTimeline();
  });
}
for (const [id, step] of [["prev", -1], ["next", 1]]) {
  byId(id).addEventListener("click", () => {
    stopPlayback();
    selected = Math.max(0, Math.min(rows.length - 1, selected + step));
    renderTimeline();
  });
}
for (const id of ["scene", "comparison", "policy"]) byId(id).addEventListener("change", renderComparison);
for (const [name, value] of [["UQ 原始 CSV", evidence.sourceCSV], ["第二十轮结果包", evidence.round20], ["第二十一轮结果包", evidence.round21]]) {
  const term = document.createElement("dt");
  const hash = document.createElement("dd");
  term.textContent = name;
  hash.textContent = value;
  byId("hashes").append(term, hash);
}
makeTimeline();
renderTimeline();
renderComparison();
