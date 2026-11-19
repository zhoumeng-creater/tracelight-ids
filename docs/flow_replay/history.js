/* Static evidence view: no requests, model calls, or stored user data. */
"use strict";

(() => {
  const data = window.TRACELIGHT_HISTORY;
  const timeline = document.getElementById("timeline");
  const filters = document.getElementById("category-filters");
  const statusFilter = document.getElementById("status-filter");
  const search = document.getElementById("history-search");
  const order = document.getElementById("history-order");
  const count = document.getElementById("result-count");
  const empty = document.getElementById("empty-state");
  const categoryNames = new Map(data.categories.map(item => [item.id, item.label]));
  const statusNames = new Map(data.statuses.map(item => [item.id, item.label]));
  let selectedCategory = "all";

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function sourceList(sources) {
    const details = element("details", "sources");
    details.append(element("summary", "", "查看来源与原始记录"));
    const list = element("ul", "source-list");
    sources.forEach(source => {
      const item = element("li");
      if (source.href) {
        const link = element("a", "", source.label);
        link.href = source.href;
        item.append(link, element("span", "source-kind", "仓库文档"));
      } else {
        item.append(element("span", "", source.label), element("span", "source-kind", "本地归档路径"));
        item.append(element("code", "local-path", source.path));
      }
      if (source.note) item.append(element("p", "", source.note));
      list.append(item);
    });
    details.append(list);
    return details;
  }

  function stageCard(stage) {
    const item = element("li", "stage");
    item.id = stage.id;
    item.dataset.status = stage.status;
    const article = element("article");
    article.setAttribute("aria-labelledby", `${stage.id}-title`);
    const meta = element("div", "stage-meta");
    const time = element("time", "stage-date", stage.dateLabel);
    time.dateTime = stage.date;
    meta.append(time, element("span", "stage-rounds", stage.rounds));
    meta.append(element("span", `status ${stage.status}`, statusNames.get(stage.status)));
    const title = element("h3", "", stage.title);
    title.id = `${stage.id}-title`;
    article.append(meta, title, element("p", "purpose", stage.purpose), element("span", "scope", stage.scope));
    const evidence = element("dl", "evidence-grid");
    const positive = element("div", "positive");
    positive.append(element("dt", "", stage.positiveLabel || "已取得的结果"), element("dd", "", stage.positive));
    const limitation = element("div", "limit");
    limitation.append(element("dt", "", "失败或限制"), element("dd", "", stage.limitation));
    evidence.append(positive, limitation);
    const decision = element("p", "disposition");
    decision.append(element("span", "disposition-label", "现行处置"), element("span", "", stage.disposition));
    const tags = element("div", "stage-tags");
    stage.categories.forEach(category => tags.append(element("span", "stage-tag", categoryNames.get(category))));
    article.append(evidence, decision, tags, sourceList(stage.sources));
    item.append(article);
    return item;
  }

  function render() {
    const query = search.value.trim().toLocaleLowerCase();
    const shown = data.stages.filter(stage => {
      const categoryMatch = selectedCategory === "all" || stage.categories.includes(selectedCategory);
      const statusMatch = statusFilter.value === "all" || stage.status === statusFilter.value;
      const haystack = [stage.rounds, stage.dateLabel, stage.title, stage.purpose, stage.scope,
        stage.positive, stage.limitation, stage.disposition,
        ...stage.categories.map(category => categoryNames.get(category))].join(" ").toLocaleLowerCase();
      return categoryMatch && statusMatch && (!query || haystack.includes(query));
    });
    if (order.value === "descending") shown.reverse();
    timeline.replaceChildren(...shown.map(stageCard));
    count.textContent = `显示 ${shown.length} / ${data.stages.length} 个阶段`;
    empty.hidden = shown.length !== 0;
    timeline.hidden = shown.length === 0;
    filters.querySelectorAll("button").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.category === selectedCategory));
    });
  }

  [{ id: "all", label: "全部主题" }, ...data.categories].forEach(category => {
    const button = element("button", "category-button");
    button.type = "button";
    button.dataset.category = category.id;
    const total = category.id === "all" ? data.stages.length : data.stages.filter(stage => stage.categories.includes(category.id)).length;
    button.append(element("span", "", category.label), element("span", "category-count", String(total)));
    button.addEventListener("click", () => { selectedCategory = category.id; render(); });
    filters.append(button);
  });
  data.statuses.forEach(status => {
    const option = element("option", "", status.label);
    option.value = status.id;
    statusFilter.append(option);
  });
  search.addEventListener("input", render);
  statusFilter.addEventListener("change", render);
  order.addEventListener("change", render);
  document.getElementById("reset-filters").addEventListener("click", () => {
    selectedCategory = "all";
    statusFilter.value = "all";
    search.value = "";
    order.value = "ascending";
    render();
  });
  render();
})();
