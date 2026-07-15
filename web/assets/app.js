(() => {
  "use strict";

  const OFFICIAL_HOSTS = new Set([
    "modelcontextprotocol.io",
    "docs.langchain.com",
    "docs.python.org",
    "json-schema.org",
    "docs.docker.com",
  ]);

  const INTENT_LABELS = {
    design: "设计 / 历史",
    implementation: "当前实现",
    official: "官方规范",
    comparison: "跨源比较",
    out_of_scope: "超出范围",
  };

  const ROLE_LABELS = {
    current_implementation: "当前实现",
    indexed_implementation: "索引实现候选",
    internal_design: "内部设计",
    internal_history: "历史记录",
    external_normative: "官方规范",
    supporting: "补充证据",
  };

  const WARNING_LABELS = {
    model_generation_failed_fallback_used: "DeepSeek 调用失败，已安全降级为确定性证据摘要。",
    model_generation_empty_fallback_used: "DeepSeek 返回了空内容，已安全降级为确定性证据摘要。",
    model_generation_invalid_citations_fallback_used: "DeepSeek 返回的引用不符合证据约束，已安全降级为确定性证据摘要。",
  };

  const state = {
    mode: "retrieve",
    requestController: null,
    lastRequest: null,
  };

  const dom = {
    serviceStatus: document.querySelector("#service-status"),
    serviceStatusText: document.querySelector("#service-status-text"),
    healthButton: document.querySelector("#health-button"),
    metricBuild: document.querySelector("#metric-build"),
    metricFreshness: document.querySelector("#metric-freshness"),
    metricDocuments: document.querySelector("#metric-documents"),
    metricPartitions: document.querySelector("#metric-partitions"),
    metricLive: document.querySelector("#metric-live"),
    metricBranch: document.querySelector("#metric-branch"),
    metricRevision: document.querySelector("#metric-revision"),
    metricAnswerProvider: document.querySelector("#metric-answer-provider"),
    metricAnswerModel: document.querySelector("#metric-answer-model"),
    modeButtons: [...document.querySelectorAll(".mode-button")],
    form: document.querySelector("#query-form"),
    queryInput: document.querySelector("#query-input"),
    queryCount: document.querySelector("#query-count"),
    queryHint: document.querySelector("#query-hint"),
    topKInput: document.querySelector("#top-k-input"),
    topKDown: document.querySelector("#top-k-down"),
    topKUp: document.querySelector("#top-k-up"),
    tokenInput: document.querySelector("#token-input"),
    toggleToken: document.querySelector("#toggle-token"),
    formError: document.querySelector("#form-error"),
    submitButton: document.querySelector("#submit-button"),
    submitLabel: document.querySelector("#submit-button .button-label"),
    cancelButton: document.querySelector("#cancel-button"),
    retryButton: document.querySelector("#retry-button"),
    resultPanel: document.querySelector(".result-panel"),
    requestTime: document.querySelector("#request-time"),
    emptyState: document.querySelector("#empty-state"),
    loadingState: document.querySelector("#loading-state"),
    errorState: document.querySelector("#error-state"),
    errorTitle: document.querySelector("#error-title"),
    errorMessage: document.querySelector("#error-message"),
    responseState: document.querySelector("#response-state"),
    responseSummary: document.querySelector("#response-summary"),
    responseAlerts: document.querySelector("#response-alerts"),
    answerBlock: document.querySelector("#answer-block"),
    answerText: document.querySelector("#answer-text"),
    evidenceLabel: document.querySelector("#evidence-label"),
    evidenceCount: document.querySelector("#evidence-count"),
    evidenceCaption: document.querySelector("#evidence-caption"),
    evidenceList: document.querySelector("#evidence-list"),
    examples: [...document.querySelectorAll("[data-question]")],
  };

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function token() {
    return dom.tokenInput.value.trim();
  }

  function redacted(message) {
    const secret = token();
    const text = String(message || "");
    return secret ? text.split(secret).join("[redacted]") : text;
  }

  async function requestJson(path, options = {}) {
    const controller = options.controller || new AbortController();
    const timeout = window.setTimeout(() => controller.abort("timeout"), 120_000);
    const headers = { Accept: "application/json" };
    const suppliedToken = token();
    if (suppliedToken) headers.Authorization = `Bearer ${suppliedToken}`;
    if (options.body) headers["Content-Type"] = "application/json";

    try {
      const response = await fetch(path, {
        method: options.method || "GET",
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
        credentials: "same-origin",
        redirect: "error",
      });
      const raw = await response.text();
      let payload = null;
      if (raw) {
        try {
          payload = JSON.parse(raw);
        } catch (_error) {
          throw new ApiError(response.status, "服务返回了无效的 JSON 响应");
        }
      }
      if (!response.ok) {
        const detail = payload && payload.detail;
        const message = Array.isArray(detail)
          ? detail.map((item) => item.msg || "输入不合法").join("；")
          : detail || `请求失败（${response.status}）`;
        throw new ApiError(response.status, redacted(message));
      }
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new ApiError(response.status, "服务响应不是 JSON 对象");
      }
      return payload;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function setServiceStatus(kind, text) {
    dom.serviceStatus.className = `status-badge status-${kind}`;
    dom.serviceStatusText.textContent = text;
  }

  function shortHash(value, size = 12) {
    const text = String(value || "");
    return text ? text.slice(0, size) : "—";
  }

  function renderHealth(payload) {
    const index = payload.index || {};
    const revision = payload.live_revision || {};
    const fresh = payload.index_fresh === true;
    setServiceStatus(fresh ? "ok" : "warning", fresh ? "索引新鲜" : "索引可能过期");

    dom.metricBuild.textContent = shortHash(index.build_id, 22);
    dom.metricFreshness.textContent = fresh
      ? "与当前 manifest 一致"
      : "静态知识可能落后";
    dom.metricDocuments.textContent = Number(index.document_count || 0).toLocaleString("zh-CN");
    dom.metricPartitions.textContent = `${Number(payload.partition_count || 0)} 个物理分区`;

    const liveChecks = [
      payload.live_code_enabled,
      payload.live_ast_enabled,
      payload.live_git_enabled,
    ];
    const liveCount = liveChecks.filter(Boolean).length;
    dom.metricLive.textContent = liveCount === 3 ? "全部可用" : `${liveCount}/3 可用`;
    dom.metricBranch.textContent = revision.branch || "—";
    const dirty = revision.dirty === true ? " · 含未提交修改" : "";
    dom.metricRevision.textContent = `${shortHash(revision.commit_sha)}${dirty}`;
    dom.metricAnswerProvider.textContent = payload.model_generation_enabled
      ? "DeepSeek 已配置"
      : "确定性";
    dom.metricAnswerModel.textContent = payload.answer_model || "本地证据摘要";
  }

  function resetHealthMetrics(message) {
    dom.metricBuild.textContent = "—";
    dom.metricFreshness.textContent = message;
    dom.metricDocuments.textContent = "—";
    dom.metricPartitions.textContent = "— 个物理分区";
    dom.metricLive.textContent = "—";
    dom.metricBranch.textContent = "—";
    dom.metricRevision.textContent = "等待工作区信息";
    dom.metricAnswerProvider.textContent = "—";
    dom.metricAnswerModel.textContent = "等待服务配置";
  }

  async function checkHealth() {
    dom.healthButton.disabled = true;
    dom.healthButton.textContent = "连接中…";
    setServiceStatus("pending", "加载服务中");
    try {
      const payload = await requestJson("/health");
      renderHealth(payload);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setServiceStatus("warning", "需要访问令牌");
        resetHealthMetrics("请在查询面板填写令牌");
      } else {
        setServiceStatus("error", "服务不可用");
        resetHealthMetrics(redacted(error.message || "无法连接服务"));
      }
    } finally {
      dom.healthButton.disabled = false;
      dom.healthButton.textContent = "检查服务";
    }
  }

  function setMode(mode) {
    state.mode = mode;
    dom.modeButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    dom.submitLabel.textContent = mode === "answer" ? "生成回答" : "开始检索";
    dom.queryHint.textContent =
      mode === "answer"
        ? "证据充分后按服务配置生成；在线失败时安全降级"
        : "问题会先经过证据路由";
  }

  function clampTopK(value) {
    const parsed = Number.parseInt(String(value), 10);
    const valid = Number.isFinite(parsed) ? parsed : 5;
    return Math.min(20, Math.max(1, valid));
  }

  function validateQuery() {
    const query = dom.queryInput.value.trim();
    const topK = clampTopK(dom.topKInput.value);
    dom.topKInput.value = String(topK);
    if (!query) return { error: "请输入一个工程知识问题。" };
    if (query.length > 1000) return { error: "问题不能超过 1000 个字符。" };
    return { query, topK };
  }

  function showState(name) {
    dom.emptyState.hidden = name !== "empty";
    dom.loadingState.hidden = name !== "loading";
    dom.errorState.hidden = name !== "error";
    dom.responseState.hidden = name !== "response";
    dom.resultPanel.setAttribute("aria-busy", String(name === "loading"));
  }

  function setBusy(busy) {
    dom.submitButton.disabled = busy;
    dom.cancelButton.hidden = !busy;
    dom.modeButtons.forEach((button) => {
      button.disabled = busy;
    });
  }

  function errorCopy(error) {
    if (error instanceof ApiError) {
      if (error.status === 401) {
        return ["访问令牌无效", "请展开“本地访问令牌”，填写与服务端一致的令牌后重试。"];
      }
      if (error.status === 422) {
        return ["查询参数不合法", redacted(error.message)];
      }
      if (error.status === 503) {
        return ["知识服务尚未就绪", "请确认索引和 manifest 已存在；首次加载模型时可稍后重试。"];
      }
      return [`服务返回 ${error.status}`, redacted(error.message)];
    }
    if (error && error.name === "AbortError") {
      return ["请求已取消", "没有产生新的检索结果。"];
    }
    return ["无法连接知识服务", redacted(error && error.message ? error.message : "请确认服务已启动。")];
  }

  function showError(error) {
    const [title, message] = errorCopy(error);
    dom.errorTitle.textContent = title;
    dom.errorMessage.textContent = message;
    showState("error");
  }

  function summaryItem(label, value, stateClass = "") {
    const card = element("div", `summary-item ${stateClass}`.trim());
    card.append(element("span", "", label), element("strong", "", value));
    return card;
  }

  function addAlert(kind, text) {
    dom.responseAlerts.append(element("div", `alert alert-${kind}`, text));
  }

  function safeOfficialUrl(value) {
    try {
      const parsed = new URL(String(value));
      const host = parsed.hostname.toLowerCase();
      const allowed = [...OFFICIAL_HOSTS].some(
        (candidate) => host === candidate || host.endsWith(`.${candidate}`),
      );
      return parsed.protocol === "https:" && allowed ? parsed.href : null;
    } catch (_error) {
      return null;
    }
  }

  function roleLabel(role) {
    return ROLE_LABELS[role] || role || "未分类证据";
  }

  function metadataText(item) {
    const values = [];
    if (item.symbol) values.push(`symbol: ${item.symbol}`);
    if (item.line_start) {
      const lines = item.line_end && item.line_end !== item.line_start
        ? `${item.line_start}-${item.line_end}`
        : String(item.line_start);
      values.push(`lines: ${lines}`);
    }
    if (item.retriever) values.push(`retriever: ${item.retriever}`);
    if (typeof item.score === "number") values.push(`score: ${item.score.toFixed(4)}`);
    if (item.authority) values.push(`authority: ${item.authority}`);
    return values;
  }

  function sourceNode(source) {
    const safeUrl = safeOfficialUrl(source);
    if (!safeUrl) return element("span", "evidence-source", source || "unknown source");
    const link = element("a", "evidence-source-link", source);
    link.href = safeUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  }

  function renderEvidence(item, index) {
    const role = item.evidence_role || "supporting";
    const card = document.createElement("details");
    card.className = "evidence-card";
    card.open = index < 1;

    const summary = document.createElement("summary");
    const title = element("div", "evidence-title");
    const top = element("div", "evidence-title-top");
    top.append(element("span", `role-chip role-${role}`, roleLabel(role)));
    if (item.live_verified === true) top.append(element("span", "live-chip", "LIVE VERIFIED"));
    title.append(top, sourceNode(item.citation || item.source || "unknown source"));

    const meta = element("div", "evidence-meta");
    metadataText(item).forEach((value) => meta.append(element("span", "", value)));
    title.append(meta);
    summary.append(title, element("span", "expand-symbol", "+"));
    card.append(summary);

    const rawContent = String(item.content || "").trim();
    const content = rawContent.length > 12_000
      ? `${rawContent.slice(0, 12_000)}\n\n[页面展示已截断]`
      : rawContent || "该结果没有可展示的文本摘录。";
    card.append(element("pre", "evidence-content", content));
    return card;
  }

  function renderCitation(citation) {
    const card = element("article", "evidence-card citation-only");
    const top = element("div", "evidence-title-top");
    const role = citation.evidence_role || "supporting";
    top.append(element("span", `role-chip role-${role}`, roleLabel(role)));
    if (citation.live_verified === true) top.append(element("span", "live-chip", "LIVE VERIFIED"));
    card.append(top, sourceNode(citation.source || "unknown source"));
    const meta = element("div", "evidence-meta");
    const fields = [];
    if (citation.symbol) fields.push(`symbol: ${citation.symbol}`);
    if (citation.line_start) {
      fields.push(`lines: ${citation.line_start}${citation.line_end ? `-${citation.line_end}` : ""}`);
    }
    if (citation.revision) fields.push(`revision: ${shortHash(citation.revision)}`);
    fields.forEach((value) => meta.append(element("span", "", value)));
    card.append(meta);
    return card;
  }

  function renderResponse(payload, mode, elapsedMs) {
    const isRetrieve = mode === "retrieve";
    const sufficient = isRetrieve ? payload.sufficient_evidence === true : payload.refused !== true;
    const intent = INTENT_LABELS[payload.intent] || payload.intent || "未知";
    const liveAttempted = isRetrieve ? payload.live_verification_attempted === true : null;
    const generationLabel = payload.generation_mode === "model"
      ? (payload.generation_provider === "deepseek" ? "DeepSeek" : (payload.generation_provider || "模型"))
      : payload.generation_mode === "deterministic_fallback"
        ? "证据兜底"
        : payload.generation_mode === "refusal"
          ? "拒答"
          : "确定性摘要";

    dom.responseSummary.replaceChildren(
      summaryItem("路由", intent),
      summaryItem(
        "证据状态",
        sufficient ? "充分" : isRetrieve ? "不足" : "已拒答",
        sufficient ? "summary-good" : "summary-bad",
      ),
      summaryItem(
        isRetrieve ? "实时核验" : "生成方式",
        isRetrieve ? (liveAttempted ? "已执行" : "未触发") : generationLabel,
      ),
      summaryItem("请求耗时", `${elapsedMs.toFixed(0)} ms`),
    );
    dom.requestTime.textContent = `${elapsedMs.toFixed(0)} ms`;

    dom.responseAlerts.replaceChildren();
    if (payload.refusal_reason) addAlert("error", `拒答原因：${payload.refusal_reason}`);
    if (isRetrieve && !sufficient && (payload.results || []).length) {
      addAlert("warning", "以下内容仅是候选线索，不能作为该问题的充分证明。 ");
    }
    (payload.warnings || []).forEach((warning) => {
      addAlert("warning", WARNING_LABELS[warning] || warning);
    });
    if (isRetrieve && payload.live_revision && payload.live_revision.dirty === true) {
      const revision = payload.live_revision;
      addAlert(
        "info",
        `实时结果来自 ${revision.branch || "当前分支"}，commit ${shortHash(revision.commit_sha)}；工作区包含未提交修改。`,
      );
    }

    dom.answerBlock.hidden = isRetrieve;
    const rawAnswer = String(payload.answer || "没有生成回答。");
    dom.answerText.textContent = isRetrieve
      ? ""
      : rawAnswer.length > 6_000
        ? `${rawAnswer.slice(0, 6_000)}\n\n[页面展示已截断，请查看下方引用]`
        : rawAnswer;

    const items = isRetrieve ? payload.results || [] : payload.citations || [];
    dom.evidenceCount.textContent = String(items.length);
    dom.evidenceLabel.textContent = isRetrieve ? "证据列表" : "回答引用";
    dom.evidenceCaption.textContent = isRetrieve ? "按综合排名展示" : "用于约束上方摘要";
    dom.evidenceList.replaceChildren();
    if (!items.length) {
      dom.evidenceList.append(
        element("div", "no-evidence", "没有可展示的证据。请查看拒答原因或修改问题。"),
      );
    } else {
      items.forEach((item, index) => {
        dom.evidenceList.append(isRetrieve ? renderEvidence(item, index) : renderCitation(item));
      });
    }
    showState("response");
  }

  async function submitQuery(request = null) {
    dom.formError.hidden = true;
    const validated = request || validateQuery();
    if (validated.error) {
      dom.formError.textContent = validated.error;
      dom.formError.hidden = false;
      dom.queryInput.focus();
      return;
    }

    const activeRequest = {
      query: validated.query,
      topK: validated.topK,
      mode: validated.mode || state.mode,
    };
    state.lastRequest = activeRequest;
    const controller = new AbortController();
    state.requestController = controller;
    setBusy(true);
    showState("loading");
    dom.requestTime.textContent = "处理中";
    const started = performance.now();

    try {
      const payload = await requestJson(`/${activeRequest.mode}`, {
        method: "POST",
        body: { query: activeRequest.query, top_k: activeRequest.topK },
        controller,
      });
      if (activeRequest.mode === "retrieve" && payload.schema_version !== "engineering-retrieval/v1") {
        throw new ApiError(502, "检索响应版本不受支持");
      }
      renderResponse(payload, activeRequest.mode, performance.now() - started);
      setServiceStatus("ok", "服务在线");
    } catch (error) {
      showError(error);
    } finally {
      if (state.requestController === controller) state.requestController = null;
      setBusy(false);
    }
  }

  dom.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  dom.queryInput.addEventListener("input", () => {
    dom.queryCount.textContent = String(dom.queryInput.value.length);
    dom.formError.hidden = true;
  });

  dom.queryInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      dom.form.requestSubmit();
    }
  });

  dom.examples.forEach((button) => {
    button.addEventListener("click", () => {
      dom.queryInput.value = button.dataset.question || "";
      dom.queryCount.textContent = String(dom.queryInput.value.length);
      dom.queryInput.focus();
    });
  });

  dom.topKDown.addEventListener("click", () => {
    dom.topKInput.value = String(clampTopK(clampTopK(dom.topKInput.value) - 1));
  });
  dom.topKUp.addEventListener("click", () => {
    dom.topKInput.value = String(clampTopK(clampTopK(dom.topKInput.value) + 1));
  });
  dom.topKInput.addEventListener("change", () => {
    dom.topKInput.value = String(clampTopK(dom.topKInput.value));
  });

  dom.toggleToken.addEventListener("click", () => {
    const show = dom.tokenInput.type === "password";
    dom.tokenInput.type = show ? "text" : "password";
    dom.toggleToken.textContent = show ? "隐藏" : "显示";
  });

  dom.tokenInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      checkHealth();
    }
  });

  dom.form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuery();
  });

  dom.cancelButton.addEventListener("click", () => {
    if (state.requestController) state.requestController.abort("user");
  });

  dom.retryButton.addEventListener("click", () => {
    if (state.lastRequest) {
      setMode(state.lastRequest.mode);
      submitQuery(state.lastRequest);
    }
  });

  dom.healthButton.addEventListener("click", checkHealth);

  setMode("retrieve");
  showState("empty");
  checkHealth();
})();
