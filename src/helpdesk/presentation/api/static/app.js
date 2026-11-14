(() => {
  "use strict";

  const MAX_QUESTION_LENGTH = 20000;
  const POLL_INTERVAL_MS = 1000;
  const POLL_MAX_MS = 360000;
  const PAGE_SIZE = 30;
  const TITLE_LENGTH = 30;
  const OPEN_STATES = new Set(["accepted", "retrieving", "generating"]);

  const state = {
    conversations: [],
    nextCursor: null,
    conversationId: null,
    turns: [],
    active: null,
    pollTimer: null,
    pollDeadline: 0,
    composing: false,
  };

  const elements = {
    list: document.getElementById("conversation-list"),
    listEmpty: document.getElementById("conversation-empty"),
    loadMore: document.getElementById("load-more"),
    title: document.getElementById("conversation-title"),
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("empty-state"),
    status: document.getElementById("status"),
    question: document.getElementById("question"),
    counter: document.getElementById("counter"),
    send: document.getElementById("send"),
    confirm: document.getElementById("confirm-request"),
    newConversation: document.getElementById("new-conversation"),
  };

  const progressLabels = {
    accepted: "檢索中",
    retrieving: "檢索中",
    generating: "產生回答中",
  };

  const outcomes = {
    answered: { label: "已根據文件回答" },
    unable_to_answer: {
      label: "暫時無法回答這個問題",
      text: "目前無法根據已取得的文件可靠回答此問題。",
      hint: "可以換個說法、補充系統名稱或情境後再問一次。",
    },
    insufficient_evidence: {
      label: "文件中找不到相關依據",
      text: "目前找不到足夠的文件依據，無法可靠回答。",
      hint: "請補充更多細節（例如部門、系統或流程名稱），或換個關鍵字再問一次。",
    },
    external_generation_blocked: {
      label: "依外送政策未產生答案",
      text: "已找到相關文件，但依資料外送政策，這些內容不能送至外部 AI 產生答案。",
      hint: "請參考下列文件，或聯絡文件負責單位。",
    },
    outbound_policy_blocked: {
      label: "依 AI 外送政策未產生答案",
      text: "此問題或相關內容依資料外送政策無法送至目前的 AI 服務，因此未產生答案。",
      hint: "如果問題裡有金鑰、身分證字號或卡號等敏感資料，請移除後再問一次。",
    },
  };

  const failures = {
    provider_rate_limited: "請稍候再試",
    provider_unavailable: "回答模型暫時無法使用，請稍後再試。",
    provider_timeout: "回答模型處理逾時，無法確認結果，請重新提問。",
    provider_contract_error: "回答模型回傳的格式無法驗證，請重新提問。",
    prompt_too_large: "問題或相關文件太長，超過回答模型的處理上限。",
    question_too_long: "問題太長，請縮短後再問。",
    retrieval_failed: "文件檢索暫時無法使用，請稍後再試。",
    processing_interrupted: "處理中斷，無法確認結果，請重新提問。",
  };

  class ApiError extends Error {
    constructor(status, payload, uncertain = false) {
      super("api error");
      this.status = status;
      this.payload = payload;
      this.uncertain = uncertain;
    }
  }

  function newUuid() {
    if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  function setStatus(message, isError = false) {
    elements.status.textContent = message;
    elements.status.classList.toggle("error", isError);
  }

  function busy() {
    return state.active !== null && !state.active.uncertain;
  }

  function refreshControls() {
    const isBusy = busy();
    elements.send.disabled = state.active !== null;
    elements.question.disabled = state.active !== null;
    elements.newConversation.disabled = isBusy;
    elements.loadMore.disabled = isBusy;
    for (const button of elements.list.querySelectorAll("button")) button.disabled = isBusy;
    elements.confirm.classList.toggle("hidden", !state.active || !state.active.uncertain);
  }

  function updateCounter() {
    elements.counter.textContent = `${[...elements.question.value].length.toLocaleString("zh-TW")} / 20,000`;
  }

  function waitText(seconds) {
    if (!Number.isInteger(seconds)) return "";
    if (seconds >= 90) return `（約 ${Math.ceil(seconds / 60)} 分鐘後）`;
    return `（約 ${Math.max(seconds, 1)} 秒後）`;
  }

  function failureText(error) {
    if (!error) return "問題處理失敗，請稍後再試。";
    const text = failures[error.code] || "問題處理失敗，請稍後再試。";
    if (error.code === "provider_rate_limited") return `${text}${waitText(error.retry_after_seconds)}`;
    return text;
  }

  function citationText(citation) {
    const section = citation.section_path ? `，${citation.section_path}` : "";
    const page = Number.isInteger(citation.page) ? `，第 ${citation.page} 頁` : "";
    return `${citation.title}${section}${page}`;
  }

  function citationList(citations) {
    if (!Array.isArray(citations) || citations.length === 0) return null;
    const list = document.createElement("ol");
    list.className = "citation-list";
    for (const citation of citations) {
      const item = document.createElement("li");
      item.value = citation.number;
      item.textContent = citationText(citation);
      list.append(item);
    }
    return list;
  }

  function paragraph(text, className) {
    const element = document.createElement("p");
    if (className) element.className = className;
    element.textContent = text;
    return element;
  }

  function bubble(role, label, extraClass) {
    const article = document.createElement("article");
    article.className = `message ${role}${extraClass ? ` ${extraClass}` : ""}`;
    if (label) {
      const heading = document.createElement("div");
      heading.className = "message-label";
      heading.textContent = label;
      article.append(heading);
    }
    return article;
  }

  function assistantBubble(turn) {
    if (OPEN_STATES.has(turn.state)) {
      const article = bubble("assistant", "Help Desk", "pending");
      article.append(paragraph(`${progressLabels[turn.state]}……`));
      return article;
    }
    if (turn.state === "completed") {
      const outcome = outcomes[turn.outcome] || { label: "Help Desk" };
      const article = bubble("assistant", outcome.label);
      if (turn.answer) {
        for (const part of turn.answer.parts) {
          const element = paragraph(part.text);
          const markers = document.createElement("span");
          markers.className = "citation-marker";
          markers.textContent = part.citations.map((number) => ` [${number}]`).join("");
          element.append(markers);
          article.append(element);
        }
        const list = citationList(turn.answer.citations);
        if (list) article.append(list);
        return article;
      }
      if (outcome.text) article.append(paragraph(outcome.text));
      if (outcome.hint) article.append(paragraph(outcome.hint, "message-hint"));
      const list = citationList(turn.restricted_citations);
      if (list) article.append(list);
      return article;
    }
    const article = bubble("assistant", "處理失敗", "failed");
    article.append(paragraph(failureText(turn.error)));
    return article;
  }

  function renderMessages() {
    elements.messages.replaceChildren();
    if (state.turns.length === 0) {
      elements.messages.append(elements.emptyState);
      return;
    }
    for (const turn of state.turns) {
      const user = bubble("user", "你");
      user.append(paragraph(turn.question));
      elements.messages.append(user, assistantBubble(turn));
    }
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }

  function currentConversation() {
    return state.conversations.find((item) => item.conversation_id === state.conversationId) || null;
  }

  function renderTitle() {
    const conversation = currentConversation();
    elements.title.textContent = conversation && conversation.title ? conversation.title : "新對話";
  }

  function renderConversations() {
    elements.list.replaceChildren();
    for (const conversation of state.conversations) {
      const item = document.createElement("li");
      item.className = "conversation-item";
      if (conversation.conversation_id === state.conversationId) item.classList.add("current");

      const open = document.createElement("button");
      open.type = "button";
      open.className = "conversation-open";
      open.textContent = conversation.title || "未命名的對話";
      open.addEventListener("click", () => { void openConversation(conversation.conversation_id); });

      const tools = document.createElement("div");
      tools.className = "conversation-tools";
      const rename = document.createElement("button");
      rename.type = "button";
      rename.className = "secondary";
      rename.textContent = "改名";
      rename.addEventListener("click", () => { void renameConversation(conversation); });
      const archive = document.createElement("button");
      archive.type = "button";
      archive.className = "secondary";
      archive.textContent = "封存";
      archive.addEventListener("click", () => { void archiveConversation(conversation); });
      tools.append(rename, archive);

      item.append(open, tools);
      elements.list.append(item);
    }
    elements.listEmpty.classList.toggle("hidden", state.conversations.length > 0);
    elements.loadMore.classList.toggle("hidden", !state.nextCursor);
    renderTitle();
    refreshControls();
  }

  async function requestJson(path, options = {}, timeoutMs = 30000) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        ...options,
        credentials: "same-origin",
        signal: controller.signal,
        headers: { Accept: "application/json", ...(options.headers || {}) },
      });
      let payload = null;
      try {
        payload = await response.json();
      } catch (_error) {
        throw new ApiError(response.status, null, response.status >= 500);
      }
      if (!response.ok) throw new ApiError(response.status, payload, response.status >= 500);
      return { status: response.status, payload };
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, null, true);
    } finally {
      window.clearTimeout(timer);
    }
  }

  function jsonRequest(method, body, headers = {}) {
    return { method, headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) };
  }

  function errorText(error) {
    if (error.status === 0) return "無法確認伺服器是否收到請求，請使用「確認原請求結果」。";
    if (error.status === 401) return "身分未被接受。存取控制開啟時需要開發用身分 header。";
    if (error.status === 404) return "找不到這個對話或問題。";
    if (error.status === 409 && error.payload && error.payload.error === "conversation_archived") {
      return "這個對話已封存，請開新對話再問。";
    }
    if (error.status === 409) return "同一個請求代碼已用於不同的問題，請重新送出。";
    if (error.status === 413) return "問題太長，請縮短後再問。";
    if (error.status === 422) return "問題格式不正確。";
    return "服務暫時無法完成請求，請稍後再試。";
  }

  function clearPolling() {
    if (state.pollTimer !== null) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function upsertTurn(turn) {
    if (turn.conversation_id !== state.conversationId) return;
    const index = state.turns.findIndex((item) => item.turn_id === turn.turn_id);
    if (index === -1) state.turns.push(turn);
    else state.turns[index] = turn;
    renderMessages();
  }

  function finishActive(turn) {
    const active = state.active;
    state.active = null;
    state.pollDeadline = 0;
    clearPolling();
    if (turn && turn.state === "completed") {
      const outcome = outcomes[turn.outcome];
      setStatus(outcome ? [outcome.label, outcome.hint].filter(Boolean).join("。") : "完成。");
    } else if (turn) {
      setStatus(failureText(turn.error), true);
      if (active && !elements.question.value) {
        elements.question.value = active.question;
        updateCounter();
      }
    }
    refreshControls();
  }

  function trackTurn(turn) {
    upsertTurn(turn);
    if (OPEN_STATES.has(turn.state)) {
      setStatus(`${progressLabels[turn.state]}，請稍候……`);
      state.pollTimer = window.setTimeout(() => { void pollTurn(); }, POLL_INTERVAL_MS);
      return;
    }
    finishActive(turn);
  }

  async function pollTurn() {
    const active = state.active;
    if (!active || !active.statusUrl) return;
    if (Date.now() >= state.pollDeadline) {
      active.uncertain = true;
      setStatus("等候時間過長，結果尚未確認；可以使用「確認原請求結果」再查一次。", true);
      refreshControls();
      return;
    }
    try {
      const result = await requestJson(active.statusUrl);
      if (state.active !== active) return;
      trackTurn(result.payload);
    } catch (error) {
      if (state.active !== active) return;
      if (error.uncertain) {
        active.uncertain = true;
        setStatus("查詢結果不明；可以使用「確認原請求結果」再查一次。", true);
        refreshControls();
        return;
      }
      setStatus(errorText(error), true);
      finishActive(null);
    }
  }

  async function postTurn(active) {
    const result = await requestJson(
      `/api/v1/conversations/${active.conversationId}/turns`,
      jsonRequest("POST", active.body, { "Idempotency-Key": active.idempotencyKey }),
    );
    active.statusUrl = result.payload.status_url;
    state.pollDeadline = Date.now() + POLL_MAX_MS;
    trackTurn(result.payload);
  }

  async function createConversation(question) {
    const title = [...question.trim().replace(/\s+/g, " ")].slice(0, TITLE_LENGTH).join("");
    const result = await requestJson("/api/v1/conversations", jsonRequest("POST", { title: title || null }));
    state.conversations.unshift(result.payload);
    state.conversationId = result.payload.conversation_id;
    renderConversations();
    return state.conversationId;
  }

  function moveToTop(conversationId) {
    const index = state.conversations.findIndex((item) => item.conversation_id === conversationId);
    if (index > 0) {
      const [conversation] = state.conversations.splice(index, 1);
      state.conversations.unshift(conversation);
      renderConversations();
    }
  }

  async function sendQuestion() {
    if (state.active) return;
    const question = elements.question.value;
    const length = [...question].length;
    if (length === 0 || !question.trim()) {
      setStatus("請先輸入問題。", true);
      return;
    }
    if (length > MAX_QUESTION_LENGTH) {
      setStatus("問題不可超過 20,000 字元。", true);
      return;
    }
    state.active = {
      conversationId: null,
      idempotencyKey: newUuid(),
      question,
      body: { question, language: "zh-TW" },
      statusUrl: null,
      uncertain: false,
    };
    refreshControls();
    setStatus("送出中……");
    const active = state.active;
    try {
      active.conversationId = state.conversationId || (await createConversation(question));
      await postTurn(active);
      moveToTop(active.conversationId);
      elements.question.value = "";
      updateCounter();
    } catch (error) {
      if (error.uncertain && active.conversationId) {
        active.uncertain = true;
      } else if (state.active === active) {
        state.active = null;
      }
      setStatus(errorText(error), true);
      refreshControls();
    }
  }

  async function confirmRequest() {
    const active = state.active;
    if (!active || !active.uncertain) return;
    active.uncertain = false;
    refreshControls();
    setStatus("正在確認原請求結果……");
    if (active.statusUrl) {
      state.pollDeadline = Date.now() + POLL_MAX_MS;
      await pollTurn();
      return;
    }
    try {
      await postTurn(active);
      elements.question.value = "";
      updateCounter();
    } catch (error) {
      if (error.uncertain) {
        active.uncertain = true;
      } else {
        state.active = null;
      }
      setStatus(errorText(error), true);
      refreshControls();
    }
  }

  async function loadConversations(more = false) {
    const cursor = more && state.nextCursor ? `&cursor=${encodeURIComponent(state.nextCursor)}` : "";
    try {
      const result = await requestJson(`/api/v1/conversations?limit=${PAGE_SIZE}${cursor}`);
      const page = result.payload;
      state.conversations = more ? [...state.conversations, ...page.conversations] : page.conversations;
      state.nextCursor = page.next_cursor;
      renderConversations();
    } catch (error) {
      setStatus(errorText(error), true);
    }
  }

  async function openConversation(conversationId) {
    if (busy() || conversationId === state.conversationId) return;
    state.active = null;
    clearPolling();
    state.conversationId = conversationId;
    state.turns = [];
    renderConversations();
    renderMessages();
    setStatus("載入對話……");
    try {
      const result = await requestJson(`/api/v1/conversations/${conversationId}/turns`);
      if (state.conversationId !== conversationId) return;
      state.turns = result.payload.turns;
      renderMessages();
      const open = state.turns.find((turn) => OPEN_STATES.has(turn.state));
      if (open) {
        state.active = {
          conversationId,
          idempotencyKey: null,
          question: open.question,
          body: null,
          statusUrl: open.status_url,
          uncertain: false,
        };
        state.pollDeadline = Date.now() + POLL_MAX_MS;
        refreshControls();
        trackTurn(open);
      } else {
        setStatus("已載入對話，可以繼續追問。");
      }
    } catch (error) {
      setStatus(errorText(error), true);
    }
  }

  function newConversation() {
    if (busy()) return;
    state.active = null;
    clearPolling();
    state.conversationId = null;
    state.turns = [];
    elements.question.value = "";
    updateCounter();
    renderConversations();
    renderMessages();
    setStatus("已準備新的對話。");
  }

  async function renameConversation(conversation) {
    const title = window.prompt("新的對話名稱", conversation.title || "");
    if (title === null) return;
    if (!title.trim()) {
      setStatus("對話名稱不可空白。", true);
      return;
    }
    try {
      const result = await requestJson(
        `/api/v1/conversations/${conversation.conversation_id}`,
        jsonRequest("PATCH", { title: title.trim().slice(0, 200) }),
      );
      Object.assign(conversation, result.payload);
      renderConversations();
      setStatus("已改名。");
    } catch (error) {
      setStatus(errorText(error), true);
    }
  }

  async function archiveConversation(conversation) {
    if (!window.confirm("封存後這個對話會從清單移除，也不能再提問。確定要封存嗎？")) return;
    try {
      await requestJson(
        `/api/v1/conversations/${conversation.conversation_id}`,
        jsonRequest("PATCH", { status: "archived" }),
      );
      state.conversations = state.conversations.filter(
        (item) => item.conversation_id !== conversation.conversation_id,
      );
      if (conversation.conversation_id === state.conversationId) newConversation();
      renderConversations();
      setStatus("已封存對話。");
    } catch (error) {
      setStatus(errorText(error), true);
    }
  }

  elements.question.addEventListener("input", updateCounter);
  elements.question.addEventListener("compositionstart", () => { state.composing = true; });
  elements.question.addEventListener("compositionend", () => { state.composing = false; });
  elements.question.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !event.isComposing && !state.composing) {
      event.preventDefault();
      void sendQuestion();
    }
  });
  elements.send.addEventListener("click", () => { void sendQuestion(); });
  elements.confirm.addEventListener("click", () => { void confirmRequest(); });
  elements.newConversation.addEventListener("click", newConversation);
  elements.loadMore.addEventListener("click", () => { void loadConversations(true); });
  updateCounter();
  renderMessages();
  void loadConversations();
})();
