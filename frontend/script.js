/**
 * XENIA — XIM University Chatbot
 * script.js
 */

/* ── CONFIG ── */
const API_ENDPOINT = "http://localhost:8000/chat";
const USE_MOCK_API = false;


/* ── DOM REFS ── */
const chatWindow  = document.getElementById("chatWindow");
const messagesEl  = document.getElementById("messages");
const userInput   = document.getElementById("userInput");
const sendBtn     = document.getElementById("sendBtn");
const welcome     = document.getElementById("welcome");
const historyList = document.getElementById("historyList");
const newChatBtn  = document.getElementById("newChatBtn");
const statusDot   = document.getElementById("statusDot");
const statusText  = document.getElementById("statusText");


/* ── STATE ── */
let isLoading = false;
let conversationHistory = [];
let allConversations = [];
let currentSessionId = Date.now();


/* =========================================================
   CHAT HISTORY
   ========================================================= */
newChatBtn.addEventListener("click", () => {
  if (conversationHistory.length > 0) saveCurrentSession();
  startNewSession();
});

function startNewSession() {
  conversationHistory = [];
  currentSessionId = Date.now();
  messagesEl.innerHTML = "";
  welcome.style.display = "flex";
  userInput.value = "";
  updateSendBtn();
  userInput.focus();
}

function saveCurrentSession() {
  if (conversationHistory.length === 0) return;
  const firstUserMsg = conversationHistory.find(m => m.role === "user");
  const title = firstUserMsg ? truncate(firstUserMsg.content, 40) : "Conversation";
  allConversations.unshift({ id: currentSessionId, title, messages: [...conversationHistory] });
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = "";
  allConversations.forEach(session => {
    const li = document.createElement("li");
    li.className = "history-item";
    li.textContent = session.title;
    li.title = session.title;
    li.addEventListener("click", () => loadSession(session));
    historyList.appendChild(li);
  });
}

function loadSession(session) {
  if (conversationHistory.length > 0) saveCurrentSession();
  currentSessionId = session.id;
  conversationHistory = [...session.messages];
  messagesEl.innerHTML = "";
  welcome.style.display = "none";
  conversationHistory.forEach(msg =>
    appendBubble(msg.role === "user" ? "user" : "ai", msg.content, false)
  );
  scrollToBottom();
}


/* =========================================================
   INPUT
   ========================================================= */
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
  updateSendBtn();
});

userInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!isLoading && userInput.value.trim()) handleSend();
  }
});

sendBtn.addEventListener("click", () => {
  if (!isLoading && userInput.value.trim()) handleSend();
});

function updateSendBtn() {
  sendBtn.disabled = !userInput.value.trim() || isLoading;
}

document.getElementById("suggestions").addEventListener("click", e => {
  const chip = e.target.closest(".suggestion-chip");
  if (!chip) return;
  userInput.value = chip.dataset.q;
  userInput.dispatchEvent(new Event("input"));
  handleSend();
});


/* =========================================================
   SEND MESSAGE
   ========================================================= */
async function handleSend() {
  const text = userInput.value.trim();
  if (!text || isLoading) return;

  welcome.style.display = "none";
  appendBubble("user", text);
  conversationHistory.push({ role: "user", content: text });

  userInput.value = "";
  userInput.style.height = "auto";
  updateSendBtn();

  const typingEl = showTyping();
  isLoading = true;
  updateSendBtn();

  try {
    const answer = USE_MOCK_API
      ? await mockApiCall(text)
      : await sendToBackend(text);

    removeTyping(typingEl);
    await streamBubble("ai", answer);
    conversationHistory.push({ role: "assistant", content: answer });

  } catch (err) {
    removeTyping(typingEl);
    appendBubble("ai", `⚠️ Something went wrong: ${err.message}`);
  } finally {
    isLoading = false;
    updateSendBtn();
    userInput.focus();
  }
}


/* =========================================================
   BACKEND
   ========================================================= */
async function sendToBackend(message) {
  setApiStatus("connecting");

  const response = await fetch(API_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      history: conversationHistory.slice(-10),
    }),
  });

  if (!response.ok) {
    setApiStatus("error");
    throw new Error(`Server error ${response.status}`);
  }

  setApiStatus("connected");
  const data = await response.json();
  return data.answer ?? data.response ?? "No response from server.";
}

async function pingBackend() {
  if (USE_MOCK_API) return;
  try {
    const res = await fetch(API_ENDPOINT.replace("/chat", "/health"));
    setApiStatus(res.ok ? "connected" : "error");
  } catch {
    setApiStatus("disconnected");
  }
}
pingBackend();

function setApiStatus(state) {
  statusDot.className = "status-dot";
  if (state === "connected") {
    statusDot.classList.add("connected");
    statusText.textContent = "Backend: Connected";
  } else if (state === "connecting") {
    statusText.textContent = "Backend: Connecting…";
  } else if (state === "error") {
    statusDot.classList.add("error");
    statusText.textContent = "Backend: Error";
  } else {
    statusText.textContent = "Backend: Not connected";
  }
}


/* =========================================================
   BUBBLES
   ========================================================= */
function appendBubble(role, content, animate = true) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  if (!animate) row.style.animation = "none";

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "ai" ? "X" : "U";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = parseMarkdown(content);
  if (role === "ai") addCopyButton(bubble, content);

  const time = document.createElement("span");
  time.className = "msg-time";
  time.textContent = formatTime(new Date());

  if (role === "ai") {
    row.appendChild(avatar);
    row.appendChild(bubble);
    row.appendChild(time);
  } else {
    row.appendChild(time);
    row.appendChild(bubble);
    row.appendChild(avatar);
  }

  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

async function streamBubble(role, fullText) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "ai" ? "X" : "U";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const time = document.createElement("span");
  time.className = "msg-time";
  time.textContent = formatTime(new Date());

  row.appendChild(avatar);
  row.appendChild(bubble);
  row.appendChild(time);
  messagesEl.appendChild(row);

  let displayed = "";
  const words = fullText.split(" ");
  for (let i = 0; i < words.length; i++) {
    displayed += (i === 0 ? "" : " ") + words[i];
    bubble.innerHTML = parseMarkdown(displayed) + '<span style="opacity:0.5">▌</span>';
    scrollToBottom();
    await sleep(20 + Math.random() * 15);
  }

  bubble.innerHTML = parseMarkdown(fullText);
  addCopyButton(bubble, fullText);
  scrollToBottom();
}

function addCopyButton(bubble, content) {
  const actions = document.createElement("div");
  actions.className = "bubble-actions";
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.textContent = "Copy";
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(content).then(() => {
      btn.textContent = "Copied!";
      setTimeout(() => btn.textContent = "Copy", 1500);
    });
  });
  actions.appendChild(btn);
  bubble.appendChild(actions);
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "typing-row";
  row.innerHTML = `
    <div class="avatar ai">E</div>
    <div class="typing-bubble">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function removeTyping(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}


/* =========================================================
   HELPERS
   ========================================================= */
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + "…" : str;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function parseMarkdown(text) {
  if (!text) return "";
  let s = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  s = s.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
  s = s.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const paras = s.split(/\n{2,}/);
  return paras.length > 1
    ? paras.map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("")
    : s.replace(/\n/g, "<br>");
}


/* =========================================================
   MOCK API
   ========================================================= */
const MOCK_RESPONSES = [
  "XIM University offers MBA, PGDM, PhD, and specialized programs in HRM, Rural Management, and Business Management.",
  "Admissions involve a national entrance exam (CAT/XAT), followed by GD and PI rounds.",
  "Our campus in Bhubaneswar has a library, sports complex, hostels, cafeteria, and modern classrooms.",
  "We offer merit-based and need-based scholarships. Contact the financial aid office for details.",
];
let mockIdx = 0;
async function mockApiCall(message) {
  await sleep(1200 + Math.random() * 600);
  const lc = message.toLowerCase();
  if (lc.includes("program") || lc.includes("course")) return MOCK_RESPONSES[0];
  if (lc.includes("admission") || lc.includes("apply")) return MOCK_RESPONSES[1];
  if (lc.includes("campus") || lc.includes("facilit")) return MOCK_RESPONSES[2];
  if (lc.includes("scholarship") || lc.includes("fee")) return MOCK_RESPONSES[3];
  return MOCK_RESPONSES[mockIdx++ % MOCK_RESPONSES.length];
}


/* ── INIT ── */
updateSendBtn();
userInput.focus();
