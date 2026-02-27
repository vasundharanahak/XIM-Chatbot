/**
 * ENVIE — XIM University Chatbot
 * script.js | Vanilla JS, production-quality
 *
 * ─────────────────────────────────────────────────────────
 * TO CONNECT A FASTAPI BACKEND:
 *   1. Set API_ENDPOINT to your server URL, e.g.:
 *      const API_ENDPOINT = "http://localhost:8000/chat";
 *   2. Set USE_MOCK_API = false.
 *   3. Your FastAPI endpoint should accept POST:
 *        { "message": "...", "history": [...] }
 *      and respond with JSON:
 *        { "answer": "..." }
 *   4. See sendToBackend() for the full request shape.
 * ─────────────────────────────────────────────────────────
 */

/* ── CONFIGURATION ── */
const API_ENDPOINT = "http://localhost:8000/chat";   // ← your FastAPI URL
const USE_MOCK_API = true;   // ← set false when backend is ready
const STREAM_MOCK  = true;   // ← simulate streaming in mock mode


/* ── DOM REFS ── */
const chatWindow    = document.getElementById("chatWindow");
const messagesEl    = document.getElementById("messages");
const userInput     = document.getElementById("userInput");
const sendBtn       = document.getElementById("sendBtn");
const welcome       = document.getElementById("welcome");
const themeToggle   = document.getElementById("themeToggle");
const sidebarEl     = document.getElementById("sidebar");
const sidebarOpen   = document.getElementById("sidebarOpen");
const sidebarClose  = document.getElementById("sidebarClose");
const overlay       = document.getElementById("overlay");
const newChatBtn    = document.getElementById("newChatBtn");
const historyList   = document.getElementById("historyList");
const statusDot     = document.getElementById("statusDot");
const statusText    = document.getElementById("statusText");


/* ── STATE ── */
let isLoading        = false;
let conversationHistory = [];   // { role: "user"|"assistant", content: "..." }
let allConversations = [];      // saved sessions for history panel
let currentSessionId = Date.now();


/* ── THEME ── */
const savedTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-theme", savedTheme);

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});


/* ── SIDEBAR TOGGLE ── */
function openSidebar() {
  sidebarEl.classList.add("open");
  overlay.classList.add("visible");
}
function closeSidebar() {
  sidebarEl.classList.remove("open");
  overlay.classList.remove("visible");
}
sidebarOpen.addEventListener("click", openSidebar);
sidebarClose.addEventListener("click", closeSidebar);
overlay.addEventListener("click", closeSidebar);


/* ── NEW CHAT ── */
newChatBtn.addEventListener("click", () => {
  if (conversationHistory.length > 0) {
    saveCurrentSession();
  }
  startNewSession();
  closeSidebar();
});

function startNewSession() {
  conversationHistory = [];
  currentSessionId = Date.now();
  messagesEl.innerHTML = "";
  welcome.style.display = "flex";
  userInput.value = "";
  updateSendBtn();
}

function saveCurrentSession() {
  if (conversationHistory.length === 0) return;
  const firstUserMsg = conversationHistory.find(m => m.role === "user");
  const title = firstUserMsg
    ? truncate(firstUserMsg.content, 40)
    : "Conversation";
  allConversations.unshift({ id: currentSessionId, title, messages: [...conversationHistory] });
  renderHistory();
}


/* ── HISTORY ── */
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
  if (conversationHistory.length > 0 && session.id !== currentSessionId) {
    saveCurrentSession();
  }
  currentSessionId = session.id;
  conversationHistory = [...session.messages];

  messagesEl.innerHTML = "";
  welcome.style.display = "none";

  conversationHistory.forEach(msg => {
    appendBubble(msg.role === "user" ? "user" : "ai", msg.content, false);
  });

  // highlight active
  document.querySelectorAll(".history-item").forEach(el => {
    el.classList.toggle("active", el.textContent === truncate(session.messages.find(m=>m.role==="user")?.content ?? "", 40));
  });

  closeSidebar();
  scrollToBottom();
}


/* ── AUTO-RESIZE TEXTAREA ── */
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
  updateSendBtn();
});


/* ── SEND ON ENTER (Shift+Enter = newline) ── */
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


/* ── SUGGESTION CHIPS ── */
document.getElementById("suggestions").addEventListener("click", e => {
  const chip = e.target.closest(".suggestion-chip");
  if (!chip) return;
  userInput.value = chip.dataset.q;
  userInput.dispatchEvent(new Event("input"));
  handleSend();
});


/* ── HANDLE SEND ── */
async function handleSend() {
  const text = userInput.value.trim();
  if (!text || isLoading) return;

  // Hide welcome
  welcome.style.display = "none";

  // Add user message
  appendBubble("user", text);
  conversationHistory.push({ role: "user", content: text });

  // Clear input
  userInput.value = "";
  userInput.style.height = "auto";
  updateSendBtn();

  // Show typing
  const typingEl = showTyping();
  isLoading = true;
  updateSendBtn();

  try {
    let answer;
    if (USE_MOCK_API) {
      answer = await mockApiCall(text);
    } else {
      answer = await sendToBackend(text);
    }

    removeTyping(typingEl);

    if (STREAM_MOCK && USE_MOCK_API) {
      await streamBubble("ai", answer);
    } else {
      appendBubble("ai", answer);
    }

    conversationHistory.push({ role: "assistant", content: answer });

  } catch (err) {
    removeTyping(typingEl);
    appendBubble("ai", `⚠️ Sorry, something went wrong: ${err.message}`);
  } finally {
    isLoading = false;
    updateSendBtn();
    userInput.focus();
  }
}


/* ── APPEND BUBBLE ── */
function appendBubble(role, content, animate = true) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  if (!animate) row.style.animation = "none";

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "ai" ? "E" : "U";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = parseMarkdown(content);

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


/* ── STREAM BUBBLE (simulated) ── */
async function streamBubble(role, fullText) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "ai" ? "E" : "U";

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
    bubble.innerHTML = parseMarkdown(displayed) + '<span class="cursor">▌</span>';
    scrollToBottom();
    await sleep(28 + Math.random() * 22);
  }
  bubble.innerHTML = parseMarkdown(fullText);
  scrollToBottom();
  return row;
}


/* ── TYPING INDICATOR ── */
function showTyping() {
  const row = document.createElement("div");
  row.className = "typing-row";
  row.innerHTML = `
    <div class="avatar ai">E</div>
    <div class="typing-bubble">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function removeTyping(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}


/* ── SCROLL ── */
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}


/* ── HELPERS ── */
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + "…" : str;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/** Minimal markdown parser (bold, italic, code, links) */
function parseMarkdown(text) {
  if (!text) return "";
  // Escape HTML first
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks
  safe = safe.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`);
  // Inline code
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Bold
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic
  safe = safe.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // Links
  safe = safe.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // Newlines → paragraphs
  const paragraphs = safe.split(/\n{2,}/);
  if (paragraphs.length > 1) {
    safe = paragraphs.map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
  } else {
    safe = safe.replace(/\n/g, "<br>");
  }
  return safe;
}


/* ─────────────────────────────────────────────────────────
   BACKEND CONNECTION
   ─────────────────────────────────────────────────────────

   sendToBackend() sends the user's message + conversation
   history to your FastAPI server.

   Expected FastAPI route (example):

     @app.post("/chat")
     async def chat(req: ChatRequest):
         # RAG logic here (your existing chatbot.py logic)
         return {"answer": "..."}

     class ChatRequest(BaseModel):
         message: str
         history: list[dict]   # [{role, content}, ...]

   To enable streaming from FastAPI:
     1. Use StreamingResponse in FastAPI.
     2. Replace the fetch below with an EventSource / SSE reader.
     3. Update streamBubble() to consume the stream in real-time.
   ───────────────────────────────────────────────────────── */
async function sendToBackend(message) {
  setApiStatus("connecting");

  const payload = {
    message,
    history: conversationHistory.slice(-10),   // last 10 turns for context
  };

  const response = await fetch(API_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    setApiStatus("error");
    const err = await response.text();
    throw new Error(`Server error ${response.status}: ${err}`);
  }

  setApiStatus("connected");
  const data = await response.json();
  return data.answer ?? data.response ?? data.message ?? "No response from server.";
}


/* ── API STATUS INDICATOR ── */
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

/* Call once on load to show ping status */
async function pingBackend() {
  if (USE_MOCK_API) return;
  try {
    const res = await fetch(API_ENDPOINT.replace("/chat", "/health"), { method: "GET" });
    setApiStatus(res.ok ? "connected" : "error");
  } catch {
    setApiStatus("disconnected");
  }
}
pingBackend();


/* ─────────────────────────────────────────────────────────
   MOCK API  (remove when backend is live)
   ─────────────────────────────────────────────────────────
   Replace this with sendToBackend() when your FastAPI
   server is running.
   ───────────────────────────────────────────────────────── */
const MOCK_RESPONSES = [
  "XIM University offers a variety of postgraduate and doctoral programs including MBA, PGDM, PhD, and specialized courses in Human Resource Management, Rural Management, and Business Management.",
  "The admission process at XIM University typically involves clearing a national entrance exam (like CAT/XAT), followed by a group discussion and personal interview round. Check the official admissions page for current cycle deadlines.",
  "XIM University is located in Bhubaneswar, Odisha. The campus features state-of-the-art facilities including a library, sports complex, hostels, cafeteria, and fully equipped classrooms with modern learning technology.",
  "XIM University offers merit-based scholarships and need-based financial aid. Specific schemes vary by program. Please consult the financial aid office or the official XIM website for updated scholarship criteria.",
  "The Faculty at XIM University comprises experienced academics and industry practitioners. Many faculty members hold doctoral degrees from reputed institutions and have published in leading journals.",
  "I found relevant information in the university documents. XIM University emphasizes value-based education rooted in Jesuit tradition, focusing on holistic development of students beyond just academic excellence.",
  "Based on the documents, XIM University's MBA program spans two years and covers core management disciplines with electives in your area of specialization. The curriculum is regularly updated in consultation with industry leaders.",
];

let mockIdx = 0;
async function mockApiCall(message) {
  await sleep(1200 + Math.random() * 800);
  const lc = message.toLowerCase();

  if (lc.includes("program") || lc.includes("course")) return MOCK_RESPONSES[0];
  if (lc.includes("admission") || lc.includes("apply")) return MOCK_RESPONSES[1];
  if (lc.includes("campus") || lc.includes("facilit")) return MOCK_RESPONSES[2];
  if (lc.includes("scholarship") || lc.includes("fee") || lc.includes("financial")) return MOCK_RESPONSES[3];
  if (lc.includes("faculty") || lc.includes("professor")) return MOCK_RESPONSES[4];
  if (lc.includes("mba") || lc.includes("curriculum")) return MOCK_RESPONSES[6];

  const r = MOCK_RESPONSES[mockIdx % MOCK_RESPONSES.length];
  mockIdx++;
  return r;
}


/* ── INIT ── */
updateSendBtn();
userInput.focus();
