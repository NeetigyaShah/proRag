// ProRag viewer — one file, no build step. Static page hits /chat/stream and
// pdf.js off a CDN renders the cited page + bbox highlight.

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.worker.min.js";

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const emptyChatEl = document.getElementById("empty-chat");
const fileInput = document.getElementById("file-input");
const docCountEl = document.getElementById("doc-count");

let chatId = null;
let streaming = false;

// ---------- upload ----------
document.getElementById("upload-btn").addEventListener("click", () => fileInput.click());
const emptyUploadBtn = document.getElementById("empty-upload-btn");
if (emptyUploadBtn) emptyUploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  docCountEl.textContent = `Ingesting ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const resp = await fetch("/ingest", { method: "POST", body: form });
    docCountEl.textContent = resp.ok ? `Added ${file.name}` : `Could not ingest ${file.name}`;
  } catch {
    docCountEl.textContent = `Could not ingest ${file.name}`;
  }
  fileInput.value = "";
});

// ---------- chat ----------
sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || streaming) return;
  inputEl.value = "";
  if (emptyChatEl) emptyChatEl.remove();

  appendTurn("You", "user").textContent = message;

  const answerEl = appendTurn("ProRag", "assistant");
  answerEl.classList.add("thinking");
  answerEl.textContent = "Reading your documents";
  const chipsEl = document.createElement("div");
  chipsEl.className = "chips";
  answerEl.after(chipsEl);

  streaming = true;
  sendBtn.disabled = true;
  let sources = [];
  let raw = "";

  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, chat_id: chatId }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const { event, data } = parseFrame(frame);
        if (data === null) continue;

        if (event === "sources") {
          sources = data;
          renderChips(chipsEl, sources);
        } else if (event === "token") {
          if (answerEl.classList.contains("thinking")) {
            answerEl.classList.remove("thinking");
            answerEl.textContent = "";
          }
          raw += data.t;
          renderAnswer(answerEl, raw, sources);
        } else if (event === "citation") {
          const chip = chipsEl.querySelector(`.chip[data-n="${data.n}"]`);
          if (chip) chip.classList.add("cited");
        } else if (event === "meta") {
          chatId = data.chat_id;
        }
      }
    }
  } catch (err) {
    answerEl.classList.remove("thinking");
    answerEl.classList.add("error");
    answerEl.textContent = `Something went wrong (${err.message}). Try asking again.`;
  } finally {
    streaming = false;
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

function parseFrame(frame) {
  let event = "message", data = null;
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) { try { data = JSON.parse(line.slice(5).trim()); } catch {} }
    // ":" heartbeat comments and "retry:" are ignored.
  }
  return { event, data };
}

function appendTurn(who, role) {
  const turn = document.createElement("div");
  turn.className = "turn";
  const whoEl = document.createElement("div");
  whoEl.className = "who";
  whoEl.textContent = who;
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  turn.append(whoEl, msg);
  messagesEl.appendChild(turn);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return msg;
}

// Re-render the streamed text with [Sn] markers as clickable superscripts.
function renderAnswer(el, raw, sources) {
  el.textContent = "";
  const parts = raw.split(/(\[S\d+\])/);
  for (const part of parts) {
    const m = part.match(/^\[S(\d+)\]$/);
    if (m) {
      const n = Number(m[1]);
      const sup = document.createElement("sup");
      sup.className = "cite";
      sup.textContent = `S${n}`;
      sup.title = "Open source";
      const src = sources.find((s) => s.n === n);
      if (src) sup.addEventListener("click", () => openSource(src));
      el.appendChild(sup);
    } else if (part) {
      el.appendChild(document.createTextNode(part));
    }
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderChips(chipsEl, sources) {
  chipsEl.textContent = "";
  for (const s of sources) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.dataset.n = s.n;
    chip.title = s.snippet || "";
    chip.innerHTML = "";
    chip.textContent = `S${s.n} ${s.title || s.doc_id}`;
    if (s.page) {
      const pg = document.createElement("span");
      pg.className = "pg";
      pg.textContent = `p.${s.page}`;
      chip.appendChild(pg);
    }
    chip.addEventListener("click", () => openSource(s));
    chipsEl.appendChild(chip);
  }
}

// ---------- viewer ----------
const slipEl = document.getElementById("slip");
const slipTitle = document.getElementById("slip-title");
const slipPage = document.getElementById("slip-page");
const slipDownload = document.getElementById("slip-download");
const emptyViewer = document.getElementById("empty-viewer");
const stage = document.getElementById("pdf-stage");
const canvas = document.getElementById("pdf-canvas");

let currentPdf = null;
let currentPage = 1;
let currentBbox = null;
let currentBboxPage = null;

document.getElementById("page-prev").addEventListener("click", () => gotoPage(currentPage - 1));
document.getElementById("page-next").addEventListener("click", () => gotoPage(currentPage + 1));

async function openSource(source) {
  slipEl.hidden = false;
  emptyViewer.hidden = true;
  stage.hidden = false;
  slipTitle.textContent = source.title || source.doc_id;
  slipDownload.href = `/files/${source.doc_id}/original`;

  currentPdf = await pdfjsLib.getDocument(`/files/${source.doc_id}/original`).promise;
  currentBbox = source.bbox && source.bbox.length === 4 ? source.bbox : null;
  currentBboxPage = source.page || 1;
  await gotoPage(source.page || 1);
}

async function gotoPage(pageNo) {
  if (!currentPdf || pageNo < 1 || pageNo > currentPdf.numPages) return;
  currentPage = pageNo;
  const page = await currentPdf.getPage(pageNo);
  const scale = Math.min(1.6, (stage.parentElement.clientWidth - 48) / page.getViewport({ scale: 1 }).width);
  const viewport = page.getViewport({ scale });

  canvas.width = viewport.width;
  canvas.height = viewport.height;
  await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
  slipPage.textContent = `${pageNo} / ${currentPdf.numPages}`;

  stage.querySelectorAll("#highlight").forEach((el) => el.remove());
  if (currentBbox && pageNo === currentBboxPage) {
    const [x0, y0, x1, y1] = currentBbox;
    // PDF points -> viewport pixels; PDF y-axis grows up, canvas grows down.
    const rect = document.createElement("div");
    rect.id = "highlight";
    rect.style.left = `${x0 * viewport.scale}px`;
    rect.style.top = `${viewport.height - y1 * viewport.scale}px`;
    rect.style.width = `${(x1 - x0) * viewport.scale}px`;
    rect.style.height = `${(y1 - y0) * viewport.scale}px`;
    stage.appendChild(rect);
  }
}
