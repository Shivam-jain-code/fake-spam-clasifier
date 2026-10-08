/* SpamGuard AI – frontend logic */
const $ = (id) => document.getElementById(id);
const input = $("emailInput"), result = $("result"), checkBtn = $("checkBtn");
const HISTORY_KEY = "spamguard_history";

/* ---------- Toasts ---------- */
function toast(msg, type = "info") {
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 3200);
}

/* ---------- Theme toggle (remembered in localStorage) ---------- */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("themeBtn").innerHTML = `<i class="fa-solid ${theme === "dark" ? "fa-moon" : "fa-sun"}"></i>`;
  localStorage.setItem("spamguard_theme", theme);
}
setTheme(localStorage.getItem("spamguard_theme") || "dark");
$("themeBtn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

/* ---------- Mobile menu ---------- */
$("burger").onclick = () => $("menu").classList.toggle("open");
$("menu").onclick = () => $("menu").classList.remove("open");

/* ---------- Textarea helpers ---------- */
input.addEventListener("input", () => ($("counter").textContent = `${input.value.length} / 10000`));

$("clearBtn").onclick = () => {
  input.value = "";
  input.dispatchEvent(new Event("input"));
  result.hidden = true;
  input.focus();
};

$("copyBtn").onclick = async () => {
  if (!input.value.trim()) return toast("Nothing to copy yet.", "error");
  try { await navigator.clipboard.writeText(input.value); toast("Email copied to clipboard.", "success"); }
  catch { input.select(); document.execCommand("copy"); toast("Email copied to clipboard.", "success"); }
};

document.querySelectorAll(".example").forEach((btn) =>
  btn.addEventListener("click", () => {
    input.value = btn.dataset.text;
    input.dispatchEvent(new Event("input"));
    result.hidden = true;
    toast("Example loaded. Click Check for Spam.");
    input.scrollIntoView({ behavior: "smooth", block: "center" });
  })
);

/* ---------- Prediction ---------- */
function showMessage(text) {            // error box inside the result area
  result.hidden = false;
  result.className = "result error";
  result.textContent = text;
}

function showResult(d) {
  const spam = d.prediction === "Spam";
  const pct = Math.round(d.confidence * 100);
  result.hidden = false;
  result.className = `result ${spam ? "spam" : "safe"}`;
  result.innerHTML = `
    <h3>${spam ? "🚨 SPAM DETECTED" : "✅ EMAIL IS SAFE"}</h3>
    <p>${spam ? "This email appears to be suspicious or unwanted." : "This email does not appear to be spam."}</p>
    <dl>
      <div><dt>Prediction</dt><dd>${d.prediction}</dd></div>
      <div><dt>Confidence</dt><dd>${pct}%</dd></div>
      <div><dt>Risk Level</dt><dd>${d.risk}</dd></div>
    </dl>
    <div class="bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"><i></i></div>`;
  requestAnimationFrame(() => requestAnimationFrame(() => (result.querySelector(".bar i").style.width = pct + "%")));
}

async function checkEmail() {
  const email = input.value.trim();
  if (!email) { showMessage("Please enter an email message first."); return toast("Please enter an email message first.", "error"); }

  checkBtn.disabled = true;
  checkBtn.querySelector(".spinner").hidden = false;
  checkBtn.querySelector(".label").textContent = "Analyzing...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    const data = await res.json();
    if (!res.ok) { showMessage(data.error || "Something went wrong."); return; }
    showResult(data);
    saveHistory(email, data);
  } catch {
    showMessage("Unable to connect to the prediction server. Please try again.");
    toast("Unable to connect to the prediction server. Please try again.", "error");
  } finally {
    checkBtn.disabled = false;
    checkBtn.querySelector(".spinner").hidden = true;
    checkBtn.querySelector(".label").textContent = "Check for Spam";
  }
}
checkBtn.onclick = checkEmail;
input.addEventListener("keydown", (e) => { if (e.ctrlKey && e.key === "Enter") checkEmail(); });

/* ---------- Prediction history (localStorage) ---------- */
const getHistory = () => { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; } };

function saveHistory(email, d) {
  const list = getHistory();
  list.unshift({ email, prediction: d.prediction, confidence: d.confidence, risk: d.risk });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 50)));
  renderHistory();
}

function renderHistory() {
  const q = $("searchHistory").value.toLowerCase();
  const body = $("historyBody");
  body.innerHTML = "";
  const rows = getHistory().filter((h) => h.email.toLowerCase().includes(q) || h.prediction.toLowerCase().includes(q));
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="4" class="empty">${q ? "No matches found." : "No predictions yet. Check an email above to start your history."}</td></tr>`;
    return;
  }
  rows.forEach((h) => {
    const tr = document.createElement("tr");
    const cls = h.prediction === "Spam" ? "spam" : "safe";
    const cells = [h.email, h.prediction, Math.round(h.confidence * 100) + "%", h.risk];
    cells.forEach((text, i) => {
      const td = document.createElement("td");
      if (i === 1 || i === 3) { const s = document.createElement("span"); s.className = `pill ${cls}`; s.textContent = text; td.appendChild(s); }
      else td.textContent = text;           // textContent keeps user text safe from XSS
      if (i === 0) td.title = text;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}
$("searchHistory").addEventListener("input", renderHistory);
$("clearHistory").onclick = () => { localStorage.removeItem(HISTORY_KEY); renderHistory(); toast("History cleared.", "success"); };
renderHistory();

/* ---------- Load REAL metrics + dataset info from the backend ---------- */
async function loadStats() {
  try {
    const m = await (await fetch("/api/stats")).json();
    $("heroAcc").textContent = m.accuracy + "%";
    $("algoName").textContent = m.algorithm;

    $("metricCards").innerHTML = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1 Score", m.f1]]
      .map(([l, v]) => `<div class="glass metric"><div class="num">${v}%</div><div class="lbl">${l}</div><div class="bar"><i style="width:${v}%"></i></div></div>`).join("");

    const fmt = (n) => n.toLocaleString();
    $("datasetCards").innerHTML = [["Total Emails", m.total], ["Spam Emails", m.spam], ["Ham Emails", m.ham], ["Training Samples", m.train], ["Testing Samples", m.test]]
      .map(([l, v]) => `<div class="glass metric"><div class="num">${fmt(v)}</div><div class="lbl">${l}</div></div>`).join("");
  } catch {
    toast("Could not load model statistics.", "error");
  }
}
loadStats();
