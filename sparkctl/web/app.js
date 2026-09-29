const $ = (s) => document.querySelector(s);
// 对话历史只留在浏览器里。语言要求由服务端加到最后一条用户消息上。
const messages = [];
let images = [];

function flash(text) {
  const el = $("#flash");
  el.hidden = false;
  el.textContent = text;
  setTimeout(() => { el.hidden = true; }, 4000);
}

async function refresh() {
  const res = await fetch("/api/state");
  const data = await res.json();
  const pill = $("#serve-pill");
  pill.textContent = data.ready ? "ready" : "stopped";
  pill.className = "pill" + (data.ready ? " ready" : "");
  $("#nodes").innerHTML = data.nodes.map((n) => `
    <article class="card">
      <h2>${n.name}</h2>
      <p class="meta">${n.role} · ${n.host}</p>
      <p>${n.gpu}</p>
    </article>`).join("");
  $("#model-rows").innerHTML = `
    <tr>
      <td>${data.model}</td>
      <td>TP=2 · 上下文 ${(data.max_model_len || 1048576).toLocaleString("zh-CN")}</td>
      <td>${data.speculative.method} · ${data.speculative.num_speculative_tokens}</td>
      <td>vision + aligner</td>
      <td><button type="button" class="primary" id="btn-start-row">启动</button></td>
    </tr>`;
  const limit = data.max_model_len || 1048576;
  $("#chat-max-tokens").max = String(limit);
  $("#chat-meta").textContent = data.ready
    ? `模型已就绪。上下文 ${limit.toLocaleString("zh-CN")}，生成长度可以填到这个数，提示词占用的部分会自动扣掉`
    : "模型未加载";
  $("#snippet").textContent = [
    `OpenAI  ${data.api}`,
    `控制台  ${data.console}`,
    `模型    ${data.model}`,
    `curl ${data.api}/chat/completions \\`,
    `  -H 'Content-Type: application/json' \\`,
    `  -d '{"model":"${data.model}","messages":[{"role":"user","content":"你好"}]}'`,
  ].join("\n");
  $("#btn-start-row")?.addEventListener("click", start);
}

async function start() {
  $("#serve-pill").textContent = "starting";
  $("#serve-pill").className = "pill starting";
  flash("正在拉起双机推理，权重加载需要几分钟");
  const res = await fetch("/api/serve/start", { method: "POST" });
  const data = await res.json();
  if (!data.ok) flash(data.output || "启动失败");
  refresh();
}

async function stop() {
  await fetch("/api/serve/stop", { method: "POST" });
  flash("已停止");
  refresh();
}

async function logs(which) {
  const res = await fetch("/api/logs?which=" + which);
  const data = await res.json();
  $("#log-box").textContent = data.text || "暂无日志";
}

function render() {
  $("#chat-thread").innerHTML = messages.map((m) =>
    `<p class="msg ${m.role}"><strong>${m.role === "user" ? "你" : "模型"}</strong>\n${m.content}</p>`
  ).join("");
  $("#chat-previews").innerHTML = images.map((u) =>
    u.startsWith("data:image/heic") || u.startsWith("data:image/heif")
      ? `<span class="preview heic">HEIC</span>`
      : `<img class="preview" src="${u}" alt="" />`
  ).join("");
}

$("#chat-images").addEventListener("change", async (ev) => {
  images = [];
  for (const file of ev.target.files) {
    try {
      images.push(await fileToJpeg(file));
    } catch (err) {
      flash(err.message || "无法读取这张图片，请换成 JPG、PNG 或 HEIC");
    }
  }
  render();
});

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("无法读取这张图片，请换成 JPG、PNG 或 HEIC"));
    reader.readAsDataURL(file);
  });
}

function isHeic(file) {
  const type = (file.type || "").toLowerCase();
  const name = (file.name || "").toLowerCase();
  return type === "image/heic" || type === "image/heif" || name.endsWith(".heic") || name.endsWith(".heif");
}

async function fileToJpeg(file) {
  // JPG/PNG 原样上传。浏览器能画出来的格式在本地转成 JPEG。
  // Chrome 解不开 HEIC，这种文件原样交给服务端再转。
  const type = (file.type || "").toLowerCase();
  if (type === "image/jpeg" || type === "image/jpg" || type === "image/png") {
    return readFile(file);
  }
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    if (bitmap.close) bitmap.close();
    return canvas.toDataURL("image/jpeg", 0.92);
  } catch (err) {
    if (!isHeic(file)) {
      throw new Error("无法读取这张图片，请换成 JPG、PNG 或 HEIC");
    }
    const url = await readFile(file);
    const comma = url.indexOf(",");
    return "data:image/heic;base64," + (comma >= 0 ? url.slice(comma + 1) : url);
  }
}

function errorText(error) {
  if (!error) return "请求失败";
  if (typeof error === "string") {
    try {
      const inner = JSON.parse(error);
      return inner.error?.message || error;
    } catch (err) {
      return error;
    }
  }
  return error.message || "请求失败";
}

$("#chat-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("#chat-input").value.trim();
  if (!text && !images.length) return;
  messages.push({ role: "user", content: text });
  const pending = { role: "assistant", content: "" };
  messages.push(pending);
  render();
  $("#chat-input").value = "";
  const sent = images.slice();
  images = [];
  $("#chat-images").value = "";
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.filter((m) => m !== pending),
      thinking: $("#chat-think").checked,
      reasoning_effort: $("#chat-effort").value,
      language: $("#chat-lang").value,
      max_tokens: Number($("#chat-max-tokens").value || 512),
      images: sent,
    }),
  });
  if (!res.ok || !res.body) {
    let msg = "请求失败";
    try { msg = errorText((await res.json()).error); } catch (err) { /* keep default */ }
    pending.content = msg;
    render();
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (!line || line.includes("[DONE]")) continue;
      try {
        const chunk = JSON.parse(line.slice(6));
        const choice = (chunk.choices || [])[0] || {};
        pending.content += choice.text || (choice.delta || {}).content || "";
      } catch (err) { /* ignore keepalives */ }
    }
    render();
  }
  } catch (err) {
    if (!pending.content) pending.content = "请求失败";
    render();
  }
});

$("#chat-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("#chat-form").requestSubmit();
  }
});

$("#chat-clear").addEventListener("click", () => { messages.length = 0; images = []; render(); });
$("#btn-start").addEventListener("click", start);
$("#btn-stop").addEventListener("click", stop);
$("#copy-url").addEventListener("click", () => navigator.clipboard.writeText("http://192.168.0.78:8888/v1"));
$("#btn-log-head").addEventListener("click", () => logs("head"));
$("#btn-log-worker").addEventListener("click", () => logs("worker"));
document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button, .panel").forEach((el) => el.classList.remove("on"));
    btn.classList.add("on");
    $("#tab-" + btn.dataset.tab).classList.add("on");
  });
});
refresh();
setInterval(refresh, 8000);
