(() => {
  const API = window.JIANPU_CONFIG.apiBase.replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const fileEl = $("file"), drop = $("drop"), dropText = $("dropText");
  const tonicEl = $("tonic"), modeEl = $("mode"), transposeEl = $("transpose");
  const formatEl = $("format");
  const go = $("go"), status = $("status"), backendNote = $("backendNote");

  let picked = null;

  const setStatus = (msg, cls = "") => { status.textContent = msg; status.className = "status " + cls; };

  function showFile(f) {
    if (!f) return;
    const ok = /\.(pdf|png|jpe?g)$/i.test(f.name);
    if (!ok) { setStatus("请选择 PDF / PNG / JPG 文件", "err"); return; }
    picked = f;
    drop.classList.add("has-file");
    const mb = (f.size / 1048576).toFixed(1);
    dropText.innerHTML = `<strong>${f.name}</strong><small>${mb} MB</small>`;
    go.disabled = false;
    setStatus("");
  }

  drop.addEventListener("click", () => fileEl.click());
  fileEl.addEventListener("change", (e) => showFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => showFile(e.dataTransfer.files[0]));

  async function loadOptions() {
    try {
      const r = await fetch(`${API}/api/keys`);
      if (!r.ok) throw new Error(r.status);
      const { tonics, modes } = await r.json();
      tonicEl.innerHTML = tonics.map((t) => `<option value="${t}">1 = ${t}</option>`).join("");
      modeEl.innerHTML = modes
        .map((m) => `<option value="${m.id}" ${m.enabled ? "" : "disabled"}>${m.label}${m.enabled ? "" : "（即将支持）"}</option>`)
        .join("");
      tonicEl.value = "C";
      backendNote.textContent = "后端已连接";
    } catch (e) {
      backendNote.textContent = `无法连接后端 ${API}（请检查 config.js / CORS）`;
      setStatus("后端未就绪，仍可尝试提交", "");
    }
  }

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!picked) return;
    go.disabled = true;
    const fmt = formatEl.value;
    const isImage = /\.(png|jpe?g)$/i.test(picked.name);
    if (fmt !== "jianpu" && isImage) {
      setStatus("原位版需要 PDF 输入（图片没有五线谱分格信息）", "err");
      go.disabled = false;
      return;
    }
    setStatus("正在识别并生成…（多页可能需数分钟）", "busy");

    const fd = new FormData();
    fd.append("file", picked);
    fd.append("tonic", tonicEl.value);
    fd.append("mode", modeEl.value);
    fd.append("transpose", transposeEl.checked ? "true" : "false");
    fd.append("format", fmt);

    try {
      const r = await fetch(`${API}/api/convert`, { method: "POST", body: fd });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { detail = (await r.json()).detail || detail; } catch (_) {}
        throw new Error(detail);
      }
      const key = r.headers.get("X-Output-Key") || tonicEl.value;
      const stem = picked.name.replace(/\.[^.]+$/, "") || "score";
      const ext = { both: "jianpu.zip", jianpu: "jianpu.pdf", insitu: "insitu.pdf" }[fmt];
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${stem}.${ext}`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      setStatus(`完成！输出调号 1=${key}。已开始下载 ${a.download}。`, "ok");
    } catch (err) {
      setStatus(`失败：${err.message}`, "err");
    } finally {
      go.disabled = false;
    }
  });

  loadOptions();
})();
