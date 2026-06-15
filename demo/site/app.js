async function loadResults() {
  const [resultsResponse, samplesResponse] = await Promise.all([
    fetch("./results.json"),
    fetch("./samples.json"),
  ]);
  const data = await resultsResponse.json();
  const samples = await samplesResponse.json();
  renderTable(data.systems);
  renderTargets(data.compact3_targets);
  initAudio(samples);
}

function formatWer(row) {
  if (row.wer_errors === null || row.wer_words === null) {
    return `${row.wer_percent.toFixed(2)}%`;
  }
  return `${row.wer_errors}/${row.wer_words} = ${row.wer_percent.toFixed(4)}%`;
}

function formatQuality(row) {
  if (row.utmos === null) {
    return "--";
  }
  return `${row.utmos.toFixed(4)} / ${row.dnsmos_ovrl.toFixed(4)} / ${row.dnsmos_p808.toFixed(4)}`;
}

function renderTable(systems) {
  const tbody = document.querySelector("#results-table tbody");
  tbody.innerHTML = "";
  systems.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.highlight) {
      tr.classList.add("best");
    }
    tr.innerHTML = `
      <td><strong>${row.name}</strong><br><span>${row.note}</span></td>
      <td>${row.steps}</td>
      <td>${row.ft === null ? "--" : row.ft ? "Yes" : "No"}</td>
      <td>${row.srfd === null ? "--" : row.srfd ? "Yes" : "No"}</td>
      <td><strong>${formatWer(row)}</strong></td>
      <td>${formatQuality(row)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTargets(targets) {
  const container = document.querySelector("#target-list");
  container.innerHTML = "";
  targets.forEach((target) => {
    const node = document.createElement("div");
    node.className = "target";
    node.innerHTML = `
      <strong>${target.name}</strong>
      <span>${target.extractor} · ${target.source}</span>
    `;
    container.appendChild(node);
  });
}

function formatSampleWer(value) {
  if (typeof value !== "number") {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function normalizeTranscript(value) {
  return String(value ?? "").trim().replace(/\s+/g, " ").toLowerCase();
}

function matchLabel(reference, hypothesis, wer) {
  const exact = normalizeTranscript(reference) === normalizeTranscript(hypothesis);
  if (wer === 0 && exact) {
    return {
      className: "match-note ok",
      text: "WER 0: ASR transcript exactly matches the reference transcript.",
    };
  }
  if (wer === 0) {
    return {
      className: "match-note ok",
      text: "WER 0 under the benchmark scorer.",
    };
  }
  return {
    className: "match-note diff",
    text: "Non-zero WER: compare the ASR transcript with the reference transcript.",
  };
}

function systemByKey(systems) {
  return Object.fromEntries(systems.map((system) => [system.key, system]));
}

function renderAudio(samplesData, view) {
  const container = document.querySelector("#audio-examples");
  const systems = samplesData.systems.filter((system) => view === "negative" || system.group === view);
  const lookup = systemByKey(samplesData.systems);
  const samples = samplesData.samples.filter((sample) => {
    const sampleCase = sample.case || "positive";
    return view === "negative" ? sampleCase === "negative" : sampleCase !== "negative";
  });
  container.innerHTML = "";

  samples.forEach((sample, index) => {
    const sampleNode = document.createElement("article");
    sampleNode.className = "example";
    const note = sample.note
      ? `<p class="case-note">${escapeHtml(sample.note)}</p>`
      : "";
    const label = view === "negative" ? `Negative Case ${index + 1}` : `Example ${index + 1}`;
    sampleNode.innerHTML = `
      <div class="example-head">
        <span>${label}</span>
        <strong>Reference transcript for WER: ${escapeHtml(sample.reference)}</strong>
        <p>Every model card below shows this same reference transcript together with that model's ASR transcript.</p>
        ${note}
        <div class="reference-audio">
          <div>
            <span>Prompt reference audio</span>
            <audio controls preload="none" src="${escapeHtml(sample.prompt_audio)}"></audio>
          </div>
          <div>
            <span>Target reference audio</span>
            <audio controls preload="none" src="${escapeHtml(sample.target_audio)}"></audio>
          </div>
        </div>
      </div>
      <div class="audio-grid"></div>
    `;

    const grid = sampleNode.querySelector(".audio-grid");
    const rows = sample.systems.filter((row) => systems.some((system) => system.key === row.key));
    rows.forEach((row) => {
      const system = lookup[row.key];
      const card = document.createElement("div");
      const match = matchLabel(sample.reference, row.hyp, row.wer);
      card.className = system.highlight ? "audio-card highlight" : "audio-card";
      card.innerHTML = `
        <div class="audio-card-head">
          <strong>${escapeHtml(system.name)}</strong>
          <span>${escapeHtml(system.detail)}</span>
        </div>
        <audio controls preload="none" src="${escapeHtml(row.audio)}"></audio>
        <dl>
          <div><dt>WER</dt><dd>${formatSampleWer(row.wer)}</dd></div>
          <div><dt>Full Set</dt><dd>${escapeHtml(system.metrics.wer)}</dd></div>
        </dl>
        <div class="transcripts">
          <div class="transcript">
            <span>Reference transcript</span>
            <p>${escapeHtml(sample.reference)}</p>
          </div>
          <div class="transcript">
            <span>ASR transcript</span>
            <p>${escapeHtml(row.hyp || "")}</p>
          </div>
        </div>
        <p class="${match.className}">${escapeHtml(match.text)}</p>
      `;
      grid.appendChild(card);
    });

    container.appendChild(sampleNode);
  });
}

function initAudio(samplesData) {
  let currentView = "core";
  const buttons = [...document.querySelectorAll("[data-audio-view]")];
  const setView = (view) => {
    currentView = view;
    buttons.forEach((button) => {
      button.classList.toggle("active", button.dataset.audioView === currentView);
    });
    renderAudio(samplesData, currentView);
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.audioView));
  });
  setView(currentView);
}

loadResults().catch((error) => {
  console.error(error);
  document.body.dataset.loadError = "true";
});
