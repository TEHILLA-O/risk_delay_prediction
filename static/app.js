/**
 * PMO Delay Risk UI — calls POST /explain and renders gauge + SHAP bar chart.
 */

let chartRisk = null;
let chartShap = null;

function riskAccentColor(pct) {
  if (pct <= 33) return "#3ecf8e";
  if (pct <= 66) return "#f0c14b";
  return "#f07178";
}

function showToast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => {
    el.hidden = true;
  }, 6000);
}

function destroyChart(ch) {
  if (ch) {
    ch.destroy();
  }
  return null;
}

function renderRiskChart(pct) {
  const ctx = document.getElementById("chart-risk");
  const rest = Math.max(0, 100 - pct);
  const color = riskAccentColor(pct);

  chartRisk = destroyChart(chartRisk);

  chartRisk = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Delay risk", "Remaining"],
      datasets: [
        {
          data: [pct, rest],
          backgroundColor: [color, "#2a3544"],
          borderWidth: 0,
          hoverOffset: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "72%",
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: { color: "#9fb0c3", boxWidth: 12 },
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              const v = ctx.raw;
              return ctx.dataIndex === 0 ? `${v}% estimated delay risk` : `${v}% on-time headroom`;
            },
          },
        },
      },
    },
  });
}

function renderShapChart(shapValues, featureLabels) {
  const ctx = document.getElementById("chart-shap");
  const entries = Object.keys(shapValues).map((key) => ({
    key,
    label: featureLabels[key] || key,
    value: shapValues[key],
  }));
  entries.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const labels = entries.map((e) => e.label);
  const values = entries.map((e) => e.value);
  const colors = values.map((v) => (v >= 0 ? "rgba(240, 113, 120, 0.85)" : "rgba(62, 207, 142, 0.85)"));

  chartShap = destroyChart(chartShap);

  chartShap = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "SHAP (impact on delay probability)",
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(ctx) {
              const v = ctx.raw;
              const sign = v >= 0 ? "increases" : "decreases";
              return ` ${sign} predicted risk (${v >= 0 ? "+" : ""}${v.toFixed(4)})`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.06)" },
          ticks: { color: "#9fb0c3" },
          title: {
            display: true,
            text: "Contribution toward delay probability",
            color: "#9fb0c3",
            font: { size: 11 },
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#c5d1e0" },
        },
      },
    },
  });
}

async function analyze(payload) {
  const res = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText || "Request failed";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function formToPayload(form) {
  return {
    team_size: Number(form.team_size.value),
    deadline_days: Number(form.deadline_days.value),
    budget: Number(form.budget.value),
    task_complexity: Number(form.task_complexity.value),
    dependencies_count: Number(form.dependencies_count.value),
  };
}

function renderInputsSummary(payload, featureLabels) {
  const box = document.getElementById("inputs-summary");
  const keys = [
    "team_size",
    "deadline_days",
    "budget",
    "task_complexity",
    "dependencies_count",
  ];
  const fmt = (k, v) => {
    if (k === "budget") return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(v);
    if (k === "task_complexity") return Number(v).toLocaleString(undefined, { maximumFractionDigits: 1 });
    return String(v);
  };
  box.innerHTML = keys
    .map(
      (k) =>
        `<div class="kv"><span class="k">${featureLabels[k] || k}</span><span class="v">${fmt(k, payload[k])}</span></div>`
    )
    .join("");
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("risk-form");
  const results = document.getElementById("results");
  const btn = document.getElementById("submit-btn");
  const pctDisplay = document.getElementById("pct-display");
  const probSub = document.getElementById("prob-sub");
  const explanationText = document.getElementById("explanation-text");
  const gaugeLabel = document.getElementById("gauge-label");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    results.hidden = true;

    try {
      const payload = formToPayload(form);
      const data = await analyze(payload);

      const pct = data.delay_risk_percent;
      const p = data.delay_risk_probability;

      pctDisplay.textContent = String(pct);
      pctDisplay.style.color = riskAccentColor(pct);
      probSub.textContent = `Raw model output: ${(p * 100).toFixed(2)}% probability of delay (binary target).`;
      explanationText.textContent = data.explanation;
      gaugeLabel.textContent = `${pct}%`;
      gaugeLabel.style.color = riskAccentColor(pct);

      renderInputsSummary(payload, data.feature_labels);
      renderRiskChart(pct);
      renderShapChart(data.shap_values, data.feature_labels);

      results.hidden = false;
    } catch (err) {
      showToast(err.message || "Could not analyze. Train the model (python train.py) and retry.");
    } finally {
      btn.disabled = false;
    }
  });
});
