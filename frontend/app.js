async function runSimulation() {

  const mode = document.getElementById("mode").value;

  const body =
    mode === "synthetic"
      ? { mode: "synthetic" }
      : { mode: "user", sequence: userSequence };

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const data = await res.json();

  animateVolume(data.frames);
  plotMetrics(data.max_temp_log, data.cluster_count_log);
  showRisk(data);
}

function animateVolume(frames) {
  const T = frames.length;
  const size = frames[0].length;

  let X = [], Y = [], Z = [];
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      for (let z = 0; z < size; z++) {
        X.push(x); Y.push(y); Z.push(z);
      }
    }
  }

  const initial = {
    type: "volume",
    x: X,
    y: Y,
    z: Z,
    value: frames[0].flat(2),
    isomin: 25,
    isomax: 250,
    opacity: 0.15,
    surface: { count: 20 },
    colorscale: "Turbo"
  };

  const plotFrames = frames
    .filter((_, i) => i % 4 === 0)
    .map((f, i) => ({
      name: i.toString(),
      data: [{ value: f.flat(2) }]
    }));

  Plotly.newPlot("volumePlot", [initial], {
    title: "3D Silo Thermal Evolution",
    scene: { aspectmode: "cube" },
    updatemenus: [{
      type: "buttons",
      buttons: [{
        label: "▶ Play",
        method: "animate",
        args: [null, { frame: { duration: 100, redraw: true } }]
      }]
    }],
    sliders: [{
      steps: plotFrames.map(f => ({
        label: f.name,
        method: "animate",
        args: [[f.name], { mode: "immediate" }]
      }))
    }]
  });

  Plotly.addFrames("volumePlot", plotFrames);
}

function plotMetrics(maxTemp, clusters) {
  const trace1 = {
    x: [...Array(maxTemp.length).keys()],
    y: maxTemp,
    name: "Max Temperature (°C)",
    yaxis: "y1",
    type: "scatter"
  };

  const trace2 = {
    x: [...Array(clusters.length).keys()],
    y: clusters,
    name: "Active Clusters",
    yaxis: "y2",
    type: "step"
  };

  Plotly.newPlot("chartPlot", [trace1, trace2], {
    title: "Silo Lifecycle Metrics",
    yaxis: { title: "Temperature (°C)" },
    yaxis2: {
      title: "Clusters",
      overlaying: "y",
      side: "right"
    }
  });
}

function showRisk({ risk_score, risk_level }) {
  const color =
    risk_level === "HIGH" ? "red" :
    risk_level === "MEDIUM" ? "orange" : "green";

  document.getElementById("riskBox").innerHTML =
    `<span style="color:${color}">
      ⚠️ Risk Level: ${risk_level} (${risk_score})
     </span>`;
}

