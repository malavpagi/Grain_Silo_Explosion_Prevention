async function runSimulation() {
  const mode = document.getElementById("mode").value;
  const body = mode === "synthetic" 
    ? { mode: "synthetic" } 
    : { mode: "user", sequence: userSequence }; // Ensure userSequence is defined if used

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  const data = await res.json();

  // Call the new visualizer
  visualizeLifecycle(data.frames, data.max_temp_log, data.cluster_count_log);
  showRisk(data);
}

function showRisk({ risk_score, risk_level }) {
  let color = "red";
  if (risk_level === "SAFE" || risk_level === "VERY LOW") color = "blue";
  else if (risk_level === "LOW") color = "green";
  else if (risk_level === "MEDIUM") color = "orange";

  document.getElementById("riskBox").innerHTML =
    `<span style="color:${color}">
      ⚠️ Risk Level: ${risk_level} (${risk_score})
     </span>`;
}

// --- YOUR NEW FUNCTION (Adapted for IDs and Auto-Config) ---
function visualizeLifecycle(frames, max_log, count_log) {

  // 1. AUTO-DETECT CONFIG (No need to pass it manually)
  const STEPS = frames.length;
  const SIZE = frames[0].length; // Assuming cubic grid (e.g., 50)
  
  /* ===============================
       2D DASHBOARD (Temperature + Clusters)
    =============================== */
  // Use existing HTML ID "chartPlot"
  const chartId = "chartPlot"; 

  const timeSteps = [...Array(max_log.length).keys()];

  const tempTrace = {
    x: timeSteps,
    y: max_log,
    name: "Max Silo Temp (°C)",
    type: "scatter",
    mode: "lines",
    line: { color: "#d62728", width: 2 }, // Tab:red hex
    yaxis: "y1"
  };

  const ignitionLine = {
    x: [0, timeSteps.length - 1],
    y: [100, 100],
    name: "Ignition",
    type: "scatter",
    mode: "lines",
    line: { color: "orange", dash: "dot" },
    yaxis: "y1",
    hoverinfo: "skip"
  };

  const clusterTrace = {
    x: timeSteps,
    y: count_log,
    name: "Active Clusters (#)",
    type: "scatter",
    mode: "lines",
    line: { shape: "hv", color: "#1f77b4" }, // Tab:blue hex
    opacity: 0.6,
    yaxis: "y2"
  };

  const layout2D = {
    title: "Silo Lifecycle: Temperature vs. Active Clusters",
    xaxis: { title: "Time Steps" },
    yaxis: {
      title: "Max Silo Temp (°C)",
      titlefont: { color: "#d62728" },
      tickfont: { color: "#d62728" }
    },
    yaxis2: {
      title: "Active Clusters (#)",
      titlefont: { color: "#1f77b4" },
      tickfont: { color: "#1f77b4" },
      overlaying: "y",
      side: "right"
    },
    margin: { t: 50, l: 60, r: 60, b: 50 }
  };

  Plotly.newPlot(chartId, [tempTrace, ignitionLine, clusterTrace], layout2D);


  /* ===============================
       3D ANIMATION
    =============================== */
  // Use existing HTML ID "volumePlot"
  const volumeId = "volumePlot";
  console.log("Generating 3D Movie...");

  const X = [];
  const Y = [];
  const Z = [];

  // Use detected SIZE instead of config.X
  for (let x = 0; x < SIZE; x++) {
    for (let y = 0; y < SIZE; y++) {
      for (let z = 0; z < SIZE; z++) {
        X.push(x); Y.push(y); Z.push(z);
      }
    }
  }

  // Define Custom Gradient: Blue -> Green -> Orange -> Red
  const customColors = [
    [0.0, "blue"],   // Safe
    [0.33, "green"], 
    [0.66, "orange"],
    [1.0, "red"]     // Danger
  ];

  const initialVolume = {
    type: "volume",
    x: X,
    y: Y,
    z: Z,
    value: frames[0].flat(2),
    isomin: 25,
    isomax: 250,
    opacity: 0.2,
    surface: { count: 20 },
    // Switch to 'Turbo' here if you prefer that over the custom customColors
    colorscale: customColors 
  };

  const skip = 1;
  const animationFrames = [];

  for (let t = 0; t < STEPS; t += skip) {
    animationFrames.push({
      name: String(t),
      data: [{
        value: frames[t].flat(2),
        isomin: 25,
        isomax: 250
      }]
    });
  }

  const layout3D = {
    title: "200-Step Lifecycle: Emergence & Decay",
    scene: { aspectmode: "cube" },
    margin: { t: 40, l: 0, r: 0, b: 0 },
    updatemenus: [{
      type: "buttons",
      showactive: false,
      x: 0, y: 0,
      buttons: [{
        label: "▶ Play Lifecycle",
        method: "animate",
        args: [
          null,
          {
            frame: { duration: 100, redraw: true },
            fromcurrent: true
          }
        ]
      }]
    }],
    sliders: [{
      pad: {t: 30},
      steps: animationFrames.map(f => ({
        label: f.name,
        method: "animate",
        args: [
          [f.name],
          { mode: "immediate", frame: { duration: 0, redraw: true } }
        ]
      }))
    }]
  };

  Plotly.newPlot(volumeId, [initialVolume], layout3D)
    .then(() => Plotly.addFrames(volumeId, animationFrames));
}