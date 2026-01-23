async function runSimulation() {
    // 1. UI: Show Loader & Reset
    const loader = document.getElementById("loader");
    loader.classList.remove("hidden");
    
    try {
        const mode = document.getElementById("mode").value;
        const body = mode === "synthetic"
            ? { mode: "synthetic" }
            : { mode: "user", sequence: typeof userSequence !== 'undefined' ? userSequence : [] };

        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        if (!res.ok) throw new Error("Simulation failed to start");

        const data = await res.json();

        // 2. Render Visuals
        visualizeLifecycle(data.frames, data.max_temp_log, data.cluster_count_log);
        showRisk(data);

    } catch (error) {
        alert("Error running simulation: " + error.message);
        console.error(error);
    } finally {
        // 3. UI: Hide Loader
        loader.classList.add("hidden");
    }
}

function showRisk({ risk_score, risk_level }) {
    const riskBox = document.getElementById("riskBox");
    const content = document.getElementById("riskContent");
    
    riskBox.classList.remove("hidden");
    
    let color = "#ef4444"; // Default Red
    let icon = "⚠️";
    let glow = "0 0 15px rgba(239, 68, 68, 0.5)"; // Red Glow

    if (risk_level === "SAFE" || risk_level === "VERY LOW") {
        color = "#3b82f6"; // Blue
        icon = "🛡️";
        glow = "0 0 15px rgba(59, 130, 246, 0.5)";
    } else if (risk_level === "LOW") {
        color = "#10b981"; // Green
        icon = "✅";
        glow = "0 0 15px rgba(16, 185, 129, 0.5)";
    } else if (risk_level === "MEDIUM") {
        color = "#f59e0b"; // Orange
        icon = "⚠️";
        glow = "0 0 15px rgba(245, 158, 11, 0.5)";
    }

    riskBox.style.borderColor = color;
    riskBox.style.boxShadow = glow;
    
    content.innerHTML = `
        <span style="color:${color}; font-size: 1.8em; font-weight: bold;">
            ${icon} ${risk_level}
        </span>
        <div style="font-size: 0.9em; margin-top:5px; color: #cbd5e1;">
            Risk Score: ${risk_score.toFixed(4)}
        </div>
    `;
}

function visualizeLifecycle(frames, max_log, count_log) {
    const STEPS = frames.length;
    const SIZE = frames[0].length; 

    // Common styling for Dark Theme
    const darkLayoutConfig = {
        paper_bgcolor: 'rgba(0,0,0,0)', // Transparent
        plot_bgcolor: 'rgba(0,0,0,0)',  // Transparent
        font: { color: '#94a3b8', family: "Inter, sans-serif" },
        grid: { color: '#334155' }
    };

    /* ===============================
       2D DASHBOARD (Temperature + Clusters)
       =============================== */
    const chartId = "chartPlot";
    const timeSteps = [...Array(max_log.length).keys()];

    const tempTrace = {
        x: timeSteps, y: max_log,
        name: "Max Temp (°C)",
        type: "scatter", mode: "lines",
        line: { color: "#ef4444", width: 3 }, // Bright Red
        fill: 'tozeroy', // Add area fill for better look
        fillcolor: 'rgba(239, 68, 68, 0.1)'
    };

    const ignitionLine = {
        x: [0, timeSteps.length - 1], y: [100, 100],
        name: "Ignition Threshold",
        type: "scatter", mode: "lines",
        line: { color: "#f59e0b", dash: "dash", width: 2 },
        hoverinfo: "skip"
    };

    const clusterTrace = {
        x: timeSteps, y: count_log,
        name: "Active Clusters",
        type: "scatter", mode: "lines",
        line: { shape: "hv", color: "#3b82f6", width: 2 },
        yaxis: "y2"
    };

    const layout2D = {
        ...darkLayoutConfig,
        margin: { t: 30, l: 50, r: 50, b: 40 },
        legend: { orientation: 'h', y: -0.2 },
        xaxis: { 
            title: "Time Steps", 
            gridcolor: '#334155',
            zerolinecolor: '#334155'
        },
        yaxis: {
            title: "Temp (°C)",
            gridcolor: '#334155',
            zerolinecolor: '#334155'
        },
        yaxis2: {
            title: "Clusters (#)",
            overlaying: "y",
            side: "right",
            showgrid: false
        }
    };

    Plotly.newPlot(chartId, [tempTrace, ignitionLine, clusterTrace], layout2D, {responsive: true});

    /* ===============================
       3D ANIMATION
       =============================== */
    const volumeId = "volumePlot";
    const X = []; const Y = []; const Z = [];

    for (let x = 0; x < SIZE; x++) {
        for (let y = 0; y < SIZE; y++) {
            for (let z = 0; z < SIZE; z++) {
                X.push(x); Y.push(y); Z.push(z);
            }
        }
    }

    const customColors = [
        [0.0, "#3b82f6"], // Blue
        [0.33, "#10b981"], // Green
        [0.66, "#f59e0b"], // Orange
        [1.0, "#ef4444"]   // Red
    ];

    const initialVolume = {
        type: "volume",
        x: X, y: Y, z: Z,
        value: frames[0].flat(2),
        isomin: 25, isomax: 250,
        opacity: 0.15, // Slightly more transparent for style
        surface: { count: 20 },
        colorscale: customColors,
        colorbar: {
            title: 'Temp °C',
            len: 0.8,
            thickness: 10,
            tickfont: { color: '#94a3b8' }
        }
    };

    const skip = 1;
    const animationFrames = [];
    for (let t = 0; t < STEPS; t += skip) {
        animationFrames.push({
            name: String(t),
            data: [{ value: frames[t].flat(2), isomin: 25, isomax: 250 }]
        });
    }

    /* ===============================
       3D ANIMATION LAYOUT (Updated)
       =============================== */
    const layout3D = {
        ...darkLayoutConfig,
        title: {
            text: "Volumetric Heat Map (Silo View)",
            font: { color: '#f8fafc', size: 14 }
        },
        margin: { t: 30, l: 0, r: 0, b: 0 }, // Small top margin for title
        scene: {
            aspectmode: "cube",
            // 1. Restore the Axes (The Grid)
            xaxis: { 
                title: 'Width (X)',
                color: '#94a3b8',           // Text color
                showgrid: true, 
                gridcolor: '#334155',       // Dark grey grid lines
                zerolinecolor: '#334155', 
                showbackground: true,       // Show walls
                backgroundcolor: 'rgba(30, 41, 59, 0.3)' // Semi-transparent wall
            },
            yaxis: { 
                title: 'Depth (Y)',
                color: '#94a3b8',
                showgrid: true, 
                gridcolor: '#334155',
                zerolinecolor: '#334155',
                showbackground: true,
                backgroundcolor: 'rgba(30, 41, 59, 0.3)'
            },
            zaxis: { 
                title: 'Height (Z)',
                color: '#94a3b8',
                showgrid: true, 
                gridcolor: '#334155',
                zerolinecolor: '#334155',
                showbackground: true,
                backgroundcolor: 'rgba(30, 41, 59, 0.3)'
            },
            camera: {
                eye: { x: 1.6, y: 1.6, z: 1.6 } // Zoomed out slightly to see full grid
            }
        },
        updatemenus: [{
            type: "buttons",
            showactive: false,
            x: 0, y: 0, // Position button bottom-left
            bgcolor: 'rgba(30, 41, 59, 0.8)',
            bordercolor: '#475569',
            font: { color: '#f8fafc' },
            buttons: [{
                label: "▶ PLAY SIMULATION",
                method: "animate",
                args: [null, { frame: { duration: 100, redraw: true }, fromcurrent: true }]
            }]
        }],
        sliders: [{
            pad: { t: 10 },
            currentvalue: { 
                prefix: "Time Step: ", 
                font: { color: '#f8fafc' } 
            },
            font: { color: '#94a3b8' },
            len: 0.9,
            x: 0.05,
            steps: animationFrames.map(f => ({
                label: f.name,
                method: "animate",
                args: [[f.name], { mode: "immediate", frame: { duration: 0, redraw: true } }]
            }))
        }]
    };

    Plotly.newPlot(volumeId, [initialVolume], layout3D, {responsive: true})
        .then(() => Plotly.addFrames(volumeId, animationFrames));
}