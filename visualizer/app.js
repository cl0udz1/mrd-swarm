/**
 * app.js — High-Fidelity 3D WebGL Tactical Combat Visualizer for MRD-SWARM (V4)
 * 
 * Features:
 * - Live Aerospace Evaluation & Metrics Window (SE(3) RMSE, Graph Laplacian Fiedler value lambda_2, Track Ratio).
 * - Instant 100 Hz Black Box Telemetry CSV Export & Download.
 * - Draggable & Minimizable HUD Floating Windows.
 * - Clean Cinematic View Mode Toggle (Fades out all UI for unobstructed 3D environment visualization).
 * - Dual Three.js Rendering: Main 3D Tactical City + Dedicated Live Onboard FPV Camera Feed (PiP Canvas).
 * - Dynamic Laser Designation Beams & 3D Volumetric Smoke Aerosol System.
 * - Multi-Spectrum POV Sensor Modes (Daylight RGB, Green NVG Night-Vision, FLIR Thermal White-Hot IR).
 * - Web Audio API Sound Synthesizer (Quadrotor Doppler hum, target lock tone, radio static).
 * - Interactive Combat Action Deck with live WebSocket commands.
 */

const CONFIG = {
  wsUrl: "ws://127.0.0.1:8765",
  reconnectInterval: 2000,
  lerpFactor: 0.35,
};

const DRONE_COLORS = [
  { main: 0x38bdf8, hex: "#38bdf8", name: "HEAVY_SCOUT" },
  { main: 0xf43f5e, hex: "#f43f5e", name: "FAST_INTERCEPTOR" },
  { main: 0x22c55e, hex: "#22c55e", name: "THERMAL_SURVEYOR" },
  { main: 0xc084fc, hex: "#c084fc", name: "COMMS_RELAY" },
];

let scene, camera, renderer, controls;
let pipCamera, pipRenderer;
let activePipDroneId = 1;
let currentCamMode = "orbit";
let sensorSpectrumMode = "rgb";
let sfxEnabled = true;
let cinematicMode = false;

const droneEntities = {};
const targetEntities = {};
const buildingMeshes = [];
let radarDishMesh = null;
let rfMeshLinesGroup;
let laserLinesGroup;
let smokeParticlesGroup;
const activeSmokeParticles = [];

let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 60;
let ws = null;
let lastTelemetry = null;
let telemetryHistory = [];

// ==============================================================================
// Web Audio API Sound Engine
// ==============================================================================
let audioCtx = null;
let motorOsc1 = null, motorOsc2 = null, motorGain = null;

function initAudio() {
  if (audioCtx) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContext();

    motorOsc1 = audioCtx.createOscillator();
    motorOsc2 = audioCtx.createOscillator();
    motorGain = audioCtx.createGain();

    motorOsc1.type = "sawtooth";
    motorOsc1.frequency.setValueAtTime(110, audioCtx.currentTime);

    motorOsc2.type = "sine";
    motorOsc2.frequency.setValueAtTime(220, audioCtx.currentTime);

    motorGain.gain.setValueAtTime(0.03, audioCtx.currentTime);

    motorOsc1.connect(motorGain);
    motorOsc2.connect(motorGain);
    motorGain.connect(audioCtx.destination);

    motorOsc1.start();
    motorOsc2.start();
  } catch (e) {
    console.warn("Web Audio not supported or blocked:", e);
  }
}

function playTargetLockTone() {
  if (!sfxEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.setValueAtTime(1760, audioCtx.currentTime + 0.08);

    gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.22);

    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
  } catch (e) {}
}

function playRadioChirp() {
  if (!sfxEnabled || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(2400, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.001, audioCtx.currentTime + 0.06);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.07);
  } catch (e) {}
}

// ==============================================================================
// Main Initialization
// ==============================================================================
window.addEventListener("DOMContentLoaded", () => {
  initMainThree();
  initPipRenderer();
  initHUD();
  initDraggableWindows();
  connectWebSocket();
  animate();

  document.body.addEventListener("click", () => {
    if (sfxEnabled && !audioCtx) initAudio();
  }, { once: true });
});

function initMainThree() {
  const container = document.getElementById("canvas-container");
  const width = window.innerWidth;
  const height = window.innerHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x04060d);
  scene.fog = new THREE.FogExp2(0x04060d, 0.012);

  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 500);
  camera.position.set(38, 32, 42);

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 - 0.02;
  controls.minDistance = 3;
  controls.maxDistance = 180;
  controls.target.set(0, 3, 0);

  const ambientLight = new THREE.AmbientLight(0x1e293b, 1.8);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xe0f2fe, 2.2);
  dirLight.position.set(40, 60, 30);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  const d = 35;
  dirLight.shadow.camera.left = -d;
  dirLight.shadow.camera.right = d;
  dirLight.shadow.camera.top = d;
  dirLight.shadow.camera.bottom = -d;
  scene.add(dirLight);

  const gridHelper = new THREE.GridHelper(60, 40, 0x0284c7, 0x1e293b);
  gridHelper.position.y = 0.01;
  scene.add(gridHelper);

  const floorMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ color: 0x070b16, roughness: 0.85, metalness: 0.2 })
  );
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.receiveShadow = true;
  scene.add(floorMesh);

  rfMeshLinesGroup = new THREE.Group();
  laserLinesGroup = new THREE.Group();
  smokeParticlesGroup = new THREE.Group();
  scene.add(rfMeshLinesGroup);
  scene.add(laserLinesGroup);
  scene.add(smokeParticlesGroup);

  createDroneEntities();
  createTargetEntities();

  window.addEventListener("resize", onWindowResize);
}

function initPipRenderer() {
  const pipCanvas = document.getElementById("pip-canvas");
  if (!pipCanvas) return;

  pipRenderer = new THREE.WebGLRenderer({ canvas: pipCanvas, antialias: true });
  pipRenderer.setSize(pipCanvas.clientWidth, pipCanvas.clientHeight);
  pipRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  pipCamera = new THREE.PerspectiveCamera(75, pipCanvas.clientWidth / pipCanvas.clientHeight, 0.05, 300);
}

// ==============================================================================
// 3D Asset Creation
// ==============================================================================
function buildUrbanBuildings(buildings) {
  buildingMeshes.forEach(b => scene.remove(b));
  buildingMeshes.length = 0;

  buildings.forEach(obs => {
    const [ox, oy, oz] = obs.pos;
    const [hw, hl, hh] = obs.size;
    const height = obs.height || hh * 2;
    const width = hw * 2;
    const length = hl * 2;

    const geo = new THREE.BoxGeometry(width, height, length);
    const mat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.6, metalness: 0.4 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(ox, height / 2, -oy);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    const edges = new THREE.EdgesGeometry(geo);
    const edgeColor = obs.color ? parseInt(obs.color.replace("#", "0x")) : 0x38bdf8;
    mesh.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: edgeColor, linewidth: 1.5, transparent: true, opacity: 0.7 })));

    const roofRing = new THREE.Mesh(
      new THREE.RingGeometry(0.8, 1.2, 16),
      new THREE.MeshBasicMaterial({ color: edgeColor, side: THREE.DoubleSide })
    );
    roofRing.rotation.x = Math.PI / 2;
    roofRing.position.y = height / 2 + 0.05;
    mesh.add(roofRing);

    if (obs.name.includes("Radar")) {
      const dishGroup = new THREE.Group();
      dishGroup.position.set(0, height / 2 + 0.8, 0);
      const dish = new THREE.Mesh(
        new THREE.CylinderGeometry(1.2, 0.2, 0.3, 16),
        new THREE.MeshStandardMaterial({ color: 0xa855f7, metalness: 0.8 })
      );
      dish.rotation.z = Math.PI / 4;
      dishGroup.add(dish);
      mesh.add(dishGroup);
      radarDishMesh = dishGroup;
    }

    scene.add(mesh);
    buildingMeshes.push(mesh);
  });
}

function createDroneEntities() {
  for (let i = 0; i < 4; i++) {
    const group = new THREE.Group();
    const colorSpec = DRONE_COLORS[i];

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.3 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.08, 0.35), bodyMat);
    body.castShadow = true;
    group.add(body);

    const armMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.2 });
    const arm1 = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.7), armMat);
    arm1.rotation.z = Math.PI / 2;
    arm1.rotation.y = Math.PI / 4;
    group.add(arm1);

    const arm2 = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.7), armMat);
    arm2.rotation.z = Math.PI / 2;
    arm2.rotation.y = -Math.PI / 4;
    group.add(arm2);

    const rotors = [];
    const rotorOffsets = [
      [0.25, 0.06, 0.25], [-0.25, 0.06, 0.25],
      [0.25, 0.06, -0.25], [-0.25, 0.06, -0.25],
    ];
    rotorOffsets.forEach(([rx, ry, rz]) => {
      const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.06), bodyMat);
      motor.position.set(rx, ry, rz);
      group.add(motor);

      const prop = new THREE.Mesh(
        new THREE.BoxGeometry(0.26, 0.005, 0.035),
        new THREE.MeshBasicMaterial({ color: colorSpec.main, transparent: true, opacity: 0.85 })
      );
      prop.position.set(rx, ry + 0.04, rz);
      group.add(prop);
      rotors.push(prop);
    });

    const beacon = new THREE.PointLight(colorSpec.main, 2.2, 6);
    beacon.position.set(0, 0.08, -0.2);
    group.add(beacon);

    const spotlight = new THREE.SpotLight(0xffffff, 4.0, 25, Math.PI / 6, 0.4, 1.2);
    spotlight.position.set(0, 0, 0.2);
    spotlight.target.position.set(0, -10, 4);
    group.add(spotlight);
    group.add(spotlight.target);

    const fovCone = new THREE.Mesh(
      new THREE.ConeGeometry(0.9, 3.2, 4, 1, true),
      new THREE.MeshBasicMaterial({ color: colorSpec.main, wireframe: true, transparent: true, opacity: 0.25 })
    );
    fovCone.rotation.x = -Math.PI / 2;
    fovCone.position.set(0, 0, 1.6);
    group.add(fovCone);

    scene.add(group);

    const trailMaxPoints = 120;
    const trailPositions = new Float32Array(trailMaxPoints * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: colorSpec.main, transparent: true, opacity: 0.6 }));
    scene.add(trailLine);

    droneEntities[i] = {
      group,
      rotors,
      spotlight,
      trailLine,
      trailHistory: [],
      targetPos: new THREE.Vector3(0, 1, 0),
      targetQuat: new THREE.Quaternion(),
      currentPos: new THREE.Vector3(0, 1, 0),
      currentQuat: new THREE.Quaternion(),
    };
  }
}

function createTargetEntities() {
  for (let i = 0; i < 3; i++) {
    const group = new THREE.Group();

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x991b1b, metalness: 0.7, roughness: 0.4 });
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.4, 1.2), bodyMat);
    body.position.y = 0.2;
    body.castShadow = true;
    group.add(body);

    const diamond = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.45),
      new THREE.MeshBasicMaterial({ color: 0xf43f5e, wireframe: true })
    );
    diamond.position.y = 1.1;
    group.add(diamond);

    scene.add(group);

    targetEntities[i] = {
      group,
      diamond,
      targetPos: new THREE.Vector3(0, 0.3, 0),
      currentPos: new THREE.Vector3(0, 0.3, 0),
    };
  }
}

// ==============================================================================
// Draggable Windows & UI Controls
// ==============================================================================
function initDraggableWindows() {
  const windows = document.querySelectorAll(".hud-window");
  windows.forEach(win => {
    const header = win.querySelector(".win-header");
    if (header) {
      makeDraggable(win, header);
    }
  });

  document.querySelectorAll(".min-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const targetId = btn.dataset.target;
      const win = document.getElementById(targetId);
      if (win) {
        win.classList.toggle("minimized");
        btn.innerText = win.classList.contains("minimized") ? "+" : "_";
      }
    });
  });

  const hudToggleBtn = document.getElementById("hud-toggle-btn");
  if (hudToggleBtn) {
    hudToggleBtn.addEventListener("click", () => {
      cinematicMode = !cinematicMode;
      const hudRoot = document.getElementById("hud-root");
      if (cinematicMode) {
        hudRoot.classList.add("cinematic-mode");
        hudToggleBtn.innerText = "👁️ SHOW HUD";
      } else {
        hudRoot.classList.remove("cinematic-mode");
        hudToggleBtn.innerText = "👁️ HIDE HUD";
      }
    });
  }

  // Export CSV Button
  const exportBtn = document.getElementById("export-csv-btn");
  if (exportBtn) {
    exportBtn.addEventListener("click", exportBlackboxCSV);
  }

  const resetBtn = document.getElementById("reset-layout-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      const fleet = document.getElementById("win-fleet");
      const actions = document.getElementById("win-actions");
      const fpv = document.getElementById("win-fpv");
      const targets = document.getElementById("win-targets");
      const evalWin = document.getElementById("win-eval");

      if (fleet) { fleet.style.top = "80px"; fleet.style.left = "16px"; fleet.classList.remove("minimized"); }
      if (actions) { actions.style.top = "80px"; actions.style.left = "350px"; actions.classList.remove("minimized"); }
      if (fpv) { fpv.style.bottom = "85px"; fpv.style.left = "350px"; fpv.style.top = "auto"; fpv.classList.remove("minimized"); }
      if (targets) { targets.style.top = "80px"; targets.style.right = "16px"; targets.style.left = "auto"; targets.classList.remove("minimized"); }
      if (evalWin) { evalWin.style.bottom = "85px"; evalWin.style.right = "16px"; evalWin.style.top = "auto"; evalWin.style.left = "auto"; evalWin.classList.remove("minimized"); }
      addTerminalLog("[HUD] Window layout reset to default", "info");
    });
  }
}

function makeDraggable(element, handle) {
  let posX = 0, posY = 0, mouseX = 0, mouseY = 0;

  handle.onmousedown = dragMouseDown;

  function dragMouseDown(e) {
    e.preventDefault();
    document.querySelectorAll(".hud-window").forEach(w => w.style.zIndex = "20");
    element.style.zIndex = "30";

    mouseX = e.clientX;
    mouseY = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e.preventDefault();
    posX = mouseX - e.clientX;
    posY = mouseY - e.clientY;
    mouseX = e.clientX;
    mouseY = e.clientY;

    const newTop = element.offsetTop - posY;
    const newLeft = element.offsetLeft - posX;

    element.style.top = Math.max(10, Math.min(window.innerHeight - 60, newTop)) + "px";
    element.style.left = Math.max(10, Math.min(window.innerWidth - 80, newLeft)) + "px";
    element.style.bottom = "auto";
    element.style.right = "auto";
  }

  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

function exportBlackboxCSV() {
  if (telemetryHistory.length === 0) {
    addTerminalLog("[EXPORT] Telemetry buffer initializing...", "alert");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "sim_time,uncertainty_pct,num_links,d0_x,d0_y,d0_z,d0_spd,d1_x,d1_y,d1_z,d1_spd,d2_x,d2_y,d2_z,d2_spd,d3_x,d3_y,d3_z,d3_spd\n";

  telemetryHistory.forEach(r => {
    const row = [
      r.time, r.uncertainty_pct, r.rf_mesh.total_links,
      r.drones[0].pos[0], r.drones[0].pos[1], r.drones[0].pos[2], r.drones[0].speed,
      r.drones[1].pos[0], r.drones[1].pos[1], r.drones[1].pos[2], r.drones[1].speed,
      r.drones[2].pos[0], r.drones[2].pos[1], r.drones[2].pos[2], r.drones[2].speed,
      r.drones[3].pos[0], r.drones[3].pos[1], r.drones[3].pos[2], r.drones[3].speed,
    ].join(",");
    csvContent += row + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `mrd_swarm_flight_log_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  addTerminalLog(`[EXPORT] Downloaded ${telemetryHistory.length} frames of 100Hz Black Box flight data!`, "success");
}

// ==============================================================================
// WebSocket Telemetry Client & Action Dispatcher
// ==============================================================================
function connectWebSocket() {
  const indicator = document.getElementById("connection-indicator");
  ws = new WebSocket(CONFIG.wsUrl);

  ws.onopen = () => {
    indicator.className = "pulse-indicator live";
    addTerminalLog("[WS] Tactical Combat Bridge Connected (ws://127.0.0.1:8765)", "success");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "WORLD_METADATA") {
        if (data.buildings) buildUrbanBuildings(data.buildings);
      } else if (data.type === "TELEMETRY_UPDATE") {
        handleTelemetryUpdate(data);
      }
    } catch (e) {
      console.error("Telemetry parse error:", e);
    }
  };

  ws.onclose = () => {
    indicator.className = "pulse-indicator";
    addTerminalLog("[WS] Link interrupted. Reconnecting...", "alert");
    setTimeout(connectWebSocket, CONFIG.reconnectInterval);
  };
}

function sendAction(action, payload = {}) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    const msg = JSON.stringify({ action, ...payload });
    ws.send(msg);
    playRadioChirp();
    addTerminalLog(`[CMD] Dispatched -> ${action}`, "cmd");
  }
}

function handleTelemetryUpdate(data) {
  lastTelemetry = data;
  telemetryHistory.push(data);
  if (telemetryHistory.length > 3600) telemetryHistory.shift();

  if (data.drones) {
    Object.values(data.drones).forEach(d => {
      const entity = droneEntities[d.id];
      if (entity) {
        entity.targetPos.set(d.pos[0], d.pos[2], -d.pos[1]);
        entity.targetQuat.set(d.quat[1], d.quat[3], -d.quat[2], d.quat[0]);
      }
    });
  }

  if (data.targets) {
    Object.values(data.targets).forEach(t => {
      const entity = targetEntities[t.id];
      if (entity) {
        entity.targetPos.set(t.pos[0], t.pos[2], -t.pos[1]);
      }
    });
  }

  if (data.rf_mesh && data.rf_mesh.active_links) {
    updateRFMeshGraphics(data.rf_mesh.active_links);
  }

  if (data.combat_effects) {
    updateCombatEffects(data.combat_effects);
  }

  updateHUD(data);
}

function updateRFMeshGraphics(links) {
  while (rfMeshLinesGroup.children.length > 0) {
    rfMeshLinesGroup.remove(rfMeshLinesGroup.children[0]);
  }

  links.forEach(([a, b]) => {
    if (droneEntities[a] && droneEntities[b]) {
      const p1 = droneEntities[a].currentPos;
      const p2 = droneEntities[b].currentPos;
      const geo = new THREE.BufferGeometry().setFromPoints([p1.clone(), p2.clone()]);
      const mat = new THREE.LineBasicMaterial({
        color: (a === 3 || b === 3) ? 0xc084fc : 0x00f0ff,
        transparent: true,
        opacity: 0.8,
        linewidth: 2,
      });
      rfMeshLinesGroup.add(new THREE.Line(geo, mat));
    }
  });
}

function updateCombatEffects(effects) {
  while (laserLinesGroup.children.length > 0) {
    laserLinesGroup.remove(laserLinesGroup.children[0]);
  }

  if (effects.active_lasers) {
    effects.active_lasers.forEach(l => {
      const p1 = new THREE.Vector3(l.origin[0], l.origin[2], -l.origin[1]);
      const p2 = new THREE.Vector3(l.target_pos[0], l.target_pos[2] + 0.3, -l.target_pos[1]);
      const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
      const colorVal = parseInt(l.color.replace("#", "0x")) || 0xef4444;
      const mat = new THREE.LineBasicMaterial({
        color: colorVal,
        linewidth: 3,
        transparent: true,
        opacity: 0.9,
      });
      laserLinesGroup.add(new THREE.Line(geo, mat));
    });
  }

  if (effects.active_smokes && effects.active_smokes.length > 0) {
    effects.active_smokes.forEach(smk => {
      if (Math.random() < 0.35) {
        const puffGeo = new THREE.DodecahedronGeometry(1.2 + Math.random() * 1.5);
        const puffMat = new THREE.MeshStandardMaterial({
          color: 0x94a3b8,
          transparent: true,
          opacity: 0.65,
          roughness: 0.9,
        });
        const puff = new THREE.Mesh(puffGeo, puffMat);
        puff.position.set(
          smk.pos[0] + (Math.random() - 0.5) * 3.0,
          0.8 + Math.random() * 1.8,
          -smk.pos[1] + (Math.random() - 0.5) * 3.0
        );
        puff.userData = { life: 1.0, decay: 0.015 + Math.random() * 0.01 };
        smokeParticlesGroup.add(puff);
        activeSmokeParticles.push(puff);
      }
    });
  }
}

// ==============================================================================
// Animation & Render Loop
// ==============================================================================
function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const delta = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  frameCount++;
  if (frameCount % 30 === 0) {
    fps = Math.round(1 / delta);
  }

  if (radarDishMesh) {
    radarDishMesh.rotation.y += 0.04;
  }

  for (let i = activeSmokeParticles.length - 1; i >= 0; i--) {
    const p = activeSmokeParticles[i];
    p.userData.life -= p.userData.decay;
    p.scale.multiplyScalar(1.015);
    p.position.y += 0.02;
    p.material.opacity = p.userData.life * 0.6;
    if (p.userData.life <= 0.05) {
      smokeParticlesGroup.remove(p);
      activeSmokeParticles.splice(i, 1);
    }
  }

  for (let i = 0; i < 4; i++) {
    const d = droneEntities[i];
    if (d) {
      d.currentPos.lerp(d.targetPos, CONFIG.lerpFactor);
      d.currentQuat.slerp(d.targetQuat, CONFIG.lerpFactor);
      d.group.position.copy(d.currentPos);
      d.group.quaternion.copy(d.currentQuat);

      d.rotors.forEach((r, idx) => {
        r.rotation.y += (idx % 2 === 0 ? 0.85 : -0.85);
      });

      if (d.trailHistory.length === 0 || d.currentPos.distanceTo(d.trailHistory[d.trailHistory.length - 1]) > 0.15) {
        d.trailHistory.push(d.currentPos.clone());
        if (d.trailHistory.length > 120) d.trailHistory.shift();

        const posAttr = d.trailLine.geometry.attributes.position;
        for (let pIdx = 0; pIdx < d.trailHistory.length; pIdx++) {
          posAttr.setXYZ(pIdx, d.trailHistory[pIdx].x, d.trailHistory[pIdx].y, d.trailHistory[pIdx].z);
        }
        d.trailLine.geometry.setDrawRange(0, d.trailHistory.length);
        posAttr.needsUpdate = true;
      }
    }
  }

  for (let i = 0; i < 3; i++) {
    const t = targetEntities[i];
    if (t) {
      t.currentPos.lerp(t.targetPos, CONFIG.lerpFactor);
      t.group.position.copy(t.currentPos);
      t.diamond.rotation.y += 0.04;
    }
  }

  updateMainCameraRig();
  renderer.render(scene, camera);

  renderOnboardPOV();
}

function updateMainCameraRig() {
  if (currentCamMode === "orbit") {
    controls.enabled = true;
    controls.update();
  } else if (currentCamMode === "chase1" && droneEntities[1]) {
    controls.enabled = false;
    const target = droneEntities[1].currentPos;
    const offset = new THREE.Vector3(0, 2.0, -4.2).applyQuaternion(droneEntities[1].currentQuat);
    camera.position.lerp(target.clone().add(offset), 0.22);
    camera.lookAt(target.clone().add(new THREE.Vector3(0, 0.4, 0)));
  } else if (currentCamMode === "chase2" && droneEntities[2]) {
    controls.enabled = false;
    const target = droneEntities[2].currentPos;
    const offset = new THREE.Vector3(0, 2.0, -4.2).applyQuaternion(droneEntities[2].currentQuat);
    camera.position.lerp(target.clone().add(offset), 0.22);
    camera.lookAt(target.clone().add(new THREE.Vector3(0, 0.4, 0)));
  } else if (currentCamMode === "fpv1" && droneEntities[1]) {
    controls.enabled = false;
    const nose = droneEntities[1].currentPos.clone().add(new THREE.Vector3(0, 0.05, 0.22).applyQuaternion(droneEntities[1].currentQuat));
    camera.position.copy(nose);
    const forward = new THREE.Vector3(0, -0.05, 6.0).applyQuaternion(droneEntities[1].currentQuat);
    camera.lookAt(nose.clone().add(forward));
  } else if (currentCamMode === "tactical") {
    controls.enabled = false;
    camera.position.lerp(new THREE.Vector3(0, 65, 0.01), 0.1);
    camera.lookAt(0, 0, 0);
  }
}

function renderOnboardPOV() {
  if (!pipRenderer || !pipCamera || !droneEntities[activePipDroneId]) return;

  const drone = droneEntities[activePipDroneId];
  const nosePos = drone.currentPos.clone().add(new THREE.Vector3(0, 0.05, 0.22).applyQuaternion(drone.currentQuat));
  pipCamera.position.copy(nosePos);

  const forwardTarget = nosePos.clone().add(new THREE.Vector3(0, -0.05, 8.0).applyQuaternion(drone.currentQuat));
  pipCamera.lookAt(forwardTarget);

  pipRenderer.render(scene, pipCamera);

  if (lastTelemetry && lastTelemetry.drones && lastTelemetry.drones[activePipDroneId]) {
    const d = lastTelemetry.drones[activePipDroneId];
    const tl = document.getElementById("pip-telemetry-tl");
    const br = document.getElementById("pip-telemetry-br");
    const status = document.getElementById("pip-reticle-status");
    const laserBadge = document.getElementById("laser-badge");

    if (tl) tl.innerText = `ALT: ${d.pos[2].toFixed(1)}m | SPD: ${d.speed.toFixed(1)}m/s`;
    
    const visibleCount = Object.keys(d.visible_targets || {}).length;
    if (status) {
      if (visibleCount > 0) {
        status.innerText = `🎯 TARGET LOCKED (${visibleCount} IN FOV)`;
        status.style.color = "#f43f5e";
        if (frameCount % 60 === 0) playTargetLockTone();
      } else {
        status.innerText = `OPTICAL SCANNING`;
        status.style.color = "#22c55e";
      }
    }

    if (laserBadge) {
      laserBadge.style.display = d.laser_active ? "block" : "none";
    }

    if (br) br.innerText = `ROLE: ${d.role} | BAT: ${d.battery.toFixed(1)}%`;
  }
}

// ==============================================================================
// HUD Updates & UI Interactions
// ==============================================================================
function initHUD() {
  document.querySelectorAll(".cam-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".cam-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentCamMode = btn.dataset.cam;
    });
  });

  document.querySelectorAll(".pip-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pip-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activePipDroneId = parseInt(btn.dataset.drone);
      const label = document.getElementById("pip-cam-label");
      if (label) label.innerText = `LIVE ONBOARD POV // D${activePipDroneId} [${DRONE_COLORS[activePipDroneId].name}]`;
    });
  });

  document.querySelectorAll(".spec-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".spec-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      sensorSpectrumMode = btn.dataset.spec;
      const wrapper = document.getElementById("fpv-wrapper");
      wrapper.className = `fpv-viewport-wrapper mode-${sensorSpectrumMode}`;
      addTerminalLog(`[SENSOR] POV Mode -> ${sensorSpectrumMode.toUpperCase()}`, "info");
    });
  });

  document.getElementById("btn-jam-trigger").addEventListener("click", () => {
    sendAction("TRIGGER_JAMMING");
  });
  document.getElementById("btn-smoke-trigger").addEventListener("click", () => {
    sendAction("TRIGGER_SMOKE", { target_id: 0 });
  });
  document.getElementById("btn-pincer-trigger").addEventListener("click", () => {
    sendAction("TRIGGER_PINCER");
  });
  document.getElementById("btn-rtb-trigger").addEventListener("click", () => {
    sendAction("TRIGGER_RTB", { drone_id: 1 });
  });

  const sfxBtn = document.getElementById("sfx-toggle");
  sfxBtn.addEventListener("click", () => {
    sfxEnabled = !sfxEnabled;
    sfxBtn.innerText = sfxEnabled ? "🔊 SFX: ON" : "🔇 SFX: OFF";
    if (motorGain && audioCtx) {
      motorGain.gain.setValueAtTime(sfxEnabled ? 0.03 : 0.0, audioCtx.currentTime);
    }
  });

  initAICommanderControls();
}

function initAICommanderControls() {
  const forceVisionBtn = document.getElementById("btn-force-vision");
  if (forceVisionBtn) {
    forceVisionBtn.addEventListener("click", () => {
      sendAction("TRIGGER_VISION_SCAN", { drone_id: activePipDroneId });
      addTerminalLog(`[AI VISION] Force triggered scan on D${activePipDroneId} camera POV`, "cmd");
    });
  }

  const opForm = document.getElementById("operator-cmd-form");
  const opInput = document.getElementById("operator-cmd-input");
  if (opForm && opInput) {
    opForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const val = opInput.value.trim();
      if (val) {
        sendAction("OPERATOR_COMMAND", { command: val });
        addTerminalLog(`[OPERATOR] Uplink transmitted -> "${val}"`, "success");
        opInput.value = "";
      }
    });
  }

  document.querySelectorAll(".chip-btn").forEach(chip => {
    chip.addEventListener("click", () => {
      const cmd = chip.dataset.cmd;
      if (cmd) {
        sendAction("OPERATOR_COMMAND", { command: cmd });
        addTerminalLog(`[OPERATOR CHIP] -> "${cmd}"`, "success");
      }
    });
  });

  // Tactical Doctrine Selector Buttons
  document.querySelectorAll(".doctrine-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".doctrine-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const doc = btn.dataset.doctrine;
      if (doc) {
        sendAction("SET_DOCTRINE", { doctrine: doc });
        addTerminalLog(`[TACTICAL DOCTRINE] Switched to -> ${doc}`, "warning");
      }
    });
  });
}

function initDraggableWindows() {
  // Minimize buttons
  document.querySelectorAll(".min-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const targetId = btn.dataset.target;
      const win = document.getElementById(targetId);
      if (win) {
        const body = win.querySelector(".win-body");
        if (body) {
          const isCollapsed = body.style.display === "none";
          body.style.display = isCollapsed ? "block" : "none";
          btn.innerText = isCollapsed ? "_" : "+";
        }
      }
    });
  });

  // Drag handles
  document.querySelectorAll(".hud-window").forEach(win => {
    const handle = win.querySelector(".drag-handle") || win;
    let isDragging = false;
    let startX = 0, startY = 0;
    let initialLeft = 0, initialTop = 0;

    handle.addEventListener("mousedown", (e) => {
      if (e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = win.getBoundingClientRect();
      initialLeft = rect.left;
      initialTop = rect.top;
      win.style.left = initialLeft + "px";
      win.style.top = initialTop + "px";
      win.style.bottom = "auto";
      win.style.right = "auto";
      win.style.zIndex = "100";
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      win.style.left = Math.max(0, Math.min(window.innerWidth - 100, initialLeft + dx)) + "px";
      win.style.top = Math.max(0, Math.min(window.innerHeight - 60, initialTop + dy)) + "px";
    });

    window.addEventListener("mouseup", () => {
      if (isDragging) {
        isDragging = false;
        win.style.zIndex = "20";
      }
    });
  });
}

function updateHUD(data) {
  const mins = Math.floor(data.time / 60).toString().padStart(2, "0");
  const secs = (data.time % 60).toFixed(2).padStart(5, "0");
  document.getElementById("val-time").innerText = `${mins}:${secs}`;
  document.getElementById("val-uncertainty").innerText = `${data.uncertainty_pct.toFixed(1)}%`;
  document.getElementById("val-links").innerText = data.rf_mesh.total_links;

  const ewBadge = document.getElementById("val-ew-status");
  if (ewBadge && data.combat_effects && data.combat_effects.ew_jamming) {
    const isJammed = data.combat_effects.ew_jamming.active;
    ewBadge.innerText = isJammed ? "🚨 SECTOR 2 JAMMED" : "INACTIVE";
    ewBadge.style.color = isJammed ? "#f59e0b" : "#22c55e";
  }

  // Live Aerospace Evaluation Metrics
  if (data.evaluation_metrics) {
    const em = data.evaluation_metrics;
    const rPos = document.getElementById("eval-rmse-pos");
    const rVel = document.getElementById("eval-rmse-vel");
    const l2 = document.getElementById("eval-lambda-2");
    const tRatio = document.getElementById("eval-track-ratio");

    if (rPos) rPos.innerText = `${em.rmse_pos.toFixed(2)} m`;
    if (rVel) rVel.innerText = `${em.rmse_vel.toFixed(2)} m/s`;
    if (l2) l2.innerText = em.mean_lambda_2.toFixed(3);
    if (tRatio) tRatio.innerText = `${em.track_ratio.toFixed(1)}%`;
  }

  // Drone Fleet Cards
  const fleetList = document.getElementById("drone-cards-list");
  let fleetHtml = "";
  Object.values(data.drones).forEach(d => {
    fleetHtml += `
      <div class="drone-card">
        <div class="card-top">
          <span class="drone-title">
            <span class="drone-led led-${d.id}"></span>
            D${d.id} [${d.class}]
          </span>
          <span class="role-pill">${d.role}</span>
        </div>
        <div class="card-stats">
          <span>ALT: <b>${d.pos[2].toFixed(1)}m</b></span>
          <span>SPD: <b>${d.speed.toFixed(1)}m/s</b></span>
          <span>POS: <b>[${d.pos[0].toFixed(1)}, ${d.pos[1].toFixed(1)}]</b></span>
          <span>BAT: <b>${d.battery.toFixed(1)}%</b></span>
        </div>
        <div class="battery-bar-wrap">
          <div class="battery-bar-fill" style="width: ${d.battery}%;"></div>
        </div>
        <div class="tool-active-tag">CMD: ${d.active_tool}</div>
      </div>
    `;
  });
  fleetList.innerHTML = fleetHtml;

  // Targets Cards
  const targetsList = document.getElementById("targets-list");
  let targetsHtml = "";
  Object.values(data.targets).forEach(t => {
    targetsHtml += `
      <div class="target-card ${t.state.includes('EVASION') ? 'evading' : ''}">
        <span>🎯 HVT-${t.id} [${t.name.slice(0, 10)}]</span>
        <span style="color: ${t.is_spotted ? '#f43f5e' : '#94a3b8'}; font-weight: 700;">
          ${t.smoke_active ? '💨 SMOKE ACTIVE' : (t.is_spotted ? '⚡ LOCKED' : t.state)}
        </span>
      </div>
    `;
  });
  targetsList.innerHTML = targetsHtml;

  // DeepSeek AI Swarm Commander HUD Updates
  if (data.ai_commander) {
    const aic = data.ai_commander;
    const msgEl = document.getElementById("ai-radio-msg");
    const postureEl = document.getElementById("ai-posture-badge");
    const modelEl = document.getElementById("ai-model-badge");
    const latEl = document.getElementById("ai-latency-badge");
    const cotLogs = document.getElementById("ai-cot-logs");

    if (msgEl && aic.tactical_radio_broadcast) {
      msgEl.innerText = `"${aic.tactical_radio_broadcast}"`;
    }
    if (postureEl && aic.strategic_posture) {
      postureEl.innerText = aic.strategic_posture;
    }
    if (modelEl && aic.model) {
      modelEl.innerText = aic.model;
    }
    if (latEl) {
      latEl.innerText = aic.latency_s > 0 ? `⚡ ${aic.latency_s}s` : "⚡ LIVE";
    }
    if (cotLogs && aic.reasoning_chain) {
      cotLogs.innerHTML = `<div class="log-row info" style="color:#c084fc; font-family:var(--font-mono); font-size:0.62rem;">[CoT] ${aic.reasoning_chain}</div>`;
    }
  }

  // Active Tactical Doctrine State Synchronization
  if (data.tactical_doctrine) {
    document.querySelectorAll(".doctrine-btn").forEach(btn => {
      if (btn.dataset.doctrine === data.tactical_doctrine) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  // DeepSeek Vision Reconnaissance HUD Updates
  if (data.vision_recon) {
    const vr = data.vision_recon;
    const thumbImg = document.getElementById("vision-thumb-img");
    const thumbPlaceholder = document.getElementById("vision-thumb-placeholder");
    const threatBadge = document.getElementById("vision-threat-badge");
    const targetType = document.getElementById("vision-target-type");
    const smokeStatus = document.getElementById("vision-smoke-status");
    const desc = document.getElementById("vision-desc");
    const recText = document.getElementById("vision-rec-text");

    if (thumbImg && thumbPlaceholder && vr.thumbnail_data_url) {
      thumbImg.src = vr.thumbnail_data_url;
      thumbImg.style.display = "block";
      thumbPlaceholder.style.display = "none";
    }
    if (threatBadge && vr.threat_level) {
      threatBadge.innerText = vr.threat_level;
      threatBadge.className = `threat-pill threat-${vr.threat_level.toLowerCase()}`;
    }
    if (targetType) targetType.innerText = vr.target_type;
    if (smokeStatus) {
      smokeStatus.innerText = vr.smoke_detected ? "AEROSOL 💨" : "NONE";
      smokeStatus.style.color = vr.smoke_detected ? "#f59e0b" : "#22c55e";
    }
    if (desc) desc.innerText = vr.visual_description;
    if (recText) recText.innerText = vr.tactical_recommendation;
  }
}

function addTerminalLog(msg, type = "info") {
  const terminal = document.getElementById("terminal-logs-body");
  if (!terminal) return;
  const row = document.createElement("div");
  row.className = `log-row ${type}`;
  row.innerText = msg;
  terminal.appendChild(row);
  terminal.scrollTop = terminal.scrollHeight;
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);

  const pipCanvas = document.getElementById("pip-canvas");
  if (pipCanvas && pipRenderer && pipCamera) {
    pipCamera.aspect = pipCanvas.clientWidth / pipCanvas.clientHeight;
    pipCamera.updateProjectionMatrix();
    pipRenderer.setSize(pipCanvas.clientWidth, pipCanvas.clientHeight);
  }
}
