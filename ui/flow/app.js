const API_FLOW = "/api/flow";
const FLOW_SLOT_PROPOSED = "proposed";
const FLOW_SLOT_CURRENT = "current";

const editorRoot = document.getElementById("drawflow");
const statusPill = document.getElementById("statusPill");
const storagePathLabel = document.getElementById("storagePath");
const activeSlotLabel = document.getElementById("activeSlot");
const slotToggle = document.getElementById("slotToggle");

const nodeIdEl = document.getElementById("nodeId");
const nodeLabelEl = document.getElementById("nodeLabel");
const nodeTypeDisplayEl = document.getElementById("nodeTypeDisplay");
const nodeDescriptionEl = document.getElementById("nodeDescription");

const addNodeBtn = document.getElementById("addNodeBtn");
const deleteNodeBtn = document.getElementById("deleteNodeBtn");
const applyNodeBtn = document.getElementById("applyNodeBtn");

const saveBtn = document.getElementById("saveBtn");
const copyCurrentBtn = document.getElementById("copyCurrentBtn");
const resetBtn = document.getElementById("resetBtn");
const exportBtn = document.getElementById("exportBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const canvasWrap = document.querySelector(".canvas-wrap");

const editor = new Drawflow(editorRoot);
editor.reroute = true;
editor.start();
editor.zoom_out();

const state = {
  flow: null,
  selectedNodeId: null,
  drawToLogical: new Map(),
  logicalToDraw: new Map(),
  edgeKeys: new Set(),
  suppressEvents: false,
  dirty: false,
  activeSlot: FLOW_SLOT_PROPOSED,
  readOnly: false,
};

function notify(message, tone = "info") {
  statusPill.textContent = message;
  statusPill.classList.remove("warn", "error");
  if (tone === "warn") {
    statusPill.classList.add("warn");
  }
  if (tone === "error") {
    statusPill.classList.add("error");
  }
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setDirty(value) {
  state.dirty = value;
  if (value) {
    notify("Unsaved changes", "warn");
  }
}

function setActiveSlot(slot) {
  state.activeSlot = slot;
  state.readOnly = slot === FLOW_SLOT_CURRENT;
  document.body.dataset.mode = state.activeSlot;
  activeSlotLabel.textContent = `Mode: ${slot}${state.readOnly ? " (read-only)" : ""}`;
  if (slotToggle) {
    slotToggle.checked = state.readOnly;
  }
  saveBtn.disabled = state.readOnly;
  resetBtn.disabled = state.readOnly;

  const editControls = [
    addNodeBtn,
    deleteNodeBtn,
    applyNodeBtn,
    nodeLabelEl,
    nodeDescriptionEl,
  ];
  for (const control of editControls) {
    control.disabled = state.readOnly;
  }
}

function ioForType(type) {
  if (type === "trigger") {
    return { inputs: 0, outputs: 1 };
  }
  if (type === "output") {
    return { inputs: 1, outputs: 0 };
  }
  return { inputs: 1, outputs: 1 };
}

function edgeKey(source, sourcePort, target, targetPort) {
  return `${source}|${sourcePort}|${target}|${targetPort}`;
}

function makeNodeHtml(node) {
  let statusClass = "";
  const type = String(node.type || "action").trim().toLowerCase();
  const typeClass = ["proposed", "trigger", "action", "decision", "data", "output"].includes(type)
    ? `flow-node--type-${type}`
    : "flow-node--type-action";
  if (node.status === "deprecated") {
    statusClass = "flow-node--deprecated";
  }
  return `
    <div class="flow-node ${typeClass} ${statusClass}">
      <div class="flow-node__head">
        <span class="flow-node__type">${escapeHtml(type)}</span>
        <span class="flow-node__status">${escapeHtml(node.status || "current")}</span>
      </div>
      <div class="flow-node__label">${escapeHtml(node.label || node.id)}</div>
      <div class="flow-node__desc">${escapeHtml(node.description || "No description")}</div>
    </div>
  `;
}

function findNode(nodeId) {
  if (!state.flow) {
    return null;
  }
  return state.flow.nodes.find((node) => node.id === nodeId) || null;
}

function getSelectedNode() {
  if (!state.selectedNodeId) {
    return null;
  }
  return findNode(state.selectedNodeId);
}

function refreshInspector(node) {
  if (!node) {
    nodeIdEl.value = "";
    nodeLabelEl.value = "";
    nodeTypeDisplayEl.value = "";
    nodeDescriptionEl.value = "";
    return;
  }

  nodeIdEl.value = node.id;
  nodeLabelEl.value = node.label || "";
  nodeTypeDisplayEl.value = node.type || "unknown";
  nodeDescriptionEl.value = node.description || "";
}

function normalizeIdPart(value) {
  const cleaned = String(value || "node")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || "node";
}

function makeNodeId(type) {
  const prefix = normalizeIdPart(type);
  let index = 1;
  let candidate = `${prefix}_${index}`;
  const allIds = new Set((state.flow?.nodes || []).map((node) => node.id));
  while (allIds.has(candidate)) {
    index += 1;
    candidate = `${prefix}_${index}`;
  }
  return candidate;
}

function addNodeToCanvas(node) {
  const io = ioForType(node.type);
  const drawId = String(
    editor.addNode(
      node.type || "action",
      io.inputs,
      io.outputs,
      Number(node.x || 100),
      Number(node.y || 100),
      "",
      node,
      makeNodeHtml(node),
      false
    )
  );
  state.logicalToDraw.set(node.id, drawId);
  state.drawToLogical.set(drawId, node.id);
}

function addEdgeToCanvas(edge) {
  const outputId = state.logicalToDraw.get(edge.source);
  const inputId = state.logicalToDraw.get(edge.target);
  if (!outputId || !inputId) {
    return;
  }
  const outputClass = edge.source_port || "output_1";
  const inputClass = edge.target_port || "input_1";
  const key = edgeKey(edge.source, outputClass, edge.target, inputClass);
  if (state.edgeKeys.has(key)) {
    return;
  }
  state.edgeKeys.add(key);
  editor.addConnection(outputId, inputId, outputClass, inputClass);
}

function renderFlow(flow) {
  state.suppressEvents = true;
  editor.clear();
  state.drawToLogical.clear();
  state.logicalToDraw.clear();
  state.edgeKeys.clear();

  for (const node of flow.nodes) {
    addNodeToCanvas(node);
  }
  for (const edge of flow.edges) {
    addEdgeToCanvas(edge);
  }

  state.suppressEvents = false;
  refreshInspector(getSelectedNode());
}

function updateNodeDom(logicalNodeId) {
  const node = findNode(logicalNodeId);
  const drawId = state.logicalToDraw.get(logicalNodeId);
  if (!node || !drawId) {
    return;
  }

  editor.updateNodeDataFromId(drawId, node);
  const content = document.querySelector(`#node-${drawId} .drawflow_content_node`);
  if (content) {
    content.innerHTML = makeNodeHtml(node);
  }
}

function makeEdgeId() {
  let index = state.flow.edges.length + 1;
  let candidate = `e${index}`;
  const all = new Set(state.flow.edges.map((edge) => edge.id));
  while (all.has(candidate)) {
    index += 1;
    candidate = `e${index}`;
  }
  return candidate;
}

async function loadFlow(slot = FLOW_SLOT_PROPOSED) {
  notify(`Loading ${slot} flow...`);
  try {
    const response = await fetch(`${API_FLOW}?slot=${encodeURIComponent(slot)}`, { method: "GET" });
    if (!response.ok) {
      throw new Error(`failed with ${response.status}`);
    }
    const payload = await response.json();
    setActiveSlot(payload.slot || slot);
    state.flow = payload.flow;
    state.selectedNodeId = null;
    storagePathLabel.textContent = `Storage: ${payload.source || "unknown"}`;
    renderFlow(state.flow);
    setDirty(false);
    notify(`${state.activeSlot} flow loaded${state.readOnly ? " (read-only)" : ""}`);
  } catch (error) {
    notify(`Could not load flow: ${error.message}`, "error");
  }
}

async function saveFlow() {
  if (!state.flow) {
    return;
  }
  if (state.readOnly) {
    notify("Current flow is read-only. Load proposed to edit.", "warn");
    return;
  }

  state.flow.updated_at = new Date().toISOString();

  notify("Saving proposed flow...");
  try {
    const response = await fetch(`${API_FLOW}?slot=${FLOW_SLOT_PROPOSED}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flow: state.flow }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `failed with ${response.status}`);
    }
    setActiveSlot(payload.slot || FLOW_SLOT_PROPOSED);
    state.flow = payload.flow;
    storagePathLabel.textContent = `Storage: ${payload.saved_to || "unknown"}`;
    setDirty(false);
    notify("Proposed flow saved");
  } catch (error) {
    notify(`Save failed: ${error.message}`, "error");
  }
}

async function resetFlow() {
  if (state.readOnly) {
    notify("Current flow is read-only. Load proposed to reset.", "warn");
    return;
  }

  const approved = window.confirm("Reset proposed flow from current baseline? Unsaved changes will be lost.");
  if (!approved) {
    return;
  }

  notify("Resetting proposed flow...");
  try {
    const response = await fetch(`${API_FLOW}/reset?slot=${FLOW_SLOT_PROPOSED}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `failed with ${response.status}`);
    }
    setActiveSlot(payload.slot || FLOW_SLOT_PROPOSED);
    state.flow = payload.flow;
    state.selectedNodeId = null;
    storagePathLabel.textContent = `Storage: ${payload.saved_to || "unknown"}`;
    renderFlow(state.flow);
    setDirty(false);
    notify(`Proposed reset from ${payload.reset_from || "current"}`);
  } catch (error) {
    notify(`Reset failed: ${error.message}`, "error");
  }
}

async function copyCurrentToProposed() {
  notify("Copying current -> proposed...");
  try {
    const response = await fetch(`${API_FLOW}/copy-current-to-proposed`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `failed with ${response.status}`);
    }
    notify("Copied current to proposed");
    await loadFlow(FLOW_SLOT_PROPOSED);
  } catch (error) {
    notify(`Copy failed: ${error.message}`, "error");
  }
}

function addNode(connectFromNodeId = null) {
  if (!state.flow || state.readOnly) {
    notify("Load proposed flow to edit.", "warn");
    return;
  }

  const fromNode = connectFromNodeId ? findNode(connectFromNodeId) : null;
  const node = {
    id: makeNodeId("proposed"),
    type: "proposed",
    label: "Proposed Change",
    description: "",
    status: "current",
    x: fromNode ? Number(fromNode.x || 100) + 260 : 160,
    y: fromNode ? Number(fromNode.y || 100) + 120 : 160,
  };

  state.flow.nodes.push(node);
  addNodeToCanvas(node);
  state.selectedNodeId = node.id;
  refreshInspector(node);

  if (fromNode) {
    const sourceIo = ioForType(fromNode.type);
    const targetIo = ioForType(node.type);
    if (sourceIo.outputs < 1 || targetIo.inputs < 1) {
      setDirty(true);
      return;
    }
    const newEdge = {
      id: makeEdgeId(),
      source: fromNode.id,
      target: node.id,
      source_port: "output_1",
      target_port: "input_1",
      label: "",
    };
    state.flow.edges.push(newEdge);
    addEdgeToCanvas(newEdge);
  }

  setDirty(true);
}

function deleteSelectedNode() {
  if (state.readOnly) {
    notify("Load proposed flow to edit.", "warn");
    return;
  }
  const selected = getSelectedNode();
  if (!selected) {
    notify("Select a node first", "warn");
    return;
  }

  const drawId = state.logicalToDraw.get(selected.id);
  if (!drawId) {
    return;
  }

  editor.removeNodeId(`node-${drawId}`);
}

function copyJson() {
  if (!state.flow) {
    return;
  }
  const content = JSON.stringify(state.flow, null, 2);
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(content);
    notify("Flow JSON copied");
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = content;
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
  notify("Flow JSON copied");
}

editor.on("nodeSelected", (drawId) => {
  const logicalId = state.drawToLogical.get(String(drawId));
  state.selectedNodeId = logicalId || null;
  refreshInspector(getSelectedNode());
});

editor.on("nodeUnselected", () => {
  state.selectedNodeId = null;
  refreshInspector(null);
});

editor.on("nodeMoved", (drawId) => {
  if (state.suppressEvents || !state.flow || state.readOnly) {
    return;
  }
  const logicalId = state.drawToLogical.get(String(drawId));
  if (!logicalId) {
    return;
  }

  const node = findNode(logicalId);
  const drawNode = editor.getNodeFromId(drawId);
  if (!node || !drawNode) {
    return;
  }
  node.x = Number(drawNode.pos_x || 0);
  node.y = Number(drawNode.pos_y || 0);
  setDirty(true);
});

editor.on("nodeRemoved", (drawId) => {
  if (state.suppressEvents || !state.flow || state.readOnly) {
    return;
  }

  const logicalId = state.drawToLogical.get(String(drawId));
  if (!logicalId) {
    return;
  }

  state.flow.nodes = state.flow.nodes.filter((node) => node.id !== logicalId);
  state.flow.edges = state.flow.edges.filter((edge) => edge.source !== logicalId && edge.target !== logicalId);
  state.selectedNodeId = null;
  state.drawToLogical.delete(String(drawId));
  state.logicalToDraw.delete(logicalId);
  state.edgeKeys = new Set(
    state.flow.edges.map((edge) => edgeKey(edge.source, edge.source_port || "output_1", edge.target, edge.target_port || "input_1"))
  );
  refreshInspector(null);
  setDirty(true);
});

editor.on("connectionCreated", (connection) => {
  if (state.suppressEvents || !state.flow || state.readOnly) {
    return;
  }

  const source = state.drawToLogical.get(String(connection.output_id));
  const target = state.drawToLogical.get(String(connection.input_id));
  if (!source || !target) {
    return;
  }

  const sourcePort = connection.output_class || "output_1";
  const targetPort = connection.input_class || "input_1";
  const key = edgeKey(source, sourcePort, target, targetPort);
  if (state.edgeKeys.has(key)) {
    if (typeof editor.removeSingleConnection === "function") {
      editor.removeSingleConnection(connection.output_id, connection.input_id, sourcePort, targetPort);
    }
    return;
  }

  state.edgeKeys.add(key);
  state.flow.edges.push({
    id: makeEdgeId(),
    source,
    target,
    source_port: sourcePort,
    target_port: targetPort,
    label: "",
  });
  setDirty(true);
});

editor.on("connectionRemoved", (connection) => {
  if (state.suppressEvents || !state.flow || state.readOnly) {
    return;
  }

  const source = state.drawToLogical.get(String(connection.output_id));
  const target = state.drawToLogical.get(String(connection.input_id));
  if (!source || !target) {
    return;
  }

  const sourcePort = connection.output_class || "output_1";
  const targetPort = connection.input_class || "input_1";
  const key = edgeKey(source, sourcePort, target, targetPort);
  state.edgeKeys.delete(key);

  state.flow.edges = state.flow.edges.filter(
    (edge) => !(edge.source === source && edge.target === target && (edge.source_port || "output_1") === sourcePort && (edge.target_port || "input_1") === targetPort)
  );
  setDirty(true);
});

addNodeBtn.addEventListener("click", () => {
  if (state.readOnly) {
    notify("Load proposed flow to edit.", "warn");
    return;
  }
  addNode(state.selectedNodeId);
});

deleteNodeBtn.addEventListener("click", deleteSelectedNode);

applyNodeBtn.addEventListener("click", () => {
  if (state.readOnly) {
    notify("Load proposed flow to edit.", "warn");
    return;
  }
  const node = getSelectedNode();
  if (!node) {
    notify("Select a node to edit", "warn");
    return;
  }

  node.label = nodeLabelEl.value.trim() || node.id;
  node.description = nodeDescriptionEl.value.trim();
  updateNodeDom(node.id);
  setDirty(true);
});

slotToggle.addEventListener("change", async () => {
  const nextSlot = slotToggle.checked ? FLOW_SLOT_CURRENT : FLOW_SLOT_PROPOSED;
  if (state.activeSlot === FLOW_SLOT_PROPOSED && state.dirty && nextSlot === FLOW_SLOT_CURRENT) {
    const approved = window.confirm("You have unsaved proposed changes. Switch view anyway?");
    if (!approved) {
      slotToggle.checked = false;
      return;
    }
  }
  await loadFlow(nextSlot);
});
saveBtn.addEventListener("click", saveFlow);
copyCurrentBtn.addEventListener("click", copyCurrentToProposed);
resetBtn.addEventListener("click", resetFlow);
exportBtn.addEventListener("click", copyJson);
fullscreenBtn.addEventListener("click", async () => {
  if (!canvasWrap) {
    return;
  }
  try {
    if (!document.fullscreenElement) {
      await canvasWrap.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (error) {
    notify(`Fullscreen failed: ${error.message}`, "error");
  }
});

document.addEventListener("fullscreenchange", () => {
  const isFullscreen = Boolean(document.fullscreenElement === canvasWrap);
  fullscreenBtn.classList.toggle("is-active", isFullscreen);
  const label = isFullscreen ? "Exit canvas fullscreen" : "Enter canvas fullscreen";
  fullscreenBtn.setAttribute("aria-label", label);
  fullscreenBtn.setAttribute("title", label);
  if (typeof editor.updateConnectionNodes !== "function") {
    return;
  }
  for (const drawId of state.logicalToDraw.values()) {
    editor.updateConnectionNodes(`node-${drawId}`);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Delete" || event.key === "Backspace") {
    if (state.readOnly) {
      return;
    }
    const tag = String(document.activeElement?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") {
      return;
    }
    deleteSelectedNode();
  }

  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveFlow();
  }
});

window.addEventListener("beforeunload", (event) => {
  if (!state.dirty) {
    return;
  }
  event.preventDefault();
  event.returnValue = "";
});

loadFlow(FLOW_SLOT_PROPOSED);
