// Klinik Chong Developer Monitor - Version 22
const MONITOR_STORAGE_KEY = "klinik_chong_monitor_ver22";
const DEVELOPER_CONFIG = window.KLINIK_CHONG_MODEL_CONFIG || {};
const REMOTE_DATABASE_API_BASE = String(DEVELOPER_CONFIG.bookingApiBase || "").replace(/\/+$/u, "");
const DEVELOPER_API_KEY = String(DEVELOPER_CONFIG.developerApiKey || "");
const LOCAL_BOOKING_PROXY_BASE = "/api/booking-proxy";

const turnCountElement = document.getElementById("turn-count");
const detectedIntentElement = document.getElementById("detected-intent");
const replyTypeElement = document.getElementById("reply-type");
const replyConversationElement = document.getElementById("reply-conversation");
const processingStatusElement = document.getElementById("processing-status");
const modelRuntimeModeElement = document.getElementById("model-runtime-mode");
const turnSelector = document.getElementById("turn-selector");
const originalInputElement = document.getElementById("original-input");
const normalizedInputElement = document.getElementById("normalized-input");
const tokenTableBody = document.getElementById("token-table-body");
const tokenCountElement = document.getElementById("token-count");
const thresholdTableBody = document.getElementById("threshold-table-body");
const replyJsonElement = document.getElementById("reply-json");
const gibberishReasonsElement = document.getElementById("gibberish-reasons");
const layerStatusElement = document.getElementById("layer-status");
const resourceStatusElement = document.getElementById("resource-status");
const lastUpdatedElement = document.getElementById("last-updated");
const refreshButton = document.getElementById("refresh-button");
const backToChatbotButton = document.getElementById("back-to-chatbot");
const monitorTabs = [...document.querySelectorAll("[data-monitor-tab]")];
const conversationMonitor = document.getElementById("conversation-monitor");
const intentionSourceElement = document.getElementById("intention-source");
const detectedEmotionElement = document.getElementById("detected-emotion");
const retrievalTraceElement = document.getElementById("retrieval-trace");
const retrievedContextElement = document.getElementById("retrieved-context");
const databaseMonitor = document.getElementById("database-monitor");
const databaseConnection = document.getElementById("database-connection");
const databaseName = document.getElementById("database-name");
const databaseTableCount = document.getElementById("database-table-count");
const databasePath = document.getElementById("database-path");
const databaseRefreshButton = document.getElementById("database-refresh-button");
const databaseErd = document.getElementById("database-erd");
const erdLines = document.getElementById("erd-lines");
const erdTableGrid = document.getElementById("erd-table-grid");
const relationshipList = document.getElementById("relationship-list");
const databaseTableButtons = document.getElementById("database-table-buttons");
const databaseRecordHead = document.getElementById("database-record-head");
const databaseRecordBody = document.getElementById("database-record-body");
const recordCount = document.getElementById("record-count");
const databaseError = document.getElementById("database-error");
const doctorLeaveToggle = document.getElementById("doctor-leave-toggle");
const doctorLeaveForm = document.getElementById("doctor-leave-form");
const doctorLeaveDoctor = document.getElementById("doctor-leave-doctor");
const doctorLeaveDate = document.getElementById("doctor-leave-date");
const doctorLeaveType = document.getElementById("doctor-leave-type");
const doctorLeaveReason = document.getElementById("doctor-leave-reason");
const doctorLeaveConfirm = document.getElementById("doctor-leave-confirm");
const doctorLeaveStatus = document.getElementById("doctor-leave-status");
const appointmentRefresh = document.getElementById("appointment-refresh");
const appointmentSelector = document.getElementById("appointment-selector");
const appointmentDetail = document.getElementById("appointment-detail");
const appointmentComplete = document.getElementById("appointment-complete");
const appointmentCompleteStatus = document.getElementById("appointment-complete-status");

let currentState = null;
let databaseSchema = null;
let databaseLoading = false;
let selectedDatabaseTable = null;
let developerAppointments = [];

function readMonitorState() {
  try {
    const parsed = JSON.parse(localStorage.getItem(MONITOR_STORAGE_KEY));
    return parsed && Array.isArray(parsed.history) ? parsed : null;
  } catch (error) {
    console.warn("Unable to read monitor storage:", error);
    return null;
  }
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function renderTurnOptions(state, preferredTurn = null) {
  turnSelector.innerHTML = "";

  if (!state || state.history.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No user query yet";
    turnSelector.appendChild(option);
    return null;
  }

  [...state.history].reverse().forEach((record) => {
    const option = document.createElement("option");
    option.value = String(record.turn);
    option.textContent = `Turn ${record.turn} · ${record.displayedInput || record.originalInput}`;
    turnSelector.appendChild(option);
  });

  const selectedTurn = preferredTurn || state.currentTurn || state.history.at(-1).turn;
  turnSelector.value = String(selectedTurn);
  return state.history.find((record) => record.turn === Number(selectedTurn)) || state.history.at(-1);
}

function renderTokenTable(tokens) {
  tokenTableBody.innerHTML = "";
  const rows = Array.isArray(tokens) ? tokens : [];
  tokenCountElement.textContent = `${rows.length} token${rows.length === 1 ? "" : "s"}`;

  if (rows.length === 0) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    row.innerHTML = '<td colspan="8">No token data yet.</td>';
    tokenTableBody.appendChild(row);
    return;
  }

  rows.forEach((item) => {
    const row = document.createElement("tr");
    const confidence = Number.isFinite(Number(item.confidence))
      ? `${(Number(item.confidence) * 100).toFixed(2)}%`
      : "-";
    const values = [
      item.input ?? item.token,
      item.category,
      item.language,
      item.pinyin_segment || "-",
      item.label_source,
      confidence,
      item.transform_to,
      item.transform_source
    ];

    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value ?? "-";
      if (index === 1) cell.className = "category-cell";
      if (index === 2) cell.className = `language-cell language-${String(value || "other")}`;
      if (index === 3) cell.className = "pinyin-segment-cell";
      if (index === 4) cell.className = "label-source-cell";
      if (index === 5) cell.className = "confidence-cell";
      if (index === 6) cell.className = "transform-cell";
      if (index === 7) cell.className = "transform-source-cell";
      row.appendChild(cell);
    });

    tokenTableBody.appendChild(row);
  });
}

function renderThresholdSummary(summary) {
  thresholdTableBody.innerHTML = "";
  if (!summary) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    row.innerHTML = '<td colspan="5">No threshold data yet.</td>';
    thresholdTableBody.appendChild(row);
    return;
  }

  const row = document.createElement("tr");
  const values = [
    summary.input,
    summary.malay_percent,
    summary.chinese_percent,
    summary.other_percent,
    summary.reply_route
  ];

  values.forEach((value, index) => {
    const cell = document.createElement("td");
    cell.textContent = value ?? "—";
    if (index >= 1 && index <= 3) cell.className = "threshold-cell";
    if (index === 4) cell.className = "route-cell";
    row.appendChild(cell);
  });
  thresholdTableBody.appendChild(row);
}

function renderRecord(record) {
  if (!record) {
    detectedIntentElement.textContent = "—";
    replyTypeElement.textContent = "—";
    replyConversationElement.textContent = currentState?.replyConversation || "—";
    processingStatusElement.textContent = "Waiting";
    originalInputElement.textContent = "No input received.";
    normalizedInputElement.textContent = "No processed input.";
    replyJsonElement.textContent = "No reply generated.";
    gibberishReasonsElement.textContent = "None.";
    layerStatusElement.textContent = "No layer data yet.";
    renderTokenTable([]);
    renderThresholdSummary(null);
    return;
  }

  detectedIntentElement.textContent = record.detectedIntent || "—";
  const decision = record.intentionDecision || {};
  const confidence = Number.isFinite(Number(decision.confidence))
    ? ` · ${(Number(decision.confidence) * 100).toFixed(2)}%`
    : "";
  intentionSourceElement.textContent = `${decision.source || "—"}${confidence}`;
  detectedEmotionElement.textContent = record.detectedEmotion || "neutral";
  retrievalTraceElement.textContent = record.retrievalTrace
    ? JSON.stringify(record.retrievalTrace, null, 2)
    : "No RAG or database retrieval for this turn.";
  retrievedContextElement.textContent = record.retrievedContext
    || "No retrieved context for this turn.";
  replyTypeElement.textContent = record.reply?.type || "—";
  replyConversationElement.textContent = record.replyConversation || currentState?.replyConversation || "pending";
  processingStatusElement.textContent = record.status === "typing"
    ? "Typing / pending reply"
    : record.processingStatus || record.status || "—";
  originalInputElement.textContent = record.originalInput || "";
  normalizedInputElement.textContent = record.normalizedInput || "";
  replyJsonElement.textContent = JSON.stringify(record.reply || {}, null, 2);
  gibberishReasonsElement.textContent = record.gibberishReasons?.length
    ? record.gibberishReasons.join("\n")
    : "None.";
  layerStatusElement.textContent = JSON.stringify({
    ...(record.layerStatus || {}),
    modelStatus: record.modelStatus || {},
    pinyinNormalizationStatus: record.pinyinNormalizationStatus || "—"
  }, null, 2);
  renderTokenTable(record.tokenAnalysis);
  renderThresholdSummary(record.languageSummary);
}

function renderResourceStatus(state) {
  const runtimeMode = String(state?.modelRuntime?.runtimeMode || "unknown").toUpperCase();
  if (modelRuntimeModeElement) {
    modelRuntimeModeElement.textContent = runtimeMode;
    modelRuntimeModeElement.title = state?.modelRuntime?.warning || "";
  }

  resourceStatusElement.textContent = JSON.stringify({
    counts: state?.resources || {},
    errors: state?.resourceErrors || [],
    warnings: state?.resourceWarnings || [],
    sources: state?.resourceSources || {},
    canonical_paths: state?.resourcePaths || {},
    model_runtime: state?.modelRuntime || {
      backendReachable: false,
      mBert: { loaded: false },
      pinyin2HanziHmm: { loaded: false }
    }
  }, null, 2);
}

function refreshMonitor(preferredTurn = null) {
  currentState = readMonitorState();
  turnCountElement.textContent = currentState?.turnCount ?? 0;
  lastUpdatedElement.textContent = `Last updated: ${formatDateTime(currentState?.lastUpdatedAt)}`;
  renderResourceStatus(currentState);
  const record = renderTurnOptions(currentState, preferredTurn);
  renderRecord(record);
}

turnSelector.addEventListener("change", () => {
  if (!currentState) return;
  const record = currentState.history.find((item) => item.turn === Number(turnSelector.value));
  renderRecord(record);
});

refreshButton.addEventListener("click", () => {
  const selectedTurn = Number(turnSelector.value) || null;
  refreshMonitor(selectedTurn);
});

window.addEventListener("storage", (event) => {
  if (event.key === MONITOR_STORAGE_KEY) refreshMonitor();
});

window.addEventListener("DOMContentLoaded", () => refreshMonitor());


backToChatbotButton?.addEventListener("click", () => {
  if (window.parent && window.parent !== window) {
    window.parent.postMessage({ type: "show-klinik-chatbot" }, "*");
  } else {
    window.location.href = "index_ver22.html";
  }
});

window.addEventListener("message", (event) => {
  if (event.data?.type === "refresh-klinik-monitor") refreshMonitor();
  if (event.data?.type === "refresh-klinik-database" && databaseSchema) {
    loadDatabaseSchema(true);
  }
});

async function developerApiJson(path, options = {}) {
  const requestUrl = `${LOCAL_BOOKING_PROXY_BASE}${path}`;
  const headers = {
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(options.headers || {})
  };
  const response = await fetch(requestUrl, {
    cache: "no-store",
    method: options.method || "GET",
    headers,
    body: options.body
  });
  let data = {};
  try { data = await response.json(); } catch (_) { data = {}; }
  if (!response.ok) {
    throw new Error(data.detail || data.message || `Developer API failed (${response.status}).`);
  }
  return data;
}

async function monitorBookingApiJson(path) {
  const requestUrl = `${LOCAL_BOOKING_PROXY_BASE}${path}`;
  const response = await fetch(requestUrl, { cache: "no-store" });
  let data = {};
  try { data = await response.json(); } catch (_) { data = {}; }
  if (!response.ok) {
    throw new Error(data.detail || data.message || `Booking API failed (${response.status}).`);
  }
  return data;
}

function todayIsoForDeveloper() {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

async function loadDoctorLeaveDoctors() {
  if (!doctorLeaveDoctor) return;
  doctorLeaveDoctor.disabled = true;
  doctorLeaveDoctor.replaceChildren(new Option("Loading doctors…", ""));
  try {
    const data = await monitorBookingApiJson("/doctors");
    const doctors = Array.isArray(data.doctors) ? data.doctors : [];
    doctorLeaveDoctor.replaceChildren(new Option("Select doctor", ""));
    doctors.forEach(doctor => {
      const option = document.createElement("option");
      option.value = doctor.d_id;
      const name = String(doctor.d_name || "").replace(/^Dr\.?\s*/iu, "").trim();
      option.textContent = `Dr. ${name} (${doctor.d_id} · ${doctor.d_expertise || "—"})`;
      doctorLeaveDoctor.appendChild(option);
    });
    doctorLeaveDoctor.disabled = false;
  } catch (error) {
    doctorLeaveDoctor.replaceChildren(new Option("Unable to load doctors", ""));
    if (doctorLeaveStatus) doctorLeaveStatus.textContent = error.message;
  }
}

function appointmentValue(value) {
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function appointmentDetailGroup(title, fields) {
  const section = document.createElement("section");
  section.className = "appointment-detail-group";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const list = document.createElement("dl");
  fields.forEach(([label, value]) => {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = appointmentValue(value);
    list.append(term, detail);
  });
  section.append(heading, list);
  return section;
}

function renderSelectedAppointment() {
  const appointment = developerAppointments.find(
    item => item.appointment_id === appointmentSelector?.value
  );
  appointmentDetail.replaceChildren();
  appointmentCompleteStatus.textContent = "";
  if (!appointment) {
    const placeholder = document.createElement("p");
    placeholder.className = "database-placeholder";
    placeholder.textContent = "Select an appointment to view its details.";
    appointmentDetail.appendChild(placeholder);
    appointmentComplete.disabled = true;
    return;
  }

  const status = document.createElement("span");
  status.className = "appointment-status-badge " + (
    appointment.display_status === "OVERDUE"
      ? "appointment-status-overdue"
      : appointment.display_status === "COMPLETED"
        ? "appointment-status-completed"
        : "appointment-status-active"
  );
  status.textContent = appointment.display_status;

  appointmentDetail.append(
    appointmentDetailGroup("Appointment", [
      ["Appointment ID", appointment.appointment_id],
      ["Status", appointment.display_status],
      ["Stored status", appointment.stored_status],
      ["Date", appointment.appointment_date],
      ["Time", `${appointmentValue(appointment.start_time)}–${appointmentValue(appointment.end_time)}`],
      ["Clinical reason", appointment.clinical_note],
      ["Created", appointment.appointment_created_at],
      ["Updated", appointment.appointment_updated_at],
      ["Completed", appointment.completed_at]
    ]),
    appointmentDetailGroup("Patient", [
      ["Patient ID", appointment.p_id],
      ["Name", appointment.p_name],
      ["Date of birth", appointment.p_dob],
      ["Gender", appointment.p_gender],
      ["IC", appointment.p_ic],
      ["Contact", appointment.p_contact],
      ["Created", appointment.patient_created_at],
      ["Updated", appointment.patient_updated_at]
    ]),
    appointmentDetailGroup("Doctor", [
      ["Doctor ID", appointment.d_id],
      ["Name", appointment.d_name],
      ["Gender", appointment.d_gender],
      ["Expertise", appointment.d_expertise]
    ])
  );
  appointmentDetail.prepend(status);
  appointmentComplete.disabled = ["COMPLETED", "CANCELLED", "RESCHEDULED"].includes(
    appointment.stored_status
  );
}

async function loadDeveloperAppointments(preserveSelection = true) {
  if (!appointmentSelector) return;
  const previous = preserveSelection ? appointmentSelector.value : "";
  appointmentRefresh.disabled = true;
  appointmentSelector.disabled = true;
  appointmentCompleteStatus.textContent = "Loading appointments…";
  try {
    const data = await developerApiJson("/developer/appointments");
    developerAppointments = Array.isArray(data.appointments) ? data.appointments : [];
    appointmentSelector.replaceChildren(new Option(
      developerAppointments.length ? "Select appointment" : "No appointments available",
      ""
    ));
    developerAppointments.forEach(appointment => {
      const option = document.createElement("option");
      option.value = appointment.appointment_id;
      option.textContent = `${appointment.appointment_id} · ${appointment.p_name} · ${appointment.appointment_date} ${appointment.start_time} · ${appointment.display_status}`;
      appointmentSelector.appendChild(option);
    });
    if (previous && developerAppointments.some(item => item.appointment_id === previous)) {
      appointmentSelector.value = previous;
    }
    appointmentCompleteStatus.textContent = `${developerAppointments.length} appointment(s) loaded.`;
    renderSelectedAppointment();
  } catch (error) {
    appointmentCompleteStatus.textContent = error.message;
    developerAppointments = [];
    renderSelectedAppointment();
  } finally {
    appointmentRefresh.disabled = false;
    appointmentSelector.disabled = false;
  }
}

function createErdColumn(table, column) {
  const row = document.createElement("div");
  row.className = "erd-column";

  const badges = document.createElement("span");
  badges.className = "erd-column-badges";
  if (column.primary_key_position) {
    const primaryKey = document.createElement("b");
    primaryKey.className = "key-badge primary-key";
    primaryKey.textContent = "PK";
    badges.appendChild(primaryKey);
  }
  const isForeignKey = table.foreign_keys.some(item => item.from_column === column.name);
  if (isForeignKey) {
    const foreignKey = document.createElement("b");
    foreignKey.className = "key-badge foreign-key";
    foreignKey.textContent = "FK";
    badges.appendChild(foreignKey);
  }

  const name = document.createElement("span");
  name.className = "erd-column-name";
  name.textContent = column.name;

  const type = document.createElement("span");
  type.className = "erd-column-type";
  type.textContent = column.type;

  row.append(badges, name, type);
  return row;
}

function drawRelationshipLines() {
  erdLines.replaceChildren();
  if (!databaseSchema?.relationships?.length || databaseMonitor.hidden) return;

  const canvasRect = databaseErd.getBoundingClientRect();
  const cards = new Map(
    [...erdTableGrid.querySelectorAll("[data-database-table]")]
      .map(card => [card.dataset.databaseTable, card])
  );
  erdLines.setAttribute("viewBox", `0 0 ${canvasRect.width} ${canvasRect.height}`);
  erdLines.setAttribute("width", String(canvasRect.width));
  erdLines.setAttribute("height", String(canvasRect.height));

  const namespace = "http://www.w3.org/2000/svg";
  const definitions = document.createElementNS(namespace, "defs");
  const makeMarker = (id, pathData, refX) => {
    const marker = document.createElementNS(namespace, "marker");
    marker.setAttribute("id", id);
    marker.setAttribute("markerWidth", "14");
    marker.setAttribute("markerHeight", "14");
    marker.setAttribute("refX", String(refX));
    marker.setAttribute("refY", "7");
    marker.setAttribute("orient", "auto-start-reverse");
    marker.setAttribute("markerUnits", "userSpaceOnUse");
    const symbol = document.createElementNS(namespace, "path");
    symbol.setAttribute("d", pathData);
    symbol.setAttribute("class", "erd-cardinality-marker");
    marker.appendChild(symbol);
    definitions.appendChild(marker);
  };

  // Foreign-key side = many (crow's foot); referenced key side = exactly one.
  makeMarker("erd-many", "M13 7 L2 1 M13 7 L2 7 M13 7 L2 13", 13);
  makeMarker("erd-one", "M4 1 L4 13 M9 1 L9 13", 9);
  erdLines.appendChild(definitions);

  databaseSchema.relationships.forEach(relationship => {
    const fromCard = cards.get(relationship.from_table);
    const toCard = cards.get(relationship.to_table);
    if (!fromCard || !toCard) return;
    const fromRect = fromCard.getBoundingClientRect();
    const toRect = toCard.getBoundingClientRect();
    const fromCenterX = fromRect.left - canvasRect.left + fromRect.width / 2;
    const fromCenterY = fromRect.top - canvasRect.top + fromRect.height / 2;
    const toCenterX = toRect.left - canvasRect.left + toRect.width / 2;
    const toCenterY = toRect.top - canvasRect.top + toRect.height / 2;
    const horizontal = Math.abs(toCenterX - fromCenterX) >= Math.abs(toCenterY - fromCenterY);
    const otherRectangles = [...cards.entries()]
      .filter(([name]) => ![relationship.from_table, relationship.to_table].includes(name))
      .map(([, card]) => {
        const rect = card.getBoundingClientRect();
        return {
          left: rect.left - canvasRect.left,
          right: rect.right - canvasRect.left,
          top: rect.top - canvasRect.top,
          bottom: rect.bottom - canvasRect.top
        };
      });
    const segmentHitsTable = (x1, y1, x2, y2) => otherRectangles.some(rect => {
      if (y1 === y2) {
        return y1 > rect.top && y1 < rect.bottom &&
          Math.max(Math.min(x1, x2), rect.left) < Math.min(Math.max(x1, x2), rect.right);
      }
      return x1 > rect.left && x1 < rect.right &&
        Math.max(Math.min(y1, y2), rect.top) < Math.min(Math.max(y1, y2), rect.bottom);
    });
    let pathData;

    if (horizontal) {
      const movesRight = toCenterX >= fromCenterX;
      const x1 = movesRight ? fromRect.right - canvasRect.left : fromRect.left - canvasRect.left;
      const x2 = movesRight ? toRect.left - canvasRect.left : toRect.right - canvasRect.left;
      const middleX = (x1 + x2) / 2;
      const directHitsTable = segmentHitsTable(x1, fromCenterY, middleX, fromCenterY) ||
        segmentHitsTable(middleX, fromCenterY, middleX, toCenterY) ||
        segmentHitsTable(middleX, toCenterY, x2, toCenterY);
      if (directHitsTable) {
        const topLane = 8;
        const bottomLane = Math.max(8, canvasRect.height - 8);
        const laneY = Math.abs(fromCenterY - topLane) + Math.abs(toCenterY - topLane) <=
          Math.abs(fromCenterY - bottomLane) + Math.abs(toCenterY - bottomLane)
          ? topLane : bottomLane;
        const fromExitX = x1 + (movesRight ? 14 : -14);
        const toEntryX = x2 + (movesRight ? -14 : 14);
        pathData = `M ${x1} ${fromCenterY} H ${fromExitX} V ${laneY} H ${toEntryX} V ${toCenterY} H ${x2}`;
      } else {
        pathData = `M ${x1} ${fromCenterY} H ${middleX} V ${toCenterY} H ${x2}`;
      }
    } else {
      const movesDown = toCenterY >= fromCenterY;
      const y1 = movesDown ? fromRect.bottom - canvasRect.top : fromRect.top - canvasRect.top;
      const y2 = movesDown ? toRect.top - canvasRect.top : toRect.bottom - canvasRect.top;
      const middleY = (y1 + y2) / 2;
      const directHitsTable = segmentHitsTable(fromCenterX, y1, fromCenterX, middleY) ||
        segmentHitsTable(fromCenterX, middleY, toCenterX, middleY) ||
        segmentHitsTable(toCenterX, middleY, toCenterX, y2);
      if (directHitsTable) {
        const leftLane = 8;
        const rightLane = Math.max(8, canvasRect.width - 8);
        const laneX = Math.abs(fromCenterX - leftLane) + Math.abs(toCenterX - leftLane) <=
          Math.abs(fromCenterX - rightLane) + Math.abs(toCenterX - rightLane)
          ? leftLane : rightLane;
        const fromExitY = y1 + (movesDown ? 14 : -14);
        const toEntryY = y2 + (movesDown ? -14 : 14);
        pathData = `M ${fromCenterX} ${y1} V ${fromExitY} H ${laneX} V ${toEntryY} H ${toCenterX} V ${y2}`;
      } else {
        pathData = `M ${fromCenterX} ${y1} V ${middleY} H ${toCenterX} V ${y2}`;
      }
    }

    const path = document.createElementNS(namespace, "path");
    path.setAttribute("d", pathData);
    path.setAttribute("marker-start", "url(#erd-many)");
    path.setAttribute("marker-end", "url(#erd-one)");
    erdLines.appendChild(path);
  });
}

function renderDatabaseSchema(schema) {
  databaseConnection.textContent = "Connected";
  databaseConnection.className = "database-connected";
  databaseName.textContent = schema.database_name;
  databaseTableCount.textContent = String(schema.table_count);
  databasePath.textContent = schema.database_path;
  databaseError.textContent = "";
  erdTableGrid.replaceChildren();
  relationshipList.replaceChildren();
  databaseTableButtons.replaceChildren();

  schema.tables.forEach(table => {
    const card = document.createElement("article");
    card.className = "erd-table-card";
    card.dataset.databaseTable = table.name;

    const heading = document.createElement("button");
    heading.type = "button";
    heading.className = "erd-table-heading";
    heading.dataset.openTable = table.name;
    const title = document.createElement("strong");
    title.textContent = table.name;
    const count = document.createElement("span");
    count.textContent = `${table.row_count} rows`;
    heading.append(title, count);
    card.appendChild(heading);
    table.columns.forEach(column => card.appendChild(createErdColumn(table, column)));
    erdTableGrid.appendChild(card);

    const tableButton = document.createElement("button");
    tableButton.type = "button";
    tableButton.dataset.openTable = table.name;
    tableButton.textContent = `${table.name} (${table.row_count})`;
    databaseTableButtons.appendChild(tableButton);
  });

  if (schema.relationships.length) {
    schema.relationships.forEach(relationship => {
      const item = document.createElement("code");
      item.textContent = `${relationship.from_table}.${relationship.from_column} → ${relationship.to_table}.${relationship.to_column}`;
      relationshipList.appendChild(item);
    });
  } else {
    const empty = document.createElement("p");
    empty.className = "database-placeholder";
    empty.textContent = "No foreign-key relationships are declared in this database.";
    relationshipList.appendChild(empty);
  }

  requestAnimationFrame(drawRelationshipLines);
}

function renderDatabaseRecords(data) {
  databaseRecordHead.replaceChildren();
  databaseRecordBody.replaceChildren();
  recordCount.textContent = `${data.total_records} record${data.total_records === 1 ? "" : "s"}`;

  const headerRow = document.createElement("tr");
  data.columns.forEach(column => {
    const cell = document.createElement("th");
    cell.textContent = column;
    headerRow.appendChild(cell);
  });
  databaseRecordHead.appendChild(headerRow);

  if (!data.records.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = Math.max(data.columns.length, 1);
    cell.textContent = "This table has no records.";
    row.appendChild(cell);
    databaseRecordBody.appendChild(row);
    return;
  }

  data.records.forEach(record => {
    const row = document.createElement("tr");
    data.columns.forEach(column => {
      const cell = document.createElement("td");
      const value = record[column];
      cell.textContent = value === null ? "NULL" : String(value);
      if (value === null) cell.className = "null-value";
      row.appendChild(cell);
    });
    databaseRecordBody.appendChild(row);
  });
}

async function loadDatabaseRecords(tableName) {
  selectedDatabaseTable = tableName;
  databaseError.textContent = "";
  databaseTableButtons.querySelectorAll("button").forEach(button => {
    button.classList.toggle("is-active", button.dataset.openTable === tableName);
  });
  databaseRecordHead.replaceChildren();
  databaseRecordBody.innerHTML = '<tr class="empty-row"><td>Loading records…</td></tr>';
  recordCount.textContent = "Loading…";
  try {
    const data = await developerApiJson(
      `/developer/database/tables/${encodeURIComponent(tableName)}/records?limit=100`
    );
    renderDatabaseRecords(data);
  } catch (error) {
    databaseRecordBody.innerHTML = '<tr class="empty-row"><td>Unable to load records.</td></tr>';
    databaseError.textContent = error.message;
    recordCount.textContent = "Unavailable";
  }
}

async function loadDatabaseSchema(force = false) {
  if ((databaseSchema && !force) || databaseLoading) return;
  databaseLoading = true;
  databaseConnection.textContent = "Connecting…";
  databaseConnection.className = "";
  databaseError.textContent = "";
  try {
    // First verify that localhost can reach the currently configured Colab tunnel.
    // This makes a stale trycloudflare URL obvious instead of reporting it as a SQLite failure.
    await monitorBookingApiJson("/health");
    databaseSchema = await developerApiJson("/developer/database/schema");
    renderDatabaseSchema(databaseSchema);
    if (force && selectedDatabaseTable) await loadDatabaseRecords(selectedDatabaseTable);
  } catch (error) {
    databaseConnection.textContent = "Unavailable";
    databaseConnection.className = "database-unavailable";
    databasePath.textContent = REMOTE_DATABASE_API_BASE
      ? `Configured API: ${REMOTE_DATABASE_API_BASE}`
      : "Booking API URL is not configured.";
    databaseError.textContent = error.message;
  } finally {
    databaseLoading = false;
  }
}

function switchMonitorTab(tabName) {
  const showDatabase = tabName === "database";
  conversationMonitor.hidden = showDatabase;
  databaseMonitor.hidden = !showDatabase;
  monitorTabs.forEach(button => {
    const active = button.dataset.monitorTab === tabName;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  if (showDatabase) {
    loadDatabaseSchema();
    loadDeveloperAppointments();
    if (doctorLeaveDoctor && doctorLeaveDoctor.options.length <= 1) loadDoctorLeaveDoctors();
  }
}

monitorTabs.forEach(button => {
  button.addEventListener("click", () => switchMonitorTab(button.dataset.monitorTab));
});

databaseRefreshButton?.addEventListener("click", () => loadDatabaseSchema(true));
appointmentRefresh?.addEventListener("click", () => loadDeveloperAppointments(true));
appointmentSelector?.addEventListener("change", renderSelectedAppointment);
appointmentComplete?.addEventListener("click", async () => {
  const appointmentId = appointmentSelector?.value;
  if (!appointmentId) return;
  if (!window.confirm(`Mark appointment ${appointmentId} as COMPLETED?`)) return;
  appointmentComplete.disabled = true;
  appointmentCompleteStatus.textContent = `Completing ${appointmentId}…`;
  try {
    await developerApiJson(
      `/developer/appointments/${encodeURIComponent(appointmentId)}/complete`,
      { method: "POST", body: JSON.stringify({}) }
    );
    appointmentCompleteStatus.textContent = `${appointmentId} marked as COMPLETED.`;
    await loadDeveloperAppointments(true);
    await loadDatabaseSchema(true);
    if (selectedDatabaseTable === "appointment") await loadDatabaseRecords("appointment");
  } catch (error) {
    appointmentCompleteStatus.textContent = error.message;
    appointmentComplete.disabled = false;
  }
});

document.addEventListener("click", event => {
  const button = event.target.closest("[data-open-table]");
  if (button) loadDatabaseRecords(button.dataset.openTable);
});

doctorLeaveToggle?.addEventListener("click", async () => {
  const opening = Boolean(doctorLeaveForm?.hidden);
  if (doctorLeaveForm) doctorLeaveForm.hidden = !opening;
  doctorLeaveToggle.setAttribute("aria-expanded", String(opening));
  doctorLeaveToggle.textContent = opening ? "Close leave form" : "Doctor take leave";
  if (opening) {
    if (doctorLeaveDate && !doctorLeaveDate.value) doctorLeaveDate.value = todayIsoForDeveloper();
    if (doctorLeaveDate) doctorLeaveDate.min = todayIsoForDeveloper();
    await loadDoctorLeaveDoctors();
  }
});

doctorLeaveForm?.addEventListener("submit", async event => {
  event.preventDefault();
  const dId = String(doctorLeaveDoctor?.value || "").trim();
  const leaveDate = String(doctorLeaveDate?.value || "").trim();
  const leaveType = String(doctorLeaveType?.value || "").trim().toUpperCase();
  const reason = String(doctorLeaveReason?.value || "").trim();
  if (!dId || !leaveDate || !["ANNUAL", "MEDICAL", "EMERGENCY", "OTHER"].includes(leaveType)) {
    doctorLeaveStatus.textContent = "Select a doctor, date and leave type.";
    return;
  }
  doctorLeaveConfirm.disabled = true;
  doctorLeaveStatus.textContent = "Saving doctor leave…";
  try {
    const data = await developerApiJson("/developer/database/doctor-leave", {
      method: "POST",
      body: JSON.stringify({ d_id: dId, leave_date: leaveDate, leave_type: leaveType, reason: reason || null })
    });
    const doctorName = data.doctor?.d_name || dId;
    doctorLeaveStatus.textContent = data.warning
      ? `${doctorName}: ${leaveType} leave saved for ${leaveDate}. Warning: ${data.warning}`
      : `${doctorName}: ${leaveType} leave saved for ${leaveDate}.`;
    await loadDatabaseSchema(true);
    if (selectedDatabaseTable === "doctor_leave") await loadDatabaseRecords("doctor_leave");
  } catch (error) {
    doctorLeaveStatus.textContent = error.message;
  } finally {
    doctorLeaveConfirm.disabled = false;
  }
});

window.addEventListener("resize", () => requestAnimationFrame(drawRelationshipLines));
