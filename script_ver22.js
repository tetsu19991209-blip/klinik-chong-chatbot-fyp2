// Klinik Chong Chatbot Interface - Version 22 with Emergency Safety Override
// Sequence Text Processing Layer + dictionary-based Language Detection & Normalize Layer.
// mBERT token classification and Pinyin2Hanzi HMM conversion are connected through the Version 22 model API.

const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messagesContainer = document.querySelector(".messages");
const emojiPicker = document.querySelector(".emoji-picker");
const emojiButtons = document.querySelectorAll(".emoji-grid button");
const emojiCloseButton = document.getElementById("emoji-close-button");
const chatbotView = document.getElementById("chatbot-view");
const developerView = document.getElementById("developer-view");
const developerViewButton = document.getElementById("developer-view-button");
const footerDeveloperButton = document.getElementById("footer-developer-button");
const footerLiveClock = document.getElementById("footer-live-clock");
const developerBackButton = document.getElementById("developer-back-button");
const developerMonitorFrame = document.getElementById("developer-monitor-frame");

const BOT_REPLY_DELAY = 2000;
const SOURCE_TEXT_DATA = window.KLINIK_CHONG_TEXT_DATA || {
    emoticons: [],
    resourcePaths: {}
};

let REPLY_DATA = window.KLINIK_CHONG_REPLY_DATA || {
    greeting_reply: { type: "greeting_reply", language: "bilingual", messages: [] },
    greeting_reply2: { type: "greeting_reply2", zh: [], ms: [] },
    reject_reply: { type: "reject_reply", zh: [], ms: [] },
    ask_preference_reply: { type: "ask_preference_reply", zh: [], ms: [] },
    preference_confirmation_reply: { type: "preference_confirmation_reply", zh: [], ms: [] },
    greeting_triggers: { zh: [], ms: [] }
};

const BILINGUAL_LANGUAGE_PREFERENCE_TEXT =
    "Maaf, saya hanya dapat memahami Bahasa Melayu dan Bahasa Cina 🙏.\n" +
    "抱歉，我只能理解马来文和中文 🙏。\n\n" +
    "Sila pilih bahasa pilihan anda / 请选择您的偏好语言：\n" +
    "0 = Bahasa Melayu / 马来文\n" +
    "1 = Bahasa Cina / 中文";

const MONITOR_STORAGE_KEY = "klinik_chong_monitor_ver22";

const RESOURCE_PATHS = {
    emoticons: SOURCE_TEXT_DATA.resourcePaths?.emoticons || "data/emoticons.json",
    medicalTerms: SOURCE_TEXT_DATA.resourcePaths?.medicalTerms || "data/medical_terms_wordlist.json",
    malayNormalizer: SOURCE_TEXT_DATA.resourcePaths?.malayNormalizer || "data/asrafulsyifaa_malay_normalizer.json",
    pinyinList: SOURCE_TEXT_DATA.resourcePaths?.pinyinList || "data/guoyunhe_pinyin_list.json",
    malayDictionary: SOURCE_TEXT_DATA.resourcePaths?.malayDictionary || "data/fakhrullah_malay_dictionary.dic"
};

const MODEL_API_CONFIG = window.KLINIK_CHONG_MODEL_CONFIG || {};
const MODEL_API_BASE = String(MODEL_API_CONFIG.modelApiBase || MODEL_API_CONFIG.apiBase || "").replace(/\/+$/u, "");
// V22: browser booking requests stay same-origin and are forwarded by
// model_server_ver22.py to the current Colab/Cloudflare API. This removes
// browser CORS problems and lets --booking-api-base override a new tunnel URL.
const BOOKING_API_BASE = "/api/booking-proxy";
const MODEL_ENDPOINTS = {
    health: MODEL_API_CONFIG.endpoints?.health || "/api/health",
    languageDetect: MODEL_API_CONFIG.endpoints?.languageDetect || "/api/language-detect",
    pinyinToHanzi: MODEL_API_CONFIG.endpoints?.pinyinToHanzi || "/api/pinyin-to-hanzi",
    intentionClassify: MODEL_API_CONFIG.endpoints?.intentionClassify || "/api/intention-classify",
    ragQuery: MODEL_API_CONFIG.endpoints?.ragQuery || "/api/rag-query",
    conversationState: MODEL_API_CONFIG.endpoints?.conversationState || "/api/conversation-state",
    emotionAdjust: MODEL_API_CONFIG.endpoints?.emotionAdjust || "/api/emotion-adjust",
    clinicalExtract: MODEL_API_CONFIG.endpoints?.clinicalExtract || "/api/clinical/extract",
    clinicalNote: MODEL_API_CONFIG.endpoints?.clinicalNote || "/api/clinical/note",
    clinicalRating: MODEL_API_CONFIG.endpoints?.clinicalRating || "/api/clinical/rating",
    emergencyDetect: "/api/emergency/detect",
    emergencyCalls: "/api/emergency/calls"
};
const BOOKING_ENDPOINTS = {
    health: MODEL_API_CONFIG.bookingEndpoints?.health || "/health",
    doctors: MODEL_API_CONFIG.bookingEndpoints?.doctors || "/doctors",
    availableSlots: MODEL_API_CONFIG.bookingEndpoints?.availableSlots || "/available-slots",
    patients: MODEL_API_CONFIG.bookingEndpoints?.patients || "/patients",
    appointments: MODEL_API_CONFIG.bookingEndpoints?.appointments || "/appointments",
    completeRebooking: MODEL_API_CONFIG.bookingEndpoints?.completeRebooking || "/appointments/complete-rebooking"
};
const MODEL_REQUEST_TIMEOUT_MS = Number(MODEL_API_CONFIG.requestTimeoutMs) || 300000;
const BOOKING_REQUEST_TIMEOUT_MS = Number(MODEL_API_CONFIG.bookingRequestTimeoutMs) || 30000;
const MODEL_RUNTIME = {
    backendReachable: false,
    healthChecked: false,
    runtimeMode: "unknown",
    mockMode: false,
    warning: null,
    mBert: {
        configured: true,
        loaded: false,
        modelName: MODEL_API_CONFIG.mBertName || "mBERT",
        modelPath: null,
        lastError: null
    },
    hmm: {
        configured: true,
        loaded: false,
        modelName: MODEL_API_CONFIG.hmmName || "Pinyin2Hanzi HMM",
        modelPath: null,
        lastError: null
    }
};

const LANGUAGE_RESOURCES = {
    ready: false,
    loadErrors: [],
    loadWarnings: [],
    sources: {},
    medicalAliasMap: new Map(),
    medicalPhraseMaxWords: 1,
    malayNormalizerMap: new Map(),
    malayNormalizerMaxWords: 1,
    pinyinSet: new Set(),
    malayDictionarySet: new Set(),
    counts: {
        emoticons: 0,
        medicalEntries: 0,
        medicalAliases: 0,
        malayNormalizer: 0,
        pinyinSyllables: 0,
        malayDictionary: 0
    }
};

let replyConversation = null; // current route: "malay" | "chinese" | null
let explicitLanguagePreference = null; // set only after the user selects 0 or 1
let awaitingLanguagePreference = false;
let awaitingAppointmentId = false;
let pendingCancellationId = null;
let awaitingCancellationIc = false;
let awaitingCancelConfirmation = false;
let pendingCancellationIc = null;
let cancellationMode = "cancel";
let pendingRebooking = null;
let lastBookingContext = null;
let recentCancelledAppointment = null; // V22 compatibility state
let awaitingCheckBookingIc = false;
const CONVERSATION_STATE_KEY = "klinik_chong_conversation_v22";
let conversationState = loadConversationState();

// V22 clinical conversation state. It persists across messages in the current browser session.
let clinicalState = {
    symptoms: [],
    duration: null,
    severity: null,
    descriptions: [],
    clinicalNote: null,
    rating: null
};
let lastClinicalAction = null;
let pendingBookingAfterClinical = false;
let awaitingBookingConfirmation = false;
let ratingOffered = false;
let ratingSubmitted = false;
let activeEmergencyCallSid = null;
let emergencyCallPollTimer = null;

const CHATBOT_EMOJI_MAP = new Map([
    ["😊", "happy"],
    ["😢", "sad"],
    ["😠", "angry"],
    ["🚨", "emergency"],
    ["😨", "afraid"],
    ["🤢", "nauseous"],
    ["❓", "confused"],
    ["👋", "wave"],
    ["🙂", "smile"],
    ["🙏", "thankful"],
    ["😅", "embarrassed"],
    ["❤️", "love"],
    ["❤", "love"],
    ["😂", "laughing"],
    ["😮", "surprised"],
    ["😴", "sleepy"],
    ["🤬", "swearing"],
    ["😳", "shy"],
    ["🤕", "headache"],
    ["🌧️", "depressed"],
    ["🌧", "depressed"]
]);

const COMMON_LATIN_BIGRAMS = new Set(
    `th he in er an re on at en nd ti es or te of ed is it al ar st to nt ng se ha as ou io le ve co me de hi ri ro ic ne ea ra ce li ch ll be ma si om ur ca el ta la ns di fo ho pe ec pr no us ac ot il tr ly nc et ut ss so rs un lo wa ke sa ya da ba ka pa ga ja na mi mu ku bu tu ai ia ak am ap em im ul ah ih uh ny sy kh sh ki it ay we mo ni pi sp uk gr br dr fr kr cl bl gl pl sw tw aw ow oi ei ue`.split(/\s+/u)
);

const PINYIN_PRIORITY_WORDS = new Set(
    `wo ni ta de shi bu le zai ren you zhe ge men lai shang da wei he guo dao shuo yao jiu hen hao ma ne ba bei mei xiang zenme shenme`.split(/\s+/u)
);

const FORCED_MBERT_AMBIGUOUS_WORDS = new Set(
    `can ke kan`.split(/\s+/u)
);

function showDeveloperView() {
    if (!chatbotView || !developerView) return;
    chatbotView.hidden = true;
    developerView.hidden = false;
    document.body.classList.add("developer-view-active");
    developerMonitorFrame?.contentWindow?.postMessage({ type: "refresh-klinik-monitor" }, "*");
}

function refreshDeveloperDatabase() {
    developerMonitorFrame?.contentWindow?.postMessage({ type: "refresh-klinik-database" }, "*");
}

function showChatbotView() {
    if (!chatbotView || !developerView) return;
    developerView.hidden = true;
    chatbotView.hidden = false;
    document.body.classList.remove("developer-view-active");
    messageInput?.focus();
}

function getCurrentTime() {
    return new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });
}

function scrollToLatestMessage() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function createUserMessage(message) {
    const messageRow = document.createElement("div");
    messageRow.classList.add("message-row", "user-row");
    messageRow.innerHTML = `
      <div>
        <div class="bubble user-bubble"></div>
        <time>${getCurrentTime()}</time>
      </div>
    `;
    messageRow.querySelector(".user-bubble").textContent = message;
    messagesContainer.appendChild(messageRow);
    scrollToLatestMessage();
}

function createBotMessage(message) {
    const messageRow = document.createElement("div");
    messageRow.classList.add("message-row", "bot-row");
    messageRow.innerHTML = `
      <div class="mini-avatar"><img src="assets/chatbotpic.png" alt="Klinik Chong Chatbot"></div>
      <div>
        <div class="bubble bot-bubble"></div>
        <time>${getCurrentTime()}</time>
      </div>
    `;
    messageRow.querySelector(".bot-bubble").textContent = message;
    messagesContainer.appendChild(messageRow);
    scrollToLatestMessage();
}

function createProcessingResultMessage(result, reply) {
    const messageRow = document.createElement("div");
    messageRow.classList.add("message-row", "bot-row");

    const bubbleClass = result.isGibberish
        ? "bubble bot-bubble processing-result gibberish-bubble"
        : "bubble bot-bubble processing-result";

    messageRow.innerHTML = `
      <div class="mini-avatar"><img src="assets/chatbotpic.png" alt="Klinik Chong Chatbot"></div>
      <div>
        <div class="${bubbleClass}">
          <strong class="processing-title"></strong>
          <span class="processing-output"></span>
          <small class="processing-meta"></small>
        </div>
        <time>${getCurrentTime()}</time>
      </div>
    `;

    const title = messageRow.querySelector(".processing-title");
    const output = messageRow.querySelector(".processing-output");
    const meta = messageRow.querySelector(".processing-meta");

    if (result.isGibberish) {
        title.textContent = "Gibberish detected";
        output.textContent = "<gibberish>";
        meta.textContent = result.gibberishReasons.length
            ? `Triggered rules: ${result.gibberishReasons.join(" · ")}`
            : "The input did not contain enough meaningful text.";
    } else {
        title.textContent = "Language Detection & Normalize Result";
        output.textContent = `Normalized input:\n${result.normalizedText}`;
        meta.textContent = `Reply route: ${reply.route || result.languageSummary?.reply_route || "pending"} · reply_conversation: ${reply.replyConversation || "pending"}`;
    }

    messagesContainer.appendChild(messageRow);
    scrollToLatestMessage();
}

function showTypingIndicator() {
    const typingRow = document.createElement("div");
    typingRow.classList.add("message-row", "bot-row", "typing-row");
    typingRow.setAttribute("role", "status");
    typingRow.setAttribute("aria-label", "Klinik Chong Assistant is typing");
    typingRow.innerHTML = `
      <div class="mini-avatar"><img src="assets/chatbotpic.png" alt="Klinik Chong Chatbot"></div>
      <div class="bubble bot-bubble typing-bubble" aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
    `;
    messagesContainer.appendChild(typingRow);
    scrollToLatestMessage();
    return typingRow;
}

function runAfterTyping(callback, delay = BOT_REPLY_DELAY) {
    const typingIndicator = showTypingIndicator();
    window.setTimeout(() => {
        typingIndicator.remove();
        callback();
    }, delay);
}

function replyWithTyping(message, delay = BOT_REPLY_DELAY) {
    runAfterTyping(() => createBotMessage(message), delay);
}

async function fetchJson(path) {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
}

async function fetchText(path) {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.text();
}


function buildModelApiUrl(endpoint) {
    const path = String(endpoint || "");
    if (/^https?:\/\//iu.test(path)) return path;
    return `${MODEL_API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

async function fetchModelApi(endpoint, options = {}) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(
        () => controller.abort(),
        MODEL_REQUEST_TIMEOUT_MS
    );

    try {
        const response = await fetch(buildModelApiUrl(endpoint), {
            cache: "no-store",
            ...options,
            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            },
            signal: controller.signal
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(`Model API HTTP ${response.status}: ${detail || response.statusText}`);
        }

        return response.json();
    } finally {
        window.clearTimeout(timeoutId);
    }
}

function localEmergencyCallDetection(userInput) {
    const text = String(userInput || "").normalize("NFKC").trim().toLowerCase();
    const suicideRisk = [
        /(?:自杀|自殺|轻生|輕生|不想(?:再)?活(?:了|下去)?|活不下去|结束(?:我)?(?:的)?生命|結束(?:我)?(?:的)?生命|想去死|死掉算了|割腕|跳楼|跳樓)/iu,
        /\b(?:bunuh\s+diri|nak\s+mati|mahu\s+mati|mau\s+mati|tak\s+nak\s+hidup|tidak\s+mahu\s+hidup|tamatkan\s+hidup|cederakan\s+diri|kelar\s+pergelangan|terjun\s+(?:dari\s+)?bangunan)\b/iu,
        /\b(?:suicid(?:e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die|wanna\s+die|self[ -]?harm)\b/iu,
        /\b(?:zi\s*sha|bu\s*xiang\s*huo|xiang\s*si|jie\s*shu\s*sheng\s*ming)\b/iu
    ];
    if (suicideRisk.some(pattern => pattern.test(text))) return true;
    const negated = [
        /(?:不要|不用|不需要|无需|別|别).{0,12}(?:叫|呼叫|拨打|撥打|call)?\s*(?:救护车|救護車|999)/iu,
        /\b(?:do\s*not|don't|dont|no\s+need|tak\s+perlu|tidak\s+perlu|jangan|tak\s+payah)\b.{0,24}\b(?:call|panggil|telefon|hubungi)?\s*(?:ambulance|ambulans|999)\b/iu
    ];
    if (negated.some(pattern => pattern.test(text))) return false;
    return [
        /(?:我要|我想|需要|请|請|帮我|幫我|快|赶快|趕快|立刻|马上|馬上).{0,14}(?:叫|呼叫|打|拨打|撥打|联系|聯繫).{0,8}(?:救护车|救護車|999)/iu,
        /\b(?:call|phone|ring|contact|panggil|telefon|hubungi)\b.{0,18}\b(?:an?\s+)?(?:ambulance|ambulans|999)\b/iu,
        /\b(?:nak|mahu|mau|perlu|tolong|please|need|want)\b.{0,28}\b(?:ambulance|ambulans)\b/iu,
        /\b(?:jiao|hu\s*jiao|da|bo\s*da|call)\b.{0,18}\bjiu\s*hu\s*che\b/iu
    ].some(pattern => pattern.test(text));
}

async function detectEmergencyBeforeIntent(userInput) {
    try {
        const result = await fetchModelApi(MODEL_ENDPOINTS.emergencyDetect, {
            method: "POST",
            body: JSON.stringify({ user_input: userInput })
        });
        return Boolean(result.is_emergency);
    } catch (error) {
        console.warn("Emergency API detection unavailable; using local safety rule.", error);
        return localEmergencyCallDetection(userInput);
    }
}

function emergencyLanguage(userInput, preferredLanguage = null) {
    if (preferredLanguage === "zh" || preferredLanguage === "ms") return preferredLanguage;
    return /[\u3400-\u9fff]/u.test(String(userInput || "")) ? "zh" : "ms";
}

function setComposerDisabled(disabled) {
    messageInput.disabled = disabled;
    sendButton.disabled = disabled;
    if (!disabled) messageInput.focus();
}

function stopEmergencyCallPolling() {
    if (emergencyCallPollTimer) window.clearTimeout(emergencyCallPollTimer);
    emergencyCallPollTimer = null;
}

function showEmergencyAlertCard({ userInput = "", language = null, severity = null, onCompleted = null } = {}) {
    const replyLanguage = emergencyLanguage(userInput, language);
    const isZh = replyLanguage === "zh";
    const card = createInteractiveCard("emergency-alert-card", `
      <div class="emergency-alert-title" role="alert">🚨 ${isZh ? "侦测到紧急情况!!" : "KECEMASAN DIKESAN!!"}</div>
      <p>${isZh
        ? "如有生命危险、呼吸困难或严重受伤，请立即拨打 999。也可通知 Klinik Chong 紧急联系人：016-2366802。"
        : "Jika nyawa terancam, sukar bernafas atau cedera parah, hubungi 999 segera. Anda juga boleh memaklumkan kontak kecemasan Klinik Chong: 016-2366802."}</p>
      ${severity !== null ? `<p class="emergency-severity-note">${isZh ? `您选择的严重程度为 ${severity}/10。9–10 不建议继续普通 chatbot 咨询。` : `Tahap keterukan yang dipilih ialah ${severity}/10. Untuk tahap 9–10, jangan teruskan konsultasi chatbot biasa.`}</p>` : ""}
      <div class="emergency-actions">
        <a class="emergency-999-button" href="tel:999">${isZh ? "立即拨打 999" : "Hubungi 999 sekarang"}</a>
        <button class="emergency-clinic-call" type="button">${isZh ? "通知 Klinik Chong（016-2366802）" : "Maklumkan Klinik Chong (016-2366802)"}</button>
        <button class="emergency-cancel-call" type="button" disabled>${isZh ? "取消拨打" : "Batalkan panggilan"}</button>
      </div>
      <small class="emergency-call-status" aria-live="polite">${isZh ? "尚未拨打诊所联系人。" : "Kontak klinik belum dipanggil."}</small>`);

    const callButton = card.querySelector(".emergency-clinic-call");
    const cancelButton = card.querySelector(".emergency-cancel-call");
    const status = card.querySelector(".emergency-call-status");
    let completedHandled = false;

    const finishCall = async callStatus => {
        if (completedHandled) return;
        completedHandled = true;
        stopEmergencyCallPolling();
        activeEmergencyCallSid = null;
        callButton.disabled = true;
        cancelButton.disabled = true;
        setComposerDisabled(false);
        card.classList.add("emergency-call-finished");
        if (callStatus === "completed") {
            const message = isZh
                ? "通话结束，您的请求已告知Klinik Chong。"
                : "Panggilan telah tamat. Permintaan anda telah dimaklumkan kepada Klinik Chong.";
            status.textContent = message;
            createBotMessage(message);
            if (typeof onCompleted === "function") {
                window.setTimeout(() => onCompleted(), 250);
            }
        } else if (callStatus === "canceled") {
            status.textContent = isZh
                ? "拨打已取消。如情况严重，请立即拨打 999。"
                : "Panggilan dibatalkan. Jika keadaan serius, hubungi 999 segera.";
        } else {
            status.textContent = isZh
                ? `无法接通诊所联系人（${callStatus}）。如情况严重，请立即拨打 999。`
                : `Kontak klinik tidak dapat dihubungi (${callStatus}). Jika serius, hubungi 999 segera.`;
        }
    };

    const pollCallStatus = async () => {
        if (!activeEmergencyCallSid || completedHandled) return;
        try {
            const result = await fetchModelApi(`${MODEL_ENDPOINTS.emergencyCalls}/${activeEmergencyCallSid}`, { method: "GET" });
            const callStatus = String(result.status || "unknown");
            status.textContent = isZh ? `诊所通知通话状态：${callStatus}` : `Status panggilan klinik: ${callStatus}`;
            if (["completed", "canceled", "busy", "failed", "no-answer"].includes(callStatus)) {
                await finishCall(callStatus);
                return;
            }
        } catch (error) {
            console.error("Unable to poll emergency call status:", error);
        }
        emergencyCallPollTimer = window.setTimeout(pollCallStatus, 2000);
    };

    callButton.addEventListener("click", async () => {
        const confirmed = window.confirm(isZh
            ? "此功能仅供真实紧急情况使用。诈骗、虚假紧急通报或骚扰拨打可能需要承担法律责任。\n\n确定要通知 Klinik Chong 紧急联系人吗？"
            : "Fungsi ini hanya untuk kecemasan sebenar. Panggilan bagi penipuan, laporan palsu atau gangguan boleh mengakibatkan tindakan undang-undang.\n\nAdakah anda pasti mahu memaklumkan kontak kecemasan Klinik Chong?");
        if (!confirmed) {
            status.textContent = isZh ? "您已取消本次拨打。" : "Anda membatalkan panggilan ini.";
            return;
        }
        callButton.disabled = true;
        cancelButton.disabled = true;
        setComposerDisabled(true);
        status.textContent = isZh ? "正在拨打诊所紧急联系人…" : "Sedang menghubungi kontak kecemasan klinik…";
        try {
            const result = await fetchModelApi(MODEL_ENDPOINTS.emergencyCalls, {
                method: "POST",
                body: JSON.stringify({
                    trigger: severity !== null ? "severity_9_10" : "explicit_ambulance_request",
                    severity
                })
            });
            activeEmergencyCallSid = result.call_sid;
            cancelButton.disabled = false;
            status.textContent = isZh ? "电话正在排队或响铃。" : "Panggilan sedang beratur atau berdering.";
            pollCallStatus();
        } catch (error) {
            console.error("Emergency call failed:", error);
            setComposerDisabled(false);
            callButton.disabled = false;
            status.textContent = isZh
                ? "诊所通知拨打失败。如情况严重，请立即拨打 999。"
                : "Panggilan ke klinik gagal. Jika serius, hubungi 999 segera.";
        }
    });

    cancelButton.addEventListener("click", async () => {
        if (!activeEmergencyCallSid) return;
        cancelButton.disabled = true;
        status.textContent = isZh ? "正在取消拨打…" : "Sedang membatalkan panggilan…";
        try {
            await fetchModelApi(`${MODEL_ENDPOINTS.emergencyCalls}/${activeEmergencyCallSid}/cancel`, {
                method: "POST",
                body: JSON.stringify({})
            });
            await finishCall("canceled");
        } catch (error) {
            console.error("Unable to cancel emergency call:", error);
            cancelButton.disabled = false;
            status.textContent = isZh ? "取消失败，请再试一次。" : "Pembatalan gagal. Sila cuba lagi.";
        }
    });

    return card;
}

async function checkModelBackend() {
    try {
        const health = await fetchModelApi(MODEL_ENDPOINTS.health, { method: "GET" });
        MODEL_RUNTIME.backendReachable = true;
        MODEL_RUNTIME.healthChecked = true;
        MODEL_RUNTIME.runtimeMode = String(health.runtime_mode || health.models?.runtime_mode || "unknown");
        MODEL_RUNTIME.mockMode = Boolean(health.mock_mode ?? health.models?.mock_mode ?? false);
        MODEL_RUNTIME.warning = health.warning || null;

        const mBert = health.models?.mbert || {};
        MODEL_RUNTIME.mBert.loaded = Boolean(mBert.loaded);
        MODEL_RUNTIME.mBert.modelName = mBert.model_name || MODEL_RUNTIME.mBert.modelName;
        MODEL_RUNTIME.mBert.modelPath = mBert.path || null;
        MODEL_RUNTIME.mBert.lastError = mBert.error || null;

        const hmm = health.models?.pinyin2hanzi_hmm || {};
        MODEL_RUNTIME.hmm.loaded = Boolean(hmm.loaded);
        MODEL_RUNTIME.hmm.modelName = hmm.model_name || MODEL_RUNTIME.hmm.modelName;
        MODEL_RUNTIME.hmm.modelPath = hmm.path || null;
        MODEL_RUNTIME.hmm.lastError = hmm.error || null;

        return health;
    } catch (error) {
        MODEL_RUNTIME.backendReachable = false;
        MODEL_RUNTIME.healthChecked = true;
        MODEL_RUNTIME.runtimeMode = "unreachable";
        MODEL_RUNTIME.mBert.lastError = error.message;
        MODEL_RUNTIME.hmm.lastError = error.message;
        return null;
    }
}

async function requestMBertPredictions(tokens, sentence) {
    if (!Array.isArray(tokens) || tokens.length === 0) return [];

    const response = await fetchModelApi(MODEL_ENDPOINTS.languageDetect, {
        method: "POST",
        body: JSON.stringify({
            tokens,
            sentence: String(sentence || "")
        })
    });

    MODEL_RUNTIME.backendReachable = true;
    MODEL_RUNTIME.runtimeMode = String(response.runtime_mode || MODEL_RUNTIME.runtimeMode || "unknown");
    MODEL_RUNTIME.mockMode = MODEL_RUNTIME.runtimeMode === "mock";
    MODEL_RUNTIME.mBert.loaded = !MODEL_RUNTIME.mockMode;
    MODEL_RUNTIME.mBert.modelName = response.model_name || MODEL_RUNTIME.mBert.modelName;
    MODEL_RUNTIME.mBert.modelPath = response.model_path || MODEL_RUNTIME.mBert.modelPath;
    MODEL_RUNTIME.mBert.lastError = null;

    return Array.isArray(response.predictions) ? response.predictions : [];
}

async function requestHmmConversions(pinyinTokens) {
    if (!Array.isArray(pinyinTokens) || pinyinTokens.length === 0) return [];

    const response = await fetchModelApi(MODEL_ENDPOINTS.pinyinToHanzi, {
        method: "POST",
        body: JSON.stringify({ inputs: pinyinTokens })
    });

    MODEL_RUNTIME.backendReachable = true;
    MODEL_RUNTIME.runtimeMode = String(response.runtime_mode || MODEL_RUNTIME.runtimeMode || "unknown");
    MODEL_RUNTIME.mockMode = MODEL_RUNTIME.runtimeMode === "mock";
    MODEL_RUNTIME.hmm.loaded = !MODEL_RUNTIME.mockMode;
    MODEL_RUNTIME.hmm.modelName = response.model_name || MODEL_RUNTIME.hmm.modelName;
    MODEL_RUNTIME.hmm.modelPath = response.model_path || MODEL_RUNTIME.hmm.modelPath;
    MODEL_RUNTIME.hmm.lastError = null;

    return Array.isArray(response.conversions) ? response.conversions : [];
}

function modelRuntimeSnapshot() {
    return {
        backendReachable: MODEL_RUNTIME.backendReachable,
        healthChecked: MODEL_RUNTIME.healthChecked,
        runtimeMode: MODEL_RUNTIME.runtimeMode,
        mockMode: MODEL_RUNTIME.mockMode,
        warning: MODEL_RUNTIME.warning,
        mBert: { ...MODEL_RUNTIME.mBert },
        pinyin2HanziHmm: { ...MODEL_RUNTIME.hmm }
    };
}

function containsHanzi(text) {
    return /\p{Script=Han}/u.test(String(text || ""));
}

function normalizeLookupKey(value) {
    const clean = String(value || "").trim().replace(/\s+/gu, " ");
    return containsHanzi(clean) ? clean : clean.toLocaleLowerCase("en");
}

function addMedicalAlias(alias, sourceLanguage, item) {
    const key = normalizeLookupKey(alias);
    if (!key) return;

    let entry = LANGUAGE_RESOURCES.medicalAliasMap.get(key);
    if (!entry) {
        entry = {
            key,
            item,
            sourceLanguages: new Set()
        };
        LANGUAGE_RESOURCES.medicalAliasMap.set(key, entry);
    }

    entry.sourceLanguages.add(sourceLanguage);
    const wordCount = key.split(/\s+/u).filter(Boolean).length;
    LANGUAGE_RESOURCES.medicalPhraseMaxWords = Math.max(
        LANGUAGE_RESOURCES.medicalPhraseMaxWords,
        Math.min(wordCount, 8)
    );
}

function buildMedicalResources(data) {
    const terms = Array.isArray(data?.terms) ? data.terms : [];
    terms.forEach((item) => {
        addMedicalAlias(item.english, "other", item);
        addMedicalAlias(item.chinese, "hanzi", item);
        addMedicalAlias(item.malay, "malay", item);
        (Array.isArray(item.aliases) ? item.aliases : []).forEach((alias) => {
            const language = containsHanzi(alias) ? "hanzi" : "other";
            addMedicalAlias(alias, language, item);
        });
    });

    LANGUAGE_RESOURCES.counts.medicalEntries = terms.length;
    LANGUAGE_RESOURCES.counts.medicalAliases = LANGUAGE_RESOURCES.medicalAliasMap.size;
}

function buildMalayNormalizer(data) {
    Object.entries(data || {}).forEach(([informal, formal]) => {
        const key = normalizeLookupKey(informal);
        const value = String(formal || "").trim();
        if (!key || !value) return;
        LANGUAGE_RESOURCES.malayNormalizerMap.set(key, value);
        LANGUAGE_RESOURCES.malayNormalizerMaxWords = Math.max(
            LANGUAGE_RESOURCES.malayNormalizerMaxWords,
            Math.min(key.split(/\s+/u).length, 6)
        );
    });
    LANGUAGE_RESOURCES.counts.malayNormalizer = LANGUAGE_RESOURCES.malayNormalizerMap.size;
}

function buildPinyinResources(data) {
    Object.keys(data || {}).forEach((syllable) => {
        const key = normalizeLookupKey(syllable);
        if (key) LANGUAGE_RESOURCES.pinyinSet.add(key);
    });
    LANGUAGE_RESOURCES.counts.pinyinSyllables = LANGUAGE_RESOURCES.pinyinSet.size;
}

function buildMalayDictionary(text) {
    String(text || "").split(/\r?\n/u).forEach((line, index) => {
        const clean = line.trim();
        if (!clean || (index === 0 && /^\d+$/u.test(clean))) return;
        const word = clean.split("/")[0].trim().toLocaleLowerCase("en");
        if (word) LANGUAGE_RESOURCES.malayDictionarySet.add(word);
    });
    LANGUAGE_RESOURCES.counts.malayDictionary = LANGUAGE_RESOURCES.malayDictionarySet.size;
}

function resetLanguageResources() {
    LANGUAGE_RESOURCES.ready = false;
    LANGUAGE_RESOURCES.loadErrors = [];
    LANGUAGE_RESOURCES.loadWarnings = [];
    LANGUAGE_RESOURCES.sources = {};
    LANGUAGE_RESOURCES.medicalAliasMap.clear();
    LANGUAGE_RESOURCES.medicalPhraseMaxWords = 1;
    LANGUAGE_RESOURCES.malayNormalizerMap.clear();
    LANGUAGE_RESOURCES.malayNormalizerMaxWords = 1;
    LANGUAGE_RESOURCES.pinyinSet.clear();
    LANGUAGE_RESOURCES.malayDictionarySet.clear();
    Object.keys(LANGUAGE_RESOURCES.counts).forEach((key) => {
        LANGUAGE_RESOURCES.counts[key] = 0;
    });
}

function buildEmoticonResources(data) {
    const entries = Array.isArray(data?.emoticons) ? data.emoticons : [];
    SOURCE_TEXT_DATA.emoticons = entries;
    LANGUAGE_RESOURCES.counts.emoticons = entries.length;
}

async function loadOneResource(name, remoteLoader, embeddedValue, builder) {
    try {
        const remoteValue = await remoteLoader();
        builder(remoteValue);
        LANGUAGE_RESOURCES.sources[name] = "data_file";
        return;
    } catch (error) {
        const fallbackAvailable = embeddedValue !== undefined && embeddedValue !== null;
        if (fallbackAvailable) {
            builder(embeddedValue);
            LANGUAGE_RESOURCES.sources[name] = "embedded_fallback";
            LANGUAGE_RESOURCES.loadWarnings.push(`${name}: ${error.message}; embedded fallback used`);
            return;
        }
        LANGUAGE_RESOURCES.sources[name] = "failed";
        LANGUAGE_RESOURCES.loadErrors.push(`${name}: ${error.message}`);
    }
}

async function loadLanguageResources() {
    resetLanguageResources();
    const embedded = window.KLINIK_CHONG_RESOURCE_DATA || {};

    await Promise.all([
        loadOneResource(
            "emoticons",
            () => fetchJson(RESOURCE_PATHS.emoticons),
            embedded.emoticons,
            buildEmoticonResources
        ),
        loadOneResource(
            "medicalTerms",
            () => fetchJson(RESOURCE_PATHS.medicalTerms),
            embedded.medicalTerms,
            buildMedicalResources
        ),
        loadOneResource(
            "malayNormalizer",
            () => fetchJson(RESOURCE_PATHS.malayNormalizer),
            embedded.malayNormalizer,
            buildMalayNormalizer
        ),
        loadOneResource(
            "pinyinList",
            () => fetchJson(RESOURCE_PATHS.pinyinList),
            embedded.pinyinList,
            buildPinyinResources
        ),
        loadOneResource(
            "malayDictionary",
            () => fetchText(RESOURCE_PATHS.malayDictionary),
            embedded.malayDictionaryText,
            buildMalayDictionary
        )
    ]);

    LANGUAGE_RESOURCES.ready = LANGUAGE_RESOURCES.loadErrors.length === 0 &&
        LANGUAGE_RESOURCES.counts.pinyinSyllables > 0 &&
        LANGUAGE_RESOURCES.counts.malayDictionary > 0;

    if (LANGUAGE_RESOURCES.loadErrors.length) {
        console.warn("Version 22 resource loading errors:", LANGUAGE_RESOURCES.loadErrors);
    }
    if (LANGUAGE_RESOURCES.loadWarnings.length) {
        console.info("Version 22 resource fallbacks:", LANGUAGE_RESOURCES.loadWarnings);
    }

    return LANGUAGE_RESOURCES;
}

async function loadReplyData() {
    try {
        const response = await fetch("replies_ver22.json", { cache: "no-store" });
        if (response.ok) REPLY_DATA = await response.json();
    } catch (error) {
        console.info("Using embedded Version 22 reply data.");
    }
    return REPLY_DATA;
}

function randomItem(items) {
    if (!Array.isArray(items) || items.length === 0) return null;
    return items[Math.floor(Math.random() * items.length)];
}

function getTimeGreeting(language, date = new Date()) {
    const hour = date.getHours();
    if (language === "zh") {
        if (hour >= 5 && hour < 12) return "早安";
        if (hour >= 12 && hour < 18) return "午安";
        return "晚上好";
    }
    if (hour >= 5 && hour < 12) return "Selamat pagi";
    if (hour >= 12 && hour < 19) return "Selamat petang";
    return "Selamat malam";
}

function resolveReplyTemplate(text, language) {
    return String(text || "").replace(
        /\{\{time_greeting\}\}/gu,
        getTimeGreeting(language)
    );
}

function createReplyObject(replyType, language) {
    // No response language is known at this point, so this prompt must always
    // remain bilingual even if replies_ver22.json contains older monolingual text.
    if (replyType === "ask_preference_reply") {
        return {
            type: "ask_preference_reply",
            language: "bilingual",
            id: "ask_preference_bilingual_1",
            text: BILINGUAL_LANGUAGE_PREFERENCE_TEXT
        };
    }

    const replyGroup = REPLY_DATA[replyType] || {};
    const choices = replyGroup[language] || [];
    const selected = randomItem(choices);

    if (!selected) {
        return {
            type: replyType,
            language,
            id: `${replyType}_fallback`,
            text: language === "zh" ? "请重新输入。" : "Sila masukkan semula."
        };
    }

    return {
        type: replyGroup.type || replyType,
        language,
        id: selected.id,
        text: resolveReplyTemplate(selected.text, language)
    };
}

function getInitialGreetingReply() {
    const selected = randomItem(REPLY_DATA.greeting_reply?.messages || []);
    return {
        type: REPLY_DATA.greeting_reply?.type || "greeting_reply",
        language: "bilingual",
        id: selected?.id || "greeting_bilingual_fallback",
        text: selected?.text || "Hai, saya Klinik Chong Booking Chatbot! Ada apa yang boleh saya bantu?\n\n你好，我是庄诊所预约机器人！请问有什么可以帮到您？"
    };
}

function normalizeGreetingCandidate(normalizedText) {
    return String(normalizedText || "")
        .toLocaleLowerCase("en")
        .replace(/<[^>]+>/gu, " ")
        .replace(/[!?.,，。！？~～]/gu, " ")
        .replace(/\s+/gu, " ")
        .trim();
}

function detectGreetingLanguage(normalizedText) {
    const candidate = normalizeGreetingCandidate(normalizedText);
    if ((REPLY_DATA.greeting_triggers?.zh || []).includes(candidate)) return "zh";
    if ((REPLY_DATA.greeting_triggers?.ms || []).includes(candidate)) return "ms";
    if (/^(?:hi|hello)?\s*(?:你好|您好|ni\s*hao|nihao)(?:\s*(?:hi|hello))?$/iu.test(candidate)) return "zh";
    return null;
}

function detectKeywordIntent(normalizedText) {
    const text = String(normalizedText || "").toLocaleLowerCase("en").replace(/<[^>]+>/gu, " ").trim();
    const has = (pattern) => pattern.test(text);

    // V22 deterministic overrides protect short, high-precision conversational
    // requests from being misrouted by the statistical intention classifier.
    if (has(/(?:\b(?:terima\s*kasih|thank\s*you|thanks|tq|tqvm)\b|谢谢|謝謝|感谢|感謝|多谢|感恩)/iu)) return "thanks";
    const shortText = normalizeGreetingCandidate(text);
    if (detectGreetingLanguage(shortText)) return "greeting";
    if (/^(?:(?:ok|okay|好的|好啦)\s*)?(?:bye|bye bye|goodbye|jumpa lagi|selamat tinggal|再见|再見|拜拜|明天见|明天見)$/iu.test(shortText)) return "goodbye";
    if (has(/^(?:\s*(?:1|2|yes|no|ya|tidak|tak|boleh|确定|確定|确认|確認|取消)\s*)$/iu)) return "confirmation";
    if (has(/^(?:\s*(?:(?:hello|hi|hai)\s+)?(?:my\s+name\s+is|(?:you\s+can\s+)?call\s+me|nama\s+(?:saya|sy)|panggil\s+(?:saya|sy)|(?:saya|sy)\s+(?:bernama|nama))\s+[^,.!?]{1,60}|\s*(?:(?:你好|您好)\s*)?(?:我的名字是|我叫)\s*[^，。！？]{1,30})$/iu)) return "user_intro";

    // Chatbot capability and doctor-directory questions are clinic information.
    // These high-precision rules take precedence over an XLM-R unrelated label.
    if (has(/(?:\b(?:what\s+can\s+(?:you|this\s+chatbot)\s+do|what\s+are\s+your\s+(?:functions|features)|chatbot\s+(?:functions?|features?|capabilit(?:y|ies)))\b|\b(?:awak|kamu|chatbot\s+ini)\s+boleh\s+(?:buat|bantu)\s+apa\b|\bapa\s+(?:yang\s+)?boleh\s+(?:awak|kamu|chatbot\s+ini)\s+(?:buat|bantu)\b|你(?:可以|能)做什么|你有什么功能|聊天机器人(?:可以|能)做什么|机器人有什么功能)/iu)) return "ask_info";
    if (has(/(?:\b(?:how\s+many\s+doctors?|which\s+doctors?|who\s+are\s+the\s+doctors?|doctor\s+list)\b|\b(?:berapa\s+(?:orang\s+)?doktor|doktor\s+ada\s+berapa|siapa\s+(?:sahaja\s+)?doktor|senarai\s+doktor)\b|诊所(?:有)?多少(?:个|位)?医生|診所(?:有)?多少(?:個|位)?醫生|(?:有|有哪些|都有)(?:什么|什麼)?医生|(?:有|有哪些|都有)(?:什么|什麼)?醫生|医生名单|醫生名單)/iu)) return "ask_info";

    // Clinic operating-hour questions are ask_info, never check_booking.
    // Covers common Malay abbreviations and spelling variants such as
    // "sy nak tanya bila buka" and "bila klinick awak buka".
    if (has(/(?:\b(?:bila|pukul\s+berapa|jam\s+berapa)\b[^?!.]{0,55}\b(?:buka|tutup)\b|\b(?:waktu|masa|jam)\s+(?:operasi|buka|tutup)\b|\boperating\s+hours?\b|\bopening\s+hours?\b|\bwhen\s+(?:does|is|are)?\s*(?:the\s+)?(?:clinic|klinik)?\s*(?:open|close)\b|营业时间|營業時間|开放时间|開放時間|几点开门|幾點開門|几点营业|幾點營業|什么时候开|什麼時候開)/iu)) return "ask_info";
    return null;
}

async function classifyMainIntention(userInput) {
    const result = await fetchModelApi(MODEL_ENDPOINTS.intentionClassify, {
        method: "POST",
        body: JSON.stringify({ user_input: userInput })
    });
    return {
        intention: result?.uncertain ? "uncertain" : String(result?.intention || "uncertain"),
        source: "xlmr",
        confidence: Number(result?.confidence),
        probabilities: result?.probabilities || null
    };
}

function extractIntroducedName(text) {
    const value = String(text || "").normalize("NFKC").trim();
    if (isIdentityRecallQuestion(value)) return "";
    const match = value.match(
        /(?:^|[，。,.!?！？]\s*|\b(?:hello|hi|hai)\s+|(?:你好|您好)\s*)(?:i(?:'m|\s+am)|my\s+name\s+is|nama\s+(?:saya|sy)|(?:saya|sy)\s+(?:bernama|nama|ialah|adalah)|我的名字是|我叫|我是)\s*[:：]?\s*([^,.!?，。！？]{1,60})/iu
    );
    if (!match) return "";
    const name = match[1]
        .replace(/\s+(?:and|dan|then)\s+.*$/iu, "")
        .replace(/\s*(?:你呢|你叫什么|你叫什麼|awak\s+pula|kamu\s+pula|how\s+about\s+you)\s*$/iu, "")
        .replace(/\s+/gu, " ")
        .trim();
    const nonNameMeaning = /^(?:very|really|quite|so|too|tired|sick|ill|unwell|sad|angry|afraid|worried|dizzy|hungry|in\s+pain|not\s+well|a\s+patient|the\s+patient|想|要|需要|不舒服|很累|生病|病人|害怕|担心|什么|什麼|谁|誰|哪|mahu|nak|penat|sakit|sedih|marah|takut|risau|pesakit)\b/iu;
    if (!name || nonNameMeaning.test(name) || /(?:[?？]|叫什么|叫什麼|什么名字|什麼名字|\b(?:need|want|have|feel|perlukan|mahu|nak|rasa)\b)/iu.test(name)) return "";
    return name;
}

function sanitizeRememberedName(name) {
    const cleaned = String(name || "").replace(/\s+/gu, " ").trim().slice(0, 60);
    const invalid = /^(?:very|really|quite|so|too|tired|sick|ill|unwell|sad|angry|afraid|worried|dizzy|hungry|in\s+pain|not\s+well|a\s+patient|the\s+patient|很累|不舒服|生病|病人|害怕|担心|什么|什麼|谁|誰|哪|penat|sakit|sedih|marah|takut|risau|pesakit)\b/iu;
    return !cleaned || invalid.test(cleaned) || /(?:[?？]|叫什么|叫什麼|什么名字|什麼名字)/iu.test(cleaned)
        ? ""
        : cleaned;
}

function loadConversationState() {
    try {
        const saved = JSON.parse(sessionStorage.getItem(CONVERSATION_STATE_KEY) || "null");
        if (saved && typeof saved === "object") {
            const savedHistory = Array.isArray(saved.history)
                ? saved.history.filter(item => item && ["user", "assistant"].includes(item.role) && item.content).slice(-20)
                : [];
            let safeUserName = sanitizeRememberedName(saved.userName);
            if (!safeUserName) {
                for (let index = savedHistory.length - 1; index >= 0; index -= 1) {
                    if (savedHistory[index].role !== "user") continue;
                    const recoveredName = sanitizeRememberedName(
                        extractIntroducedName(savedHistory[index].content)
                    );
                    if (recoveredName) {
                        safeUserName = recoveredName;
                        break;
                    }
                }
            }
            return {
                userName: safeUserName,
                preferredLanguage: String(saved.preferredLanguage || "").trim(),
                lastIntent: String(saved.lastIntent || "").trim(),
                activeFlow: String(saved.activeFlow || "").trim(),
                doctorName: String(saved.doctorName || "").trim(),
                appointmentDate: String(saved.appointmentDate || "").trim(),
                appointmentTime: String(saved.appointmentTime || "").trim(),
                symptoms: Array.isArray(saved.symptoms) ? saved.symptoms.slice(0, 20) : [],
                duration: String(saved.duration || "").trim(),
                severity: saved.severity ?? null,
                history: savedHistory
            };
        }
    } catch (error) {
        console.warn("Unable to load conversation state:", error);
    }
    return {
        userName: "", preferredLanguage: "", lastIntent: "", activeFlow: "",
        doctorName: "", appointmentDate: "", appointmentTime: "",
        symptoms: [], duration: "", severity: null, history: []
    };
}

function saveConversationState() {
    try {
        sessionStorage.setItem(CONVERSATION_STATE_KEY, JSON.stringify(conversationState));
    } catch (error) {
        console.warn("Unable to save conversation state:", error);
    }
}

function rememberUserName(name) {
    const cleaned = sanitizeRememberedName(name);
    if (!cleaned) return "";
    conversationState.userName = cleaned;
    saveConversationState();
    return cleaned;
}

function conversationStateForApi() {
    return {
        user_name: sanitizeRememberedName(conversationState.userName) || null,
        preferred_language: conversationState.preferredLanguage || null,
        last_intent: conversationState.lastIntent || null,
        active_flow: conversationState.activeFlow || null,
        doctor_name: conversationState.doctorName || null,
        appointment_date: conversationState.appointmentDate || null,
        appointment_time: conversationState.appointmentTime || null,
        symptoms: Array.isArray(conversationState.symptoms) ? conversationState.symptoms : [],
        duration: conversationState.duration || null,
        severity: conversationState.severity ?? null
    };
}

function mergeConversationStateFromApi(value) {
    if (!value || typeof value !== "object") return;
    const textMappings = {
        preferred_language: "preferredLanguage",
        last_intent: "lastIntent", active_flow: "activeFlow",
        doctor_name: "doctorName", appointment_date: "appointmentDate",
        appointment_time: "appointmentTime", duration: "duration"
    };
    if (value.user_name !== null && value.user_name !== undefined) {
        const safeUserName = sanitizeRememberedName(value.user_name);
        if (safeUserName) conversationState.userName = safeUserName;
    }
    Object.entries(textMappings).forEach(([source, target]) => {
        if (value[source] !== null && value[source] !== undefined && String(value[source]).trim()) {
            conversationState[target] = String(value[source]).trim();
        }
    });
    if (Array.isArray(value.symptoms) && value.symptoms.length) {
        conversationState.symptoms = value.symptoms.map(item => String(item).trim()).filter(Boolean).slice(0, 20);
    }
    if (value.severity !== null && value.severity !== undefined && value.severity !== "") {
        const severity = Number(value.severity);
        if (Number.isInteger(severity) && severity >= 0 && severity <= 10) conversationState.severity = severity;
    }
    saveConversationState();
}

function appendConversationHistory(role, content) {
    const text = String(content || "").trim();
    if (!text || !["user", "assistant"].includes(role)) return;
    if (!Array.isArray(conversationState.history)) conversationState.history = [];
    conversationState.history.push({ role, content: text });
    if (conversationState.history.length > 20) conversationState.history.splice(0, conversationState.history.length - 20);
    saveConversationState();
}

function isDoctorDirectoryFollowUp(text) {
    const current = String(text || "").normalize("NFKC").trim();
    const shortFollowUp = /^(?:有谁(?:呢)?|有誰(?:呢)?|都有谁(?:呢)?|都有誰(?:呢)?|谁(?:呢)?|誰(?:呢)?|which\s+ones?|who\s+are\s+they|siapa(?:\s+sahaja)?)[?？.!。 ]*$/iu.test(current);
    if (!shortFollowUp) return false;
    return (conversationState.history || []).slice(-6).some(item =>
        /(?:医生|醫生|\bdoctors?\b|\bdoktor\b)/iu.test(String(item?.content || ""))
    );
}

async function updateConversationStateWithLlm(userInput, normalizedInput = "") {
    // Name-recall questions must not be sent to the name extractor. Check both
    // the original message and the Pinyin-to-Hanzi normalized message so a
    // correctly converted question cannot overwrite the stored user name.
    const identityInput = `${String(userInput || "")} ${String(normalizedInput || "")}`.trim();
    if (isIdentityRecallQuestion(identityInput)) return "";

    const oldName = conversationState.userName;
    try {
        const result = await fetchModelApi(MODEL_ENDPOINTS.conversationState, {
            method: "POST",
            body: JSON.stringify({
                user_input: String(userInput || "").trim(),
                conversation_state: conversationStateForApi(),
                conversation_history: conversationState.history || []
            })
        });
        mergeConversationStateFromApi(result.conversation_state);
        const newName = conversationState.userName;
        return newName && newName !== oldName ? newName : "";
    } catch (error) {
        console.warn("LLM conversation-state extraction unavailable; using local fallback:", error);
        const fallbackName = extractIntroducedName(userInput);
        return fallbackName ? rememberUserName(fallbackName) : "";
    }
}

async function applyEmotionAwareReply(reply, userInput) {
    if (!reply || !String(reply.text || "").trim() || reply.type === "processing_result") return reply;
    // Language-selection is a control message, not a normal conversational
    // answer.  It must remain bilingual and must not be rewritten by the
    // emotion layer (where "bilingual" previously fell back to Malay).
    if (reply.type === "ask_preference_reply" || reply.language === "bilingual") {
        reply.emotion = detectTurnEmotion(userInput);
        reply.emotionApplied = false;
        return reply;
    }
    if (reply.rag?.emotion_applied) {
        reply.emotion = reply.rag.emotion || "neutral";
        reply.emotionApplied = true;
        return reply;
    }
    const safetyCritical = reply.type === "emergency_severity"
        || /(?:\b999\b|bunuh\s+diri|自杀|自殺|ambulans|ambulance)/iu.test(String(reply.text || ""));
    try {
        const result = await fetchModelApi(MODEL_ENDPOINTS.emotionAdjust, {
            method: "POST",
            body: JSON.stringify({
                user_input: String(userInput || "").trim(),
                base_response: String(reply.text || "").trim(),
                response_language: clinicalLanguageName(reply.language),
                conversation_state: conversationStateForApi(),
                conversation_history: conversationState.history || [],
                safety_critical: safetyCritical
            })
        });
        if (String(result.response || "").trim()) reply.text = String(result.response).trim();
        reply.emotion = String(result.emotion || "neutral").trim().toLowerCase();
        reply.emotionApplied = true;
    } catch (error) {
        console.warn("Emotion-aware generation unavailable; preserving the original reply:", error);
        reply.emotion = detectTurnEmotion(userInput);
        reply.emotionApplied = false;
    }
    return reply;
}

function isIdentityRecallQuestion(text) {
    const value = String(text || "")
        .normalize("NFKC")
        .toLocaleLowerCase("en")
        // HMM output can contain spaces between converted Hanzi tokens.
        .replace(/([\p{Script=Han}])\s+(?=[\p{Script=Han}])/gu, "$1");
    return /(?:你(?:还|還)?(?:记得|記得|知道)(?:我是谁|我是誰|我叫什么|我叫什麼|我的名字)|我(?:叫什么|叫什麼)(?:名字)?(?:吗|嗎|呢)?|我的名字(?:是)?(?:什么|什麼|谁|誰|[?？])|who\s+am\s+i|what(?:'s|\s+is)\s+my\s+name|do\s+you\s+(?:know|remember)\s+my\s+name|siapa\s+(?:saya|sy)|(?:siapa|sapa)\s+nama\s+(?:saya|sy)|nama\s+(?:saya|sy)\s*(?:(?:siapa|sapa|apa|ialah|adalah)\b|[?？]|$)|(?:awak|kamu)?\s*(?:tahu|ingat)\s+(?:tak\s+)?nama\s+(?:saya|sy)\b|ni\s+zhi\s+dao\s+wo\s+(?:jiao\s+shen\s+me\s+ming\s+zi|shi\s+shui)\s+ma|wo\s+de\s+ming\s+zi\s+shi(?:\s+shen\s+me)?|wo\s+jiao\s+shen\s+me\s+ming\s+zi)/iu.test(value);
}

function cancellationInteractionActive() {
    return awaitingAppointmentId || awaitingCancellationIc || awaitingCancelConfirmation;
}

async function classifyStateInterruption(processingResult) {
    const combined = `${processingResult.normalizedText || ""} ${processingResult.sequenceNormalizedText || ""} ${processingResult.originalInput || ""}`.trim();
    const keywordIntent = detectKeywordIntent(combined);
    if (keywordIntent) {
        return { intention: keywordIntent, source: "state_interrupt_keyword", confidence: 1, probabilities: null };
    }
    try {
        return await classifyMainIntention(
            String(processingResult.normalizedText || processingResult.originalInput || "").trim()
        );
    } catch (error) {
        console.warn("Unable to classify cancellation-flow interruption:", error);
        return { intention: "uncertain", source: "state_interrupt_fallback", confidence: null, probabilities: null };
    }
}

function bookingLanguage(processingResult) {
    return selectAskPreferenceLanguage(processingResult.languageSummary);
}

function rememberConversationLanguage(language) {
    if (language === "zh") replyConversation = "chinese";
    if (language === "ms") replyConversation = "malay";
}

function detectTurnEmotion(text) {
    const value = String(text || "").normalize("NFKC").toLocaleLowerCase("en");
    if (/(?:生气|愤怒|氣死|什么问题|什麼問題|marah|geram|menyampah|!!+|！？)/iu.test(value)) return "angry";
    if (/(?:害怕|担心|擔心|恐惧|恐懼|takut|risau|cemas|panic)/iu.test(value)) return "afraid";
    if (/(?:难过|難過|伤心|傷心|sedih|kecewa|menangis)/iu.test(value)) return "sad";
    if (/(?:开心|開心|满意|滿意|高兴|高興|gembira|puas hati|seronok|😊|😁)/iu.test(value)) return "happy";
    if (/(?:不明白|不懂|困惑|confuse|keliru|tak faham)/iu.test(value)) return "confuse";
    return "neutral";
}

function buildBookingApiUrl(endpoint, params = null) {
    const path = String(endpoint || "");
    const baseUrl = /^https?:\/\//iu.test(path)
        ? path
        : `${BOOKING_API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
    if (!params) return baseUrl;
    const url = new URL(baseUrl, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
            url.searchParams.set(key, value);
        }
    });
    return url.toString();
}

async function bookingApiJson(endpoint, options = {}) {
    if (!BOOKING_API_BASE) {
        throw new Error("Booking API URL is not configured in model_config_ver22.js.");
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), BOOKING_REQUEST_TIMEOUT_MS);
    let response;
    try {
        response = await fetch(buildBookingApiUrl(endpoint, options.params), {
            cache: "no-store",
            method: options.method || "GET",
            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            },
            body: options.body,
            signal: controller.signal
        });
    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("Booking API timeout. Please check the Colab runtime and public URL.");
        }
        throw new Error("Booking API is offline. Please check the Colab runtime and API URL.");
    } finally {
        window.clearTimeout(timeoutId);
    }

    let data = {};
    try { data = await response.json(); } catch (_) { data = {}; }
    if (!response.ok) {
        const detail = Array.isArray(data.detail)
            ? data.detail.map(item => item.msg || String(item)).join("; ")
            : data.detail;
        const error = new Error(detail || data.message || `Booking API request failed (${response.status}).`);
        error.status = response.status;
        error.payload = data;
        throw error;
    }
    return data;
}

function normalizePatientName(value) {
    return String(value || "")
        .replace(/[^\p{L}\s]/gu, "")
        .replace(/\s+/gu, " ")
        .replace(/^\s/gu, "")
        .slice(0, 120);
}

function formatIcInput(value) {
    const digits = String(value || "").replace(/\D/gu, "").slice(0, 12);
    return [digits.slice(0, 6), digits.slice(6, 8), digits.slice(8, 12)]
        .filter(Boolean)
        .join("-");
}

function formatContactInput(value) {
    const digits = String(value || "").replace(/\D/gu, "").slice(0, 11);
    if (digits.length <= 3) return digits;
    const local = digits.slice(3);
    const firstGroupLength = digits.length === 11 ? 4 : 3;
    const firstGroup = local.slice(0, firstGroupLength);
    const secondGroup = local.slice(firstGroupLength, firstGroupLength + 4);
    return `${digits.slice(0, 3)}-${firstGroup}${secondGroup ? `-${secondGroup}` : ""}`;
}

function databaseContact(value) {
    const digits = String(value || "").replace(/\D/gu, "");
    return digits.length >= 3 ? `${digits.slice(0, 3)}-${digits.slice(3)}` : digits;
}

function databaseDob(isoDate) {
    const [year, month, day] = String(isoDate || "").split("-");
    return year && month && day ? `${day}-${month}-${year}` : "";
}

function icDatePrefixFromDob(isoDate) {
    const [year, month, day] = String(isoDate || "").split("-");
    if (!/^\d{4}$/u.test(year || "") || !/^\d{2}$/u.test(month || "") || !/^\d{2}$/u.test(day || "")) {
        return "";
    }
    return `${year.slice(-2)}${month}${day}`;
}

function icLastDigitMatchesGender(icValue, gender) {
    const digits = String(icValue || "").replace(/\D/gu, "");
    if (digits.length !== 12 || !["M", "F"].includes(gender)) return true;
    const lastDigit = Number(digits.at(-1));
    return gender === "M" ? lastDigit % 2 === 1 : lastDigit % 2 === 0;
}

function doctorGenderLabel(gender, language) {
    const normalized = String(gender || "").trim().toUpperCase();
    const isFemale = ["F", "FEMALE", "PEREMPUAN"].includes(normalized);
    const isMale = ["M", "MALE", "LELAKI"].includes(normalized);
    if (!isFemale && !isMale) return "";
    if (language === "zh") return isFemale ? "女" : "男";
    return isFemale ? "Perempuan" : "Lelaki";
}

function doctorDisplayName(doctor, language) {
    const name = String(doctor?.d_name || "").replace(/^Dr\.?\s*/iu, "").trim();
    const gender = doctorGenderLabel(doctor?.d_gender, language);
    if (!name) return "";
    if (!gender) return `Dr. ${name}`;
    return language === "zh"
        ? `Dr. ${name}（${gender}）`
        : `Dr. ${name} (${gender})`;
}

function naturalAppointmentTime(value, language) {
    const [hourText, minuteText] = String(value || "").split(":");
    const hour = Number(hourText);
    const minute = Number(minuteText || 0);
    if (!Number.isInteger(hour) || hour < 0 || hour > 23) return String(value || "");
    const displayHour = hour % 12 || 12;
    if (language === "zh") {
        const period = hour < 12 ? "早上" : hour < 18 ? "下午" : "晚上";
        return `${period}${displayHour}点${minute ? `${minute}分` : ""}`;
    }
    const period = hour < 12 ? "pagi" : hour < 14 ? "tengah hari" : hour < 19 ? "petang" : "malam";
    return `pukul ${displayHour}${minute ? `:${String(minute).padStart(2, "0")}` : ""} ${period}`;
}

function extractHolidayNameFromMessage(message) {
    const text = String(message || "").trim();
    const match = text.match(/\sfor\s+(.+?)\.?$/iu);
    return match ? match[1].trim() : "";
}

function bookingAvailabilityMessage(data, copy, language) {
    const status = String(data?.status || "");
    if (status === "CLINIC_CLOSED_HOLIDAY") {
        const holidayName = String(data?.holiday_name || "").trim() || extractHolidayNameFromMessage(data?.message);
        if (language === "zh") return holidayName ? `${copy.publicHoliday}：${holidayName}` : copy.publicHoliday;
        return holidayName ? `${copy.publicHoliday}: ${holidayName}` : copy.publicHoliday;
    }
    if (status === "DOCTOR_ON_LEAVE") {
        const doctor = String(data?.doctor || "").replace(/^Dr\.?\s*/iu, "").trim();
        const date = String(data?.date || "").trim();
        if (language === "zh") {
            return `${doctor ? `Dr. ${doctor}` : "该医生"}${date ? ` 于 ${date}` : ""} 请假（Doctor take leave）。`;
        }
        return `${doctor ? `Dr. ${doctor}` : "Doktor ini"}${date ? ` bercuti pada ${date}` : " bercuti"} (Doctor take leave).`;
    }
    if (status === "CLINIC_CLOSED" || status === "DOCTOR_OFF_DAY") return copy.restDay;
    if (status === "FULLY_BOOKED") return copy.fullyBooked;
    return data?.message || copy.fullyBooked;
}

function bookingSuccessMessage(language, result, bookedDoctorName, rebookingCompleted, rebookingWarning = "") {
    const appointmentTime = naturalAppointmentTime(result.start_time, language);
    const doctor = bookedDoctorName ? `Dr. ${bookedDoctorName}` : "—";
    const lines = language === "zh"
        ? [
            rebookingCompleted ? "重新预约成功！" : "预约成功！",
            `预约号码：${result.appointment_id}`,
            `医生：${doctor}`,
            `日期：${result.appointment_date}`,
            `时间：${appointmentTime}`,
            "请提前 15 分钟到达诊所。😊"
        ]
        : [
            rebookingCompleted ? "Penjadualan semula berjaya!" : "Tempahan anda berjaya!",
            `ID janji temu: ${result.appointment_id}`,
            `Doktor: ${doctor}`,
            `Tarikh: ${result.appointment_date}`,
            `Masa: ${appointmentTime}`,
            "Sila tiba di klinik 15 minit lebih awal. 😊"
        ];
    if (rebookingWarning) lines.push(rebookingWarning.trim());
    return lines.join("\n");
}

async function completeRebookingWithRetry(rebookingContext, newAppointmentId, patientIc, maxAttempts = 3) {
    let lastError = null;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
            const result = await bookingApiJson(BOOKING_ENDPOINTS.completeRebooking, {
                method: "POST",
                body: JSON.stringify({
                    cancelled_appointment_id: rebookingContext.oldAppointmentId,
                    new_appointment_id: newAppointmentId,
                    patient_ic: patientIc,
                    reschedule_reason: rebookingContext.reason
                })
            });
            if (result?.status !== "REBOOKING_COMPLETED") {
                throw new Error(result?.message || "Rebooking relationship was not completed.");
            }
            return result;
        } catch (error) {
            lastError = error;
            if (attempt < maxAttempts) {
                await new Promise(resolve => window.setTimeout(resolve, 250 * attempt));
            }
        }
    }
    throw lastError || new Error("Rebooking relationship could not be completed.");
}

function yesterdayIso() {
    const value = new Date();
    value.setDate(value.getDate() - 1);
    return value.toISOString().slice(0, 10);
}

function tomorrowIso() {
    const value = new Date();
    value.setDate(value.getDate() + 1);
    return value.toISOString().slice(0, 10);
}

function malaysiaDateTimeParts(date = new Date()) {
    const parts = new Intl.DateTimeFormat("en-CA", {
        timeZone: "Asia/Kuala_Lumpur",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hourCycle: "h23"
    }).formatToParts(date);
    const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
    return {
        date: `${values.year}-${values.month}-${values.day}`,
        time: `${values.hour}:${values.minute}`,
        display: `${values.day}/${values.month}/${values.year} ${values.hour}:${values.minute}:${values.second}`
    };
}

function parseSlotMinutes(startTime) {
    const raw = String(startTime || "")
        .trim()
        .toUpperCase()
        .replaceAll(".", ":");
    const match = raw.match(/^(\d{1,2}):?(\d{2})\s*(AM|PM)?$/u);
    if (!match) return null;

    let hour = Number(match[1]);
    const minute = Number(match[2]);
    const meridiem = match[3] || null;
    if (minute < 0 || minute > 59) return null;

    if (meridiem) {
        if (hour < 1 || hour > 12) return null;
        if (hour === 12) hour = 0;
        if (meridiem === "PM") hour += 12;
    } else if (hour < 0 || hour > 23) {
        return null;
    }

    return (hour * 60) + minute;
}

function isFutureMalaysiaSlot(appointmentDate, startTime) {
    const now = malaysiaDateTimeParts();
    const targetDate = String(appointmentDate || "");
    if (targetDate > now.date) return true;
    if (targetDate < now.date) return false;

    const targetMinutes = parseSlotMinutes(startTime);
    const currentMinutes = parseSlotMinutes(now.time);
    return targetMinutes !== null
        && currentMinutes !== null
        && targetMinutes > currentMinutes;
}

function updateFooterClock() {
    if (!footerLiveClock) return;
    const now = malaysiaDateTimeParts();
    footerLiveClock.dateTime = `${now.date}T${now.time}:00+08:00`;
    footerLiveClock.textContent = now.display;
}

function bookingCopy(language) {
    const isZh = language === "zh";
    return isZh ? {
        title: "预约表单",
        step1: "第一部分：病人资料",
        step2: "第二部分：预约时间",
        name: "英文名字",
        nameHint: "请填写身份证上的英文名字，只允许文字，1–120 个字符。符号会自动移除。",
        dob: "出生日期",
        gender: "性别",
        choose: "请选择",
        male: "男（M）",
        female: "女（F）",
        ic: "身份证号码（IC）",
        icHint: "只需输入 12 位数字，连字符会自动加入。",
        contact: "联络号码",
        contactHint: "只需输入数字，格式会自动调整。",
        note: "临床备注（系统生成）",
        noteHint: "根据症状、持续时间和严重程度自动生成，无法手动修改。",
        next: "下一步",
        back: "返回修改",
        appointmentId: "预约号码",
        generated: "成功提交后自动生成",
        doctorType: "医生类型",
        doctorName: "选择医生",
        chooseDoctor: "请选择医生",
        general: "全科",
        dermatology: "皮肤科",
        obgyn: "妇产科",
        pediatrics: "儿科",
        date: "可预约日期",
        dateHelp: "选择日期后才会显示可用时间。",
        slots: "可用时间",
        selectDoctorType: "请先选择医生类型。",
        loadingDoctors: "正在读取医生资料…",
        doctorsFound: count => `已找到 ${count} 位医生，请选择您想预约的医生。`,
        loadingSlots: "正在读取可用时间…",
        chooseSlot: "请选择一个时间。",
        restDay: "休息日（Rest day）",
        publicHoliday: "公共假期（Public Holiday）",
        doctorLeave: "医生请假（Doctor take leave）",
        fullyBooked: "当天已满",
        submit: "确认预约",
        submitting: "正在提交预约…",
        success: "预约成功！",
        invalidName: "姓名必须是 1–120 个文字字符，不能包含数字。",
        invalidIc: "请输入完整的 12 位 IC 数字。",
        invalidContact: "请输入 10 或 11 位马来西亚手机号码。",
        invalidDobRange: "出生日期必须是 1900 年或以后。",
        icDobMismatch: expected => `您的出生日期和 IC 前六位数不符，应该是 ${expected}。`,
        maleIcMismatch: "男性的 IC 最后一位必须是单数（1、3、5、7、9）。",
        femaleIcMismatch: "女性的 IC 最后一位必须是双数（0、2、4、6、8）。",
        rebookIc: "重新预约必须使用原预约相同的 IC。"
    } : {
        title: "Borang Janji Temu",
        step1: "Bahagian 1: Maklumat Pesakit",
        step2: "Bahagian 2: Masa Janji Temu",
        name: "Nama dalam bahasa Inggeris",
        nameHint: "Masukkan nama bahasa Inggeris seperti pada IC. Huruf sahaja, 1–120 aksara.",
        dob: "Tarikh lahir",
        gender: "Jantina",
        choose: "Sila pilih",
        male: "Lelaki (M)",
        female: "Perempuan (F)",
        ic: "Nombor IC",
        icHint: "Masukkan 12 digit sahaja. Tanda sempang ditambah secara automatik.",
        contact: "Nombor telefon",
        contactHint: "Masukkan digit sahaja. Format akan dilaraskan secara automatik.",
        note: "Nota klinikal (dijana sistem)",
        noteHint: "Dijana secara automatik daripada simptom, tempoh dan tahap keterukan. Tidak boleh diubah oleh pengguna.",
        next: "Seterusnya",
        back: "Kembali",
        appointmentId: "ID janji temu",
        generated: "Dijana selepas penghantaran berjaya",
        doctorType: "Jenis doktor",
        doctorName: "Pilih doktor",
        chooseDoctor: "Sila pilih doktor",
        general: "Doktor Am",
        dermatology: "Dermatologi",
        obgyn: "Obstetrik dan Ginekologi",
        pediatrics: "Pediatrik",
        date: "Tarikh tersedia",
        dateHelp: "Masa tersedia dipaparkan selepas tarikh dipilih.",
        slots: "Masa tersedia",
        selectDoctorType: "Sila pilih jenis doktor dahulu.",
        loadingDoctors: "Sedang memuatkan maklumat doktor…",
        doctorsFound: count => `${count} orang doktor ditemui. Sila pilih doktor pilihan anda.`,
        loadingSlots: "Sedang memuatkan masa tersedia…",
        chooseSlot: "Sila pilih satu masa.",
        restDay: "Hari rehat (Rest day)",
        publicHoliday: "Cuti umum (Public Holiday)",
        doctorLeave: "Doktor bercuti (Doctor take leave)",
        fullyBooked: "Tempahan penuh",
        submit: "Sahkan janji temu",
        submitting: "Sedang menghantar janji temu…",
        success: "Tempahan berjaya!",
        invalidName: "Nama mesti mengandungi 1–120 huruf dan tidak boleh mempunyai nombor.",
        invalidIc: "Sila masukkan 12 digit IC yang lengkap.",
        invalidContact: "Sila masukkan nombor telefon Malaysia dengan 10 atau 11 digit.",
        invalidDobRange: "Tarikh lahir mestilah pada atau selepas tahun 1900.",
        icDobMismatch: expected => `Tarikh lahir tidak sepadan dengan enam digit pertama IC. Sepatutnya ${expected}.`,
        maleIcMismatch: "Digit terakhir IC lelaki mestilah nombor ganjil (1, 3, 5, 7 atau 9).",
        femaleIcMismatch: "Digit terakhir IC perempuan mestilah nombor genap (0, 2, 4, 6 atau 8).",
        rebookIc: "Tempahan semula mesti menggunakan IC yang sama dengan janji temu asal."
    };
}

function clinicalLanguageName(language) {
    if (language === "zh") return "chinese";
    if (language === "bilingual") return "bilingual";
    return "malay";
}

function clinicalMissingFields() {
    const missing = [];
    if (!clinicalState.symptoms.length) missing.push("symptoms");
    if (!clinicalState.duration) missing.push("duration");
    if (clinicalState.severity === null || clinicalState.severity === undefined) missing.push("severity");
    return missing;
}

function clinicalIsComplete() {
    return clinicalMissingFields().length === 0;
}

function resetClinicalStateForNextBooking() {
    clinicalState = {
        symptoms: [],
        duration: null,
        severity: null,
        descriptions: [],
        clinicalNote: null,
        rating: null
    };
    lastClinicalAction = null;
    pendingBookingAfterClinical = false;
    awaitingBookingConfirmation = false;
}

function mergeClinicalExtraction(extracted, rawDescription = "") {
    const newSymptoms = Array.isArray(extracted?.symptoms) ? extracted.symptoms : [];
    newSymptoms.forEach(symptom => {
        const value = String(symptom || "").trim();
        if (!value) return;
        const exists = clinicalState.symptoms.some(item => String(item).toLocaleLowerCase("en") === value.toLocaleLowerCase("en"));
        if (!exists) clinicalState.symptoms.push(value);
    });
    if (extracted?.duration) clinicalState.duration = String(extracted.duration).trim();
    const hasSeverity = extracted?.severity !== null
        && extracted?.severity !== undefined
        && String(extracted.severity).trim() !== "";
    const severity = hasSeverity ? Number(extracted.severity) : NaN;
    if (Number.isInteger(severity) && severity >= 0 && severity <= 10) clinicalState.severity = severity;
    const detail = String(rawDescription || "").trim();
    if (detail && !clinicalState.descriptions.includes(detail)) clinicalState.descriptions.push(detail);
}

async function extractClinicalFromUser(userInput) {
    return fetchModelApi(MODEL_ENDPOINTS.clinicalExtract, {
        method: "POST",
        body: JSON.stringify({ user_input: userInput })
    });
}

function noteWithRating(note) {
    let base = String(note || "").replace(/\n?rate to clinic:\s*[1-5]\s+stars\s*$/iu, "").trim();
    if (clinicalState.rating) base = `${base}${base ? "\n" : ""}rate to clinic: ${clinicalState.rating} stars`;
    return base;
}

async function refreshClinicalNote() {
    if (!clinicalIsComplete()) {
        clinicalState.clinicalNote = null;
        return null;
    }
    const data = await fetchModelApi(MODEL_ENDPOINTS.clinicalNote, {
        method: "POST",
        body: JSON.stringify({
            symptoms: clinicalState.symptoms,
            duration: clinicalState.duration,
            severity: clinicalState.severity,
            descriptions: clinicalState.descriptions,
            rating: clinicalState.rating
        })
    });
    clinicalState.clinicalNote = noteWithRating(data.clinical_note || "");
    return clinicalState.clinicalNote;
}

function clinicalPromptReply(language, missingFields = clinicalMissingFields()) {
    const isZh = language === "zh";
    if (missingFields.includes("symptoms")) {
        lastClinicalAction = "ask_symptoms";
        return isZh
            ? "为了完成预约资料，请告诉我您目前有哪些症状或不适。"
            : "Untuk melengkapkan maklumat janji temu, sila beritahu simptom atau ketidakselesaan yang anda alami.";
    }
    if (missingFields.includes("duration")) {
        lastClinicalAction = "ask_duration";
        return isZh
            ? "谢谢。请问这些症状持续多久了？例如：3天、1星期。"
            : "Terima kasih. Berapa lama simptom ini telah berlarutan? Contohnya: 3 hari atau 1 minggu.";
    }
    if (missingFields.includes("severity")) {
        lastClinicalAction = "ask_severity";
        return isZh
            ? "请选择目前不适的严重程度（0–10）。0–8 可继续普通流程；9–10 请立即求助。"
            : "Pilih tahap keterukan (0–10). Tahap 0–8 meneruskan aliran biasa; 9–10 memerlukan bantuan segera.";
    }
    lastClinicalAction = "clinical_complete";
    return isZh ? "您的症状资料已记录。" : "Maklumat simptom anda telah direkodkan.";
}

async function processDescriptionIntent(processingResult, language) {
    const source = String(processingResult.originalInput || processingResult.normalizedText || "").trim();
    try {
        let extracted;
        const directSeverity = lastClinicalAction === "ask_severity" && /^(?:10|[0-9])$/u.test(source)
            ? Number(source)
            : null;
        if (directSeverity !== null) {
            extracted = { symptoms: [], duration: null, severity: directSeverity };
        } else {
            extracted = await extractClinicalFromUser(source);
        }
        mergeClinicalExtraction(extracted, source);
        const missing = clinicalMissingFields();
        if (!missing.length) {
            if (Number(clinicalState.severity) >= 9) {
                lastClinicalAction = "clinical_complete";
                return {
                    type: "emergency_severity",
                    language,
                    severity: Number(clinicalState.severity),
                    id: "severity_9_10_override",
                    text: language === "zh"
                        ? "您报告的严重程度为 9–10，情况十分严重，不建议继续普通 chatbot 咨询。"
                        : "Tahap keterukan anda ialah 9–10. Keadaan ini sangat serius dan konsultasi chatbot biasa tidak disarankan.",
                    route: "emergency_override",
                    replyConversation
                };
            }
            await refreshClinicalNote();
            lastClinicalAction = "clinical_complete";
            const shouldOpenBooking = pendingBookingAfterClinical;
            if (shouldOpenBooking) pendingBookingAfterClinical = false;
            if (!shouldOpenBooking) awaitingBookingConfirmation = true;
            return {
                type: shouldOpenBooking ? "booking_form" : "intent_reply",
                language,
                id: "clinical_complete",
                text: shouldOpenBooking
                    ? (language === "zh"
                        ? "您的症状、持续时间和严重程度都已记录。请填写以下预约表格以完成预约。"
                        : "Simptom, tempoh dan tahap keterukan anda telah direkodkan. Sila isi borang janji temu di bawah untuk melengkapkan tempahan.")
                    : (language === "zh"
                        ? "谢谢，您的症状、持续时间和严重程度都已记录。请问您要现在预约医生吗？请输入 Yes 或 No。"
                        : "Terima kasih. Simptom, tempoh dan tahap keterukan anda telah direkodkan. Adakah anda mahu membuat janji temu doktor sekarang? Masukkan Yes atau No."),
                route: "description",
                replyConversation
            };
        }
        const text = clinicalPromptReply(language, missing);
        return {
            type: missing[0] === "severity" ? "clinical_severity" : "intent_reply",
            language,
            id: `clinical_${missing[0]}_prompt`,
            text,
            route: "description",
            replyConversation
        };
    } catch (error) {
        console.error("Clinical extraction failed:", error);
        return {
            type: "intent_reply",
            language,
            id: "clinical_extraction_fallback",
            text: language === "zh"
                ? "我暂时无法处理症状资料。请再告诉我您的症状和持续时间，我会继续记录。"
                : "Saya belum dapat memproses maklumat simptom tadi. Sila beritahu simptom dan tempohnya sekali lagi; saya akan terus merekodkannya.",
            route: "description",
            replyConversation
        };
    }
}

async function processAskInfoIntent(processingResult, language) {
    const userInput = String(processingResult.originalInput || processingResult.normalizedText || "").trim();
    try {
        const result = await fetchModelApi(MODEL_ENDPOINTS.ragQuery, {
            method: "POST",
            body: JSON.stringify({
                user_input: userInput,
                response_language: clinicalLanguageName(language),
                past_queries: (conversationState.history || []).map(item => `${item.role}: ${item.content}`).join("\n"),
                intent_hint: processingResult.intentionDecision?.intention || "ask_info",
                conversation_state: conversationStateForApi(),
                conversation_history: conversationState.history || []
            })
        });
        mergeConversationStateFromApi(result.conversation_state);
        return {
            type: "intent_reply",
            language,
            id: "ask_info_rag",
            text: String(result.response || "").trim() || (language === "zh"
                ? "抱歉，您的问题在我的资料库中找不到相关资料。您可以换一种方式询问。"
                : "Maaf, maklumat berkaitan soalan anda tidak ditemui dalam pangkalan pengetahuan saya. Anda boleh bertanya dengan cara lain."),
            route: "ask_info",
            replyConversation,
            rag: result
        };
    } catch (error) {
        console.error("RAG failed:", error);
        return {
            type: "intent_reply",
            language,
            id: "ask_info_rag_fallback",
            text: language === "zh"
                ? "抱歉，我暂时无法读取诊所资料。请稍后再试，或换一种方式询问。"
                : "Maaf, saya tidak dapat membaca maklumat klinik buat masa ini. Sila cuba lagi sebentar lagi atau tanya dengan cara lain.",
            route: "ask_info",
            replyConversation
        };
    }
}

function confirmationReply(language) {
    const isZh = language === "zh";
    if (lastClinicalAction === "ask_symptoms") return clinicalPromptReply(language, ["symptoms"]);
    if (lastClinicalAction === "ask_duration") return clinicalPromptReply(language, ["duration"]);
    if (lastClinicalAction === "ask_severity") return clinicalPromptReply(language, ["severity"]);
    if (pendingRebooking) {
        return isZh ? "原预约已经取消。请完成新的预约表格以重新预约。" : "Janji temu asal telah dibatalkan. Sila lengkapkan borang baharu untuk membuat penjadualan semula.";
    }
    if (lastBookingContext?.appointmentId) {
        return isZh
            ? `您最近的预约号码是 ${lastBookingContext.appointmentId}。如果您要取消或更改预约，请告诉我。`
            : `ID janji temu terkini anda ialah ${lastBookingContext.appointmentId}. Jika anda mahu membatalkan atau mengubahnya, sila beritahu saya.`;
    }
    return isZh
        ? "可以，请告诉我您要确认的是哪一项，例如预约、取消预约、医生或时间，我会继续帮您确认。"
        : "Baik. Sila jelaskan perkara yang anda mahu sahkan, contohnya tempahan, pembatalan, doktor atau masa, dan saya akan bantu mengesahkannya.";
}

async function persistRatingToAppointment(rating) {
    if (!lastBookingContext?.appointmentId) return;
    try {
        await fetchModelApi(MODEL_ENDPOINTS.clinicalRating, {
            method: "POST",
            body: JSON.stringify({
                appointment_id: lastBookingContext.appointmentId,
                rating
            })
        });
        refreshDeveloperDatabase();
    } catch (error) {
        console.warn("Unable to persist rating to booked appointment:", error);
    }
}

function createInteractiveCard(className, innerHtml) {
    const row = document.createElement("div");
    row.className = "message-row bot-row";
    const bubble = document.createElement("div");
    bubble.className = `bubble bot-bubble interactive-card ${className}`;
    bubble.innerHTML = innerHtml;
    row.appendChild(bubble);
    messagesContainer.appendChild(row);
    scrollToLatestMessage();
    return bubble;
}

function showRatingCard(language) {
    if (ratingSubmitted) return;
    const isZh = language === "zh";
    const card = createInteractiveCard("rating-card", `
      <strong>${isZh ? "请为我们的服务评分" : "Sila nilaikan perkhidmatan kami"}</strong>
      <div class="rating-stars" role="group" aria-label="Service rating">
        ${[1,2,3,4,5].map(n => `<button type="button" data-rating="${n}" aria-label="${n} ${isZh ? "星" : "bintang"}" aria-pressed="false">★</button>`).join("")}
      </div>
      <small class="rating-status" aria-live="polite">${isZh ? "将鼠标移到星星上预览，点击即可评分" : "Gerakkan tetikus pada bintang untuk pratonton, kemudian klik untuk menilai"}</small>`);

    const stars = [...card.querySelectorAll("[data-rating]")];
    const status = card.querySelector(".rating-status");
    let selectedRating = 0;

    const paintStars = (rating) => {
        stars.forEach((star, index) => {
            const filled = index < rating;
            star.classList.toggle("is-filled", filled);
            star.setAttribute("aria-pressed", String(index + 1 === selectedRating));
        });
    };

    stars.forEach(button => {
        const rating = Number(button.dataset.rating);
        button.addEventListener("mouseenter", () => { if (!ratingSubmitted) paintStars(rating); });
        button.addEventListener("focus", () => { if (!ratingSubmitted) paintStars(rating); });
        button.addEventListener("click", async () => {
            if (ratingSubmitted) return;
            selectedRating = rating;
            ratingSubmitted = true;
            clinicalState.rating = rating;
            paintStars(rating);
            stars.forEach(star => { star.disabled = true; });
            status.textContent = isZh ? `${rating} 星评分已提交` : `Penilaian ${rating} bintang telah dihantar`;
            card.classList.add("rating-submitted");

            if (clinicalState.clinicalNote) {
                clinicalState.clinicalNote = noteWithRating(clinicalState.clinicalNote);
            }
            await persistRatingToAppointment(rating);
            createBotMessage(isZh ? "谢谢您宝贵的评分！" : "Terima kasih atas penilaian anda yang berharga!");
        });
    });

    card.querySelector(".rating-stars").addEventListener("mouseleave", () => {
        if (!ratingSubmitted) paintStars(selectedRating);
    });
}

function showSeverityCard(language) {
    const isZh = language === "zh";
    const existing = document.querySelector(".severity-card:not(.severity-submitted)");
    if (existing) return;
    const card = createInteractiveCard("severity-card", `
      <strong>${isZh ? "请选择目前不适的严重程度" : "Sila pilih tahap keterukan simptom anda"}</strong>
      <div class="severity-flames" role="group" aria-label="Severity from 0 to 10">
        ${[0,1,2,3,4,5,6,7,8,9,10].map(n => `<button type="button" data-severity="${n}" aria-label="${n} / 10" aria-pressed="false"><span aria-hidden="true">${n === 0 ? "○" : "🔥"}</span><small>${n}</small></button>`).join("")}
      </div>
      <small class="severity-status" aria-live="polite">${isZh ? "0–8 可继续普通咨询；9–10 情况十分严重，不建议继续使用 chatbot，请立即求助。" : "0–8 boleh meneruskan aliran biasa; tahap 9–10 sangat serius. Jangan teruskan chatbot dan dapatkan bantuan segera."}</small>`);

    const buttons = [...card.querySelectorAll("[data-severity]")];
    const status = card.querySelector(".severity-status");
    let selected = 0;
    let submitted = false;

    const paint = value => {
        buttons.forEach(button => {
            button.classList.toggle("is-filled", Number(button.dataset.severity) <= value && value > 0);
            button.setAttribute("aria-pressed", String(Number(button.dataset.severity) === selected));
        });
    };

    const continueClinicalFlow = async () => {
        try {
            if (clinicalIsComplete()) await refreshClinicalNote();
        } catch (error) {
            console.error("Unable to generate clinical note after severity selection:", error);
        }
        if (pendingBookingAfterClinical && clinicalIsComplete()) {
            pendingBookingAfterClinical = false;
            createBotMessage(isZh
                ? "您的症状、持续时间和严重程度都已记录。请填写以下预约表格以完成预约。"
                : "Simptom, tempoh dan tahap keterukan anda telah direkodkan. Sila isi borang janji temu di bawah untuk melengkapkan tempahan.");
            window.setTimeout(() => showBookingForm(language), 250);
        } else if (clinicalIsComplete()) {
            awaitingBookingConfirmation = true;
            createBotMessage(isZh
                ? "谢谢，您的症状资料已完整记录。请问您要现在预约医生吗？请输入 Yes 或 No。"
                : "Terima kasih. Maklumat simptom anda telah lengkap direkodkan. Adakah anda mahu membuat janji temu doktor sekarang? Masukkan Yes atau No.");
        }
    };

    buttons.forEach(button => {
        const value = Number(button.dataset.severity);
        button.addEventListener("mouseenter", () => { if (!submitted) { paint(value); status.textContent = `${value}/10`; } });
        button.addEventListener("focus", () => { if (!submitted) { paint(value); status.textContent = `${value}/10`; } });
        button.addEventListener("click", async () => {
            if (submitted) return;
            selected = value;
            submitted = true;
            clinicalState.severity = value;
            lastClinicalAction = "clinical_complete";
            paint(value);
            buttons.forEach(item => { item.disabled = true; });
            card.classList.add("severity-submitted");
            if (value >= 9) {
                status.textContent = isZh
                    ? `严重程度 ${value}/10：情况十分严重，请立即求助。`
                    : `Tahap ${value}/10: keadaan sangat serius. Dapatkan bantuan segera.`;
                showEmergencyAlertCard({ language, severity: value, onCompleted: continueClinicalFlow });
                return;
            }
            status.textContent = isZh ? `已记录严重程度：${value}/10` : `Tahap keterukan direkodkan: ${value}/10`;
            await continueClinicalFlow();
        });
    });

    card.querySelector(".severity-flames").addEventListener("mouseleave", () => {
        if (!submitted) {
            paint(selected);
            status.textContent = isZh ? "0–8 可继续普通咨询；9–10 情况十分严重，请立即求助。" : "0–8 boleh teruskan aliran biasa; 9–10 sangat serius dan memerlukan bantuan segera.";
        }
    });
}

async function showBookingForm(language, options = {}) {
    const isZh = language === "zh";
    const copy = bookingCopy(language);
    const rescheduleAppointment = options.rescheduleAppointment || pendingRebooking?.appointment || null;
    const card = createInteractiveCard("booking-card", `
      <strong>${copy.title}</strong>
      <div class="booking-progress" aria-label="Booking progress">
        <span class="is-active" data-progress="1">1</span><i></i><span data-progress="2">2</span>
      </div>
      <form class="booking-form" novalidate>
        <section class="booking-section" data-step="1">
          <h3>${copy.step1}</h3>
          <label>${copy.name}
            <input name="p_name" required maxlength="120" autocomplete="name">
            <small>${copy.nameHint}</small>
          </label>
          <label>${copy.dob}
            <input name="p_dob" type="date" required min="1900-01-01" max="${yesterdayIso()}">
          </label>
          <label>${copy.gender}
            <select name="p_gender" required>
              <option value="">${copy.choose}</option>
              <option value="M">${copy.male}</option>
              <option value="F">${copy.female}</option>
            </select>
          </label>
          <label>${copy.ic}
            <input name="p_ic" required inputmode="numeric" maxlength="14" placeholder="XXXXXX-XX-XXXX" autocomplete="off">
            <small>${copy.icHint}</small>
          </label>
          <label>${copy.contact}
            <input name="p_contact" required inputmode="numeric" maxlength="13" placeholder="01X-XXX-XXXX" autocomplete="tel">
            <small>${copy.contactHint}</small>
          </label>
          <label>${copy.note}
            <textarea name="clinical_note" maxlength="720" readonly aria-readonly="true"></textarea>
            <small><span class="note-count">0</span>/720 · ${copy.noteHint}</small>
          </label>
          <div class="booking-actions booking-actions-end">
            <button class="booking-primary booking-next" type="button">${copy.next}</button>
          </div>
        </section>

        <section class="booking-section" data-step="2" hidden>
          <h3>${copy.step2}</h3>
          <label>${copy.appointmentId}
            <input name="appointment_id" value="${copy.generated}" readonly aria-readonly="true">
          </label>
          <label>${copy.doctorType}
            <select name="doctor_type" required>
              <option value="">${copy.choose}</option>
              <option value="GENERAL">${copy.general}</option>
              <option value="DERMATOLOGY">${copy.dermatology}</option>
              <option value="OBGYN">${copy.obgyn}</option>
              <option value="PEDIATRICS">${copy.pediatrics}</option>
            </select>
          </label>
          <label class="doctor-name-field">${copy.doctorName}
            <select name="doctor_name" required disabled aria-disabled="true">
              <option value="">${copy.selectDoctorType}</option>
            </select>
            <input name="d_id" type="hidden">
          </label>
          <label class="appointment-date-field" hidden>${copy.date}
            <input name="appointment_date" type="date" required min="${malaysiaDateTimeParts().date}">
            <small>${copy.dateHelp}</small>
          </label>
          <div class="date-availability" role="status" aria-live="polite">${copy.selectDoctorType}</div>
          <fieldset class="slot-fieldset" hidden>
            <legend>${copy.slots}</legend>
            <div class="slot-grid"></div>
          </fieldset>
          <input name="start_time" type="hidden">
          <div class="booking-actions">
            <button class="booking-secondary booking-back" type="button">${copy.back}</button>
            <button class="booking-primary" type="submit">${copy.submit}</button>
          </div>
        </section>
        <p class="form-status" role="status" aria-live="polite"></p>
      </form>`);

    const form = card.querySelector("form");
    const step1 = form.querySelector('[data-step="1"]');
    const step2 = form.querySelector('[data-step="2"]');
    const progress1 = card.querySelector('[data-progress="1"]');
    const progress2 = card.querySelector('[data-progress="2"]');
    const nameInput = form.elements.p_name;
    const dobInput = form.elements.p_dob;
    const genderInput = form.elements.p_gender;
    const icInput = form.elements.p_ic;
    const contactInput = form.elements.p_contact;
    const noteInput = form.elements.clinical_note;
    noteInput.value = noteWithRating(rescheduleAppointment?.clinical_note || clinicalState.clinicalNote || "");
    noteInput.readOnly = true;
    const doctorType = form.elements.doctor_type;
    const doctorName = form.elements.doctor_name;
    const doctorId = form.elements.d_id;
    const appointmentDate = form.elements.appointment_date;
    const appointmentId = form.elements.appointment_id;
    const startTime = form.elements.start_time;
    const dateField = form.querySelector(".appointment-date-field");
    const availability = form.querySelector(".date-availability");
    const slotFieldset = form.querySelector(".slot-fieldset");
    const slotGrid = form.querySelector(".slot-grid");
    const noteCount = form.querySelector(".note-count");
    noteCount.textContent = String(noteInput.value.length);
    const status = form.querySelector(".form-status");
    const submit = form.querySelector('button[type="submit"]');
    let availableDoctors = [];
    let selectedDoctor = null;
    let selectedSlot = null;

    if (rescheduleAppointment) {
        const dobParts = String(rescheduleAppointment.p_dob || "").split("-");
        nameInput.value = rescheduleAppointment.p_name || "";
        dobInput.value = dobParts.length === 3 ? `${dobParts[2]}-${dobParts[1]}-${dobParts[0]}` : "";
        genderInput.value = rescheduleAppointment.p_gender || "";
        icInput.value = formatIcInput(rescheduleAppointment.p_ic || pendingRebooking?.patientIc || "");
        contactInput.value = formatContactInput(rescheduleAppointment.p_contact || "");
        appointmentId.value = rescheduleAppointment.appointment_id || "";
        [nameInput, dobInput, genderInput, icInput, contactInput].forEach(field => {
            field.disabled = true;
            field.setAttribute("aria-disabled", "true");
        });
    }

    const setStep = (number) => {
        const first = number === 1;
        step1.hidden = !first;
        step2.hidden = first;
        progress1.classList.toggle("is-active", first);
        progress2.classList.toggle("is-active", !first);
        if (!first) doctorType.focus();
        scrollToLatestMessage();
    };

    const updateFieldAppearance = field => {
        const invalid = Boolean(field.validationMessage) || !field.checkValidity();
        field.classList.toggle("is-invalid", invalid);
        field.setAttribute("aria-invalid", String(invalid));
    };

    const clearRelatedIdentityErrors = () => {
        [dobInput, genderInput, icInput].forEach(field => {
            field.setCustomValidity("");
            field.classList.remove("is-invalid");
            field.setAttribute("aria-invalid", "false");
        });
    };

    const validateIdentityFields = ({ requireComplete = false } = {}) => {
        clearRelatedIdentityErrors();
        const dob = dobInput.value;
        const expectedPrefix = icDatePrefixFromDob(dob);
        const icDigits = icInput.value.replace(/\D/gu, "");

        if (dob && dob < "1900-01-01") {
            dobInput.setCustomValidity(copy.invalidDobRange);
            updateFieldAppearance(dobInput);
            return false;
        }
        if (icDigits.length !== 12) {
            if (requireComplete) {
                icInput.setCustomValidity(copy.invalidIc);
                updateFieldAppearance(icInput);
                return false;
            }
            return true;
        }
        if (expectedPrefix && icDigits.slice(0, 6) !== expectedPrefix) {
            const message = copy.icDobMismatch(expectedPrefix);
            dobInput.setCustomValidity(message);
            icInput.setCustomValidity(message);
            updateFieldAppearance(dobInput);
            updateFieldAppearance(icInput);
            return false;
        }
        if (!icLastDigitMatchesGender(icDigits, genderInput.value)) {
            const message = genderInput.value === "M"
                ? copy.maleIcMismatch
                : copy.femaleIcMismatch;
            genderInput.setCustomValidity(message);
            icInput.setCustomValidity(message);
            updateFieldAppearance(genderInput);
            updateFieldAppearance(icInput);
            return false;
        }
        return true;
    };

    const syncIcPrefixFromDob = () => {
        const expectedPrefix = icDatePrefixFromDob(dobInput.value);
        const currentDigits = icInput.value.replace(/\D/gu, "");
        if (expectedPrefix && currentDigits.length <= 6) {
            icInput.value = formatIcInput(expectedPrefix);
        }
        validateIdentityFields({ requireComplete: false });
    };

    form.addEventListener("invalid", event => {
        event.target.classList.add("is-invalid");
        event.target.setAttribute("aria-invalid", "true");
    }, true);

    nameInput.addEventListener("input", () => {
        const cursor = nameInput.selectionStart;
        const cleaned = normalizePatientName(nameInput.value);
        if (nameInput.value !== cleaned) nameInput.value = cleaned;
        nameInput.setSelectionRange(Math.min(cursor, cleaned.length), Math.min(cursor, cleaned.length));
        nameInput.setCustomValidity(cleaned.trim().length ? "" : copy.invalidName);
        updateFieldAppearance(nameInput);
    });
    icInput.addEventListener("input", () => {
        icInput.value = formatIcInput(icInput.value);
        validateIdentityFields({ requireComplete: false });
    });
    dobInput.addEventListener("change", syncIcPrefixFromDob);
    dobInput.addEventListener("input", syncIcPrefixFromDob);
    genderInput.addEventListener("change", () => validateIdentityFields({ requireComplete: false }));
    contactInput.addEventListener("input", () => {
        contactInput.value = formatContactInput(contactInput.value);
        const length = contactInput.value.replace(/\D/gu, "").length;
        contactInput.setCustomValidity(length === 10 || length === 11 ? "" : copy.invalidContact);
        updateFieldAppearance(contactInput);
    });

    form.querySelector(".booking-next").addEventListener("click", () => {
        nameInput.value = normalizePatientName(nameInput.value).trim();
        nameInput.setCustomValidity(nameInput.value.length ? "" : copy.invalidName);
        const phoneLength = contactInput.value.replace(/\D/gu, "").length;
        contactInput.setCustomValidity(phoneLength === 10 || phoneLength === 11 ? "" : copy.invalidContact);
        validateIdentityFields({ requireComplete: true });
        const fields = [nameInput, dobInput, genderInput, icInput, contactInput];
        fields.forEach(updateFieldAppearance);
        const invalid = fields.find(field => !field.checkValidity());
        if (invalid) {
            invalid.reportValidity();
            invalid.focus();
            return;
        }
        if (pendingRebooking && formatIcInput(icInput.value) !== pendingRebooking.patientIc) {
            icInput.setCustomValidity(copy.rebookIc);
            updateFieldAppearance(icInput);
            icInput.reportValidity();
            icInput.focus();
            return;
        }
        icInput.setCustomValidity("");
        setStep(2);
    });

    form.querySelector(".booking-back").addEventListener("click", () => setStep(1));

    doctorType.addEventListener("change", async () => {
        availableDoctors = [];
        selectedDoctor = null;
        selectedSlot = null;
        doctorName.value = "";
        doctorName.replaceChildren(new Option(copy.selectDoctorType, ""));
        doctorName.disabled = true;
        doctorName.setAttribute("aria-disabled", "true");
        doctorId.value = "";
        appointmentDate.value = "";
        startTime.value = "";
        dateField.hidden = true;
        slotFieldset.hidden = true;
        slotGrid.innerHTML = "";
        if (!doctorType.value) {
            availability.textContent = copy.selectDoctorType;
            return;
        }
        availability.textContent = copy.loadingDoctors;
        doctorType.disabled = true;
        try {
            const data = await bookingApiJson(BOOKING_ENDPOINTS.doctors, {
                params: { expertise: doctorType.value }
            });
            if (!data.doctors?.length) throw new Error(data.message || "No active doctor was found.");
            // V22 safeguard: the local proxy should forward the expertise query,
            // but also filter the returned list here so another API/proxy regression
            // can never mix doctors from other specialties into the selected type.
            const selectedExpertise = String(doctorType.value || "").trim().toUpperCase();
            availableDoctors = data.doctors.filter(doctor =>
                String(doctor?.d_expertise || "").trim().toUpperCase() === selectedExpertise
            );
            if (!availableDoctors.length) {
                throw new Error(data.message || "No active doctor was found for the selected doctor type.");
            }
            doctorName.replaceChildren(new Option(copy.chooseDoctor, ""));
            availableDoctors.forEach(doctor => {
                const option = document.createElement("option");
                option.value = doctor.d_id;
                option.textContent = doctorDisplayName(doctor, language);
                doctorName.appendChild(option);
            });
            doctorName.disabled = false;
            doctorName.setAttribute("aria-disabled", "false");
            dateField.hidden = true;
            availability.textContent = copy.doctorsFound(availableDoctors.length);
        } catch (error) {
            doctorName.disabled = true;
            doctorName.setAttribute("aria-disabled", "true");
            availability.textContent = error.message;
        } finally {
            doctorType.disabled = false;
        }
    });

    doctorName.addEventListener("change", () => {
        selectedDoctor = availableDoctors.find(doctor => doctor.d_id === doctorName.value) || null;
        selectedSlot = null;
        doctorId.value = selectedDoctor?.d_id || "";
        appointmentDate.value = "";
        startTime.value = "";
        slotGrid.replaceChildren();
        slotFieldset.hidden = true;
        dateField.hidden = !selectedDoctor;
        availability.textContent = selectedDoctor ? copy.dateHelp : copy.chooseDoctor;
    });

    appointmentDate.addEventListener("change", async () => {
        selectedSlot = null;
        startTime.value = "";
        slotGrid.innerHTML = "";
        slotFieldset.hidden = true;
        if (!selectedDoctor || !appointmentDate.value) return;
        availability.textContent = copy.loadingSlots;
        appointmentDate.disabled = true;
        try {
            const data = await bookingApiJson(BOOKING_ENDPOINTS.availableSlots, {
                params: {
                    d_id: selectedDoctor.d_id,
                    appointment_date: appointmentDate.value
                }
            });
            let futureSlots = (data.slots || []).filter(slot =>
                isFutureMalaysiaSlot(appointmentDate.value, slot)
            );
            const originalSlot = String(rescheduleAppointment?.start_time || "");
            const viewingOriginalSchedule = Boolean(
                rescheduleAppointment
                && String(selectedDoctor?.d_id) === String(rescheduleAppointment.d_id)
                && String(appointmentDate.value) === String(rescheduleAppointment.appointment_date)
                && originalSlot
                && isFutureMalaysiaSlot(appointmentDate.value, originalSlot)
            );
            if (viewingOriginalSchedule && !futureSlots.includes(originalSlot)) {
                futureSlots = [originalSlot, ...futureSlots].sort();
            }
            if (data.status !== "AVAILABLE" || !futureSlots.length) {
                availability.textContent = bookingAvailabilityMessage(data, copy, language);
                if (data.status === "AVAILABLE" && data.slots?.length) {
                    availability.textContent = language === "zh"
                        ? "今天剩余的预约时间已经结束，请选择另一个日期。"
                        : "Semua waktu janji temu yang tinggal untuk hari ini telah berlalu. Sila pilih tarikh lain.";
                }
                return;
            }
            availability.textContent = language === "zh"
                ? `目前还有 ${futureSlots.length} 个可预约时段。`
                : `Terdapat ${futureSlots.length} slot yang masih tersedia.`;
            slotFieldset.hidden = false;
            futureSlots.forEach(slot => {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "slot-button";
                button.textContent = slot;
                button.addEventListener("click", () => {
                    selectedSlot = slot;
                    startTime.value = slot;
                    slotGrid.querySelectorAll(".slot-button").forEach(item => {
                        item.classList.toggle("is-selected", item === button);
                        item.setAttribute("aria-pressed", String(item === button));
                    });
                    status.textContent = "";
                });
                slotGrid.appendChild(button);
            });
        } catch (error) {
            availability.textContent = error.message;
        } finally {
            appointmentDate.disabled = false;
        }
    });

    if (rescheduleAppointment) {
        const waitUntil = async (condition, timeoutMs = 10000) => {
            const deadline = Date.now() + timeoutMs;
            while (Date.now() < deadline) {
                if (condition()) return true;
                await new Promise(resolve => window.setTimeout(resolve, 50));
            }
            return false;
        };

        doctorType.value = String(rescheduleAppointment.doctor_expertise || "").toUpperCase();
        doctorType.dispatchEvent(new Event("change"));
        await waitUntil(() => !doctorType.disabled && Array.from(doctorName.options)
            .some(option => option.value === rescheduleAppointment.d_id));

        doctorName.value = rescheduleAppointment.d_id || "";
        doctorName.dispatchEvent(new Event("change"));
        appointmentDate.value = rescheduleAppointment.appointment_date || "";
        appointmentDate.dispatchEvent(new Event("change"));
        await waitUntil(() => !appointmentDate.disabled && (
            slotGrid.querySelectorAll(".slot-button").length > 0
            || !slotFieldset.hidden
        ));

        const originalSlotButton = Array.from(slotGrid.querySelectorAll(".slot-button"))
            .find(button => button.textContent.trim() === String(rescheduleAppointment.start_time || "").trim());
        originalSlotButton?.click();
    }

    form.addEventListener("submit", async event => {
        event.preventDefault();
        if (!selectedDoctor || !appointmentDate.value || !selectedSlot) {
            status.textContent = copy.chooseSlot;
            return;
        }
        submit.disabled = true;
        status.textContent = copy.submitting;
        try {
            const patientIc = formatIcInput(icInput.value);
            let patient = { p_id: rescheduleAppointment?.p_id || null };
            let result;
            if (rescheduleAppointment) {
                const reasonParts = [];
                if (String(selectedDoctor.d_id) !== String(rescheduleAppointment.d_id)) {
                    reasonParts.push("CHANGE_DOCTOR");
                }
                if (String(appointmentDate.value) !== String(rescheduleAppointment.appointment_date)) {
                    reasonParts.push("CHANGE_DATE");
                }
                if (String(selectedSlot) !== String(rescheduleAppointment.start_time)) {
                    reasonParts.push("CHANGE_TIME");
                }
                result = await bookingApiJson(
                    `${BOOKING_ENDPOINTS.appointments}/${rescheduleAppointment.appointment_id}/reschedule`,
                    {
                        method: "PATCH",
                        body: JSON.stringify({
                            patient_ic: patientIc,
                            d_id: selectedDoctor.d_id,
                            appointment_date: appointmentDate.value,
                            start_time: selectedSlot,
                            clinical_note: noteInput.value.trim() || null,
                            reschedule_reason: reasonParts.length
                                ? reasonParts.join(" + ")
                                : "NO_SCHEDULE_CHANGE"
                        })
                    }
                );
                pendingRebooking = null;
            } else {
                patient = await bookingApiJson(BOOKING_ENDPOINTS.patients, {
                    method: "POST",
                    body: JSON.stringify({
                        p_name: normalizePatientName(nameInput.value).trim(),
                        p_dob: databaseDob(dobInput.value),
                        p_gender: genderInput.value,
                        p_ic: patientIc,
                        p_contact: databaseContact(contactInput.value)
                    })
                });
                result = await bookingApiJson(BOOKING_ENDPOINTS.appointments, {
                    method: "POST",
                    body: JSON.stringify({
                        p_id: patient.p_id,
                        d_id: selectedDoctor.d_id,
                        appointment_date: appointmentDate.value,
                        start_time: selectedSlot,
                        clinical_note: noteInput.value.trim() || null,
                        status: "CONFIRMED"
                    })
                });
            }

            appointmentId.value = result.appointment_id;
            lastBookingContext = {
                appointmentId: result.appointment_id,
                patientIc,
                patientId: patient.p_id
            };
            [...form.elements].forEach(element => { element.disabled = true; });
            appointmentId.disabled = false;
            appointmentId.readOnly = true;

            let rebookingCompleted = Boolean(rescheduleAppointment);
            let rebookingWarning = "";
            if (pendingRebooking && !rescheduleAppointment) {
                const rebookingContext = { ...pendingRebooking };
                try {
                    await completeRebookingWithRetry(
                        rebookingContext,
                        result.appointment_id,
                        patientIc
                    );
                    rebookingCompleted = true;
                    pendingRebooking = null;
                    recentCancelledAppointment = null;
                } catch (error) {
                    // Keep the context instead of silently losing the relationship.
                    pendingRebooking = {
                        ...rebookingContext,
                        newAppointmentId: result.appointment_id
                    };
                    rebookingWarning = isZh
                        ? `重新预约记录尚未完成连接：${error.message}`
                        : `Rekod penjadualan semula masih belum berjaya dipautkan: ${error.message}`;
                }
            }

            const bookedDoctorName = String(
                result.doctor_name || selectedDoctor.d_name || ""
            ).replace(/^Dr\.?\s*/iu, "").trim();
            const successMessage = bookingSuccessMessage(
                language,
                result,
                bookedDoctorName,
                rebookingCompleted,
                rebookingWarning
            );
            status.textContent = "";
            card.classList.add("booking-completed");
            createBotMessage(successMessage);
            refreshDeveloperDatabase();
            resetClinicalStateForNextBooking();
        } catch (error) {
            status.textContent = error.message;
            submit.disabled = false;
        }
    });

    try {
        await bookingApiJson(BOOKING_ENDPOINTS.health);
    } catch (error) {
        status.textContent = error.message;
        form.querySelector(".booking-next").disabled = true;
    }
}

function resetCancellationFlow() {
    awaitingAppointmentId = false;
    awaitingCancellationIc = false;
    awaitingCancelConfirmation = false;
    pendingCancellationId = null;
    pendingCancellationIc = null;
    cancellationMode = "cancel";
}

async function cancelAppointment(payload, language) {
    const { appointmentId, patientIc, mode } = payload;
    const isZh = language === "zh";
    const reason = mode === "rebook"
        ? "Patient requested reschedule through chatbot."
        : "Cancelled by patient through chatbot.";
    try {
        const result = await bookingApiJson(`${BOOKING_ENDPOINTS.appointments}/${appointmentId}/cancel`, {
            method: "PATCH",
            body: JSON.stringify({
                patient_ic: patientIc,
                cancellation_reason: reason
            })
        });
        refreshDeveloperDatabase();

        if (mode === "rebook") {
            recentCancelledAppointment = null;
            pendingRebooking = {
                oldAppointmentId: appointmentId,
                patientIc,
                reason
            };
            const missing = clinicalMissingFields();
            if (missing.length) {
                pendingBookingAfterClinical = true;
                const prompt = clinicalPromptReply(language, missing);
                createBotMessage(isZh
                    ? `原预约 ${appointmentId} 已取消。建立新的预约前，需要先完成症状资料。${prompt}`
                    : `Janji temu asal ${appointmentId} telah dibatalkan. Sebelum membuat tempahan baharu, maklumat simptom perlu dilengkapkan. ${prompt}`);
                if (missing[0] === "severity") showSeverityCard(language);
                return;
            }
            pendingBookingAfterClinical = false;
            try { if (!clinicalState.clinicalNote) await refreshClinicalNote(); } catch (error) { console.error(error); }
            createBotMessage(isZh
                ? `原预约 ${appointmentId} 已取消。请填写以下预约表格来选择新的预约。`
                : `Janji temu asal ${appointmentId} telah dibatalkan. Sila isi borang di bawah untuk memilih janji temu baharu.`);
            window.setTimeout(() => showBookingForm(language), 250);
            return;
        }

        // Compatibility: if the user books again in this same chatbot session, preserve the relationship.
        recentCancelledAppointment = {
            oldAppointmentId: appointmentId,
            patientIc,
            reason: "Patient cancelled and rebooked in the same chatbot session.",
            cancelledAt: new Date().toISOString()
        };
        createBotMessage(isZh
            ? `预约 ${result.appointment_id} 已成功取消。`
            : `Janji temu ${result.appointment_id} berjaya dibatalkan.`);
    } catch (error) {
        createBotMessage(isZh
            ? `无法取消预约：${error.message}`
            : `Janji temu tidak dapat dibatalkan: ${error.message}`);
    }
}

function getMonitorState() {
    try {
        const parsed = JSON.parse(localStorage.getItem(MONITOR_STORAGE_KEY));
        if (parsed && Array.isArray(parsed.history)) return parsed;
    } catch (error) {
        console.warn("Unable to read monitor data:", error);
    }

    return {
        version: 17,
        sessionStartedAt: new Date().toISOString(),
        lastUpdatedAt: null,
        turnCount: 0,
        currentTurn: null,
        initialGreeting: null,
        replyConversation: null,
        awaitingLanguagePreference: false,
        explicitLanguagePreference: null,
        resources: LANGUAGE_RESOURCES.counts,
        resourceErrors: LANGUAGE_RESOURCES.loadErrors,
        resourceWarnings: LANGUAGE_RESOURCES.loadWarnings,
        resourceSources: LANGUAGE_RESOURCES.sources,
        resourcePaths: RESOURCE_PATHS,
        modelRuntime: modelRuntimeSnapshot(),
        history: []
    };
}

function saveMonitorState(state) {
    try {
        state.lastUpdatedAt = new Date().toISOString();
        state.replyConversation = replyConversation;
        state.awaitingLanguagePreference = awaitingLanguagePreference;
        state.explicitLanguagePreference = explicitLanguagePreference;
        state.resources = { ...LANGUAGE_RESOURCES.counts };
        state.resourceErrors = [...LANGUAGE_RESOURCES.loadErrors];
        state.resourceWarnings = [...LANGUAGE_RESOURCES.loadWarnings];
        state.resourceSources = { ...LANGUAGE_RESOURCES.sources };
        state.resourcePaths = { ...RESOURCE_PATHS };
        state.modelRuntime = modelRuntimeSnapshot();
        localStorage.setItem(MONITOR_STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
        console.warn("Unable to save monitor data:", error);
    }
}

function initializeMonitorSession(initialGreeting) {
    replyConversation = null;
    explicitLanguagePreference = null;
    awaitingLanguagePreference = false;
    const state = {
        version: 17,
        sessionStartedAt: new Date().toISOString(),
        lastUpdatedAt: new Date().toISOString(),
        turnCount: 0,
        currentTurn: null,
        initialGreeting,
        replyConversation,
        awaitingLanguagePreference,
        explicitLanguagePreference,
        resources: { ...LANGUAGE_RESOURCES.counts },
        resourceErrors: [...LANGUAGE_RESOURCES.loadErrors],
        resourceWarnings: [...LANGUAGE_RESOURCES.loadWarnings],
        resourceSources: { ...LANGUAGE_RESOURCES.sources },
        resourcePaths: { ...RESOURCE_PATHS },
        modelRuntime: modelRuntimeSnapshot(),
        history: []
    };
    saveMonitorState(state);
}

function reserveMonitorTurn(record) {
    const state = getMonitorState();
    const turn = state.turnCount + 1;
    state.turnCount = turn;
    state.currentTurn = turn;
    state.history.push({
        ...record,
        turn,
        status: "typing",
        createdAt: new Date().toISOString(),
        completedAt: null
    });
    if (state.history.length > 20) state.history.splice(0, state.history.length - 20);
    saveMonitorState(state);
    return turn;
}

function finalizeMonitorTurn(turn, reply) {
    const state = getMonitorState();
    const record = state.history.find((item) => item.turn === turn);
    if (!record) return;
    record.status = "completed";
    record.completedAt = new Date().toISOString();
    record.reply = reply;
    record.replyConversation = reply.replyConversation || replyConversation;
    record.replyRoute = reply.route || record.replyRoute;
    state.currentTurn = turn;
    saveMonitorState(state);
}

function meaningToTag(meaning) {
    const normalizedMeaning = String(meaning || "emoji")
        .normalize("NFKD")
        .replace(/[\u0300-\u036f]/gu, "")
        .toLocaleLowerCase("en")
        .replace(/[^a-z0-9]+/gu, "_")
        .replace(/^_+|_+$/gu, "");
    return `<${normalizedMeaning || "emoji"}>`;
}

function buildExpressionEntries() {
    const chatbotEntries = Array.from(CHATBOT_EMOJI_MAP, ([expression, meaning]) => ({
        expression,
        tag: `<${meaning}>`,
        caseInsensitive: false
    }));

    const emoticonEntries = (SOURCE_TEXT_DATA.emoticons || []).map((item) => ({
        expression: String(item.emoticon),
        tag: meaningToTag(item.meaning),
        caseInsensitive: /[a-z]/iu.test(String(item.emoticon))
    }));

    return [...chatbotEntries, ...emoticonEntries]
        .filter((item) => item.expression)
        .sort((a, b) => b.expression.length - a.expression.length);
}

const EXPRESSION_ENTRIES = buildExpressionEntries();

function expressionMatchesAt(text, index, entry) {
    const candidate = text.slice(index, index + entry.expression.length);
    return entry.caseInsensitive
        ? candidate.toLocaleLowerCase("en") === entry.expression.toLocaleLowerCase("en")
        : candidate === entry.expression;
}

function convertKnownExpressionsToTags(input) {
    let output = "";
    let index = 0;

    while (index < input.length) {
        const match = EXPRESSION_ENTRIES.find((entry) => expressionMatchesAt(input, index, entry));
        if (match) {
            output += ` ${match.tag} `;
            index += match.expression.length;
            continue;
        }
        output += input[index];
        index += 1;
    }

    return output.replace(
        /\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?/gu,
        " <emoji> "
    );
}

function getMedicalEntry(token) {
    const key = normalizeLookupKey(token);
    return LANGUAGE_RESOURCES.medicalAliasMap.get(key) || null;
}

function isMedicalWord(token) {
    return Boolean(getMedicalEntry(token));
}

function isTag(text) {
    return /^<[a-z0-9_]+>$/u.test(text);
}

function removeSentenceEdgePunctuation(token) {
    const leadingMatch = token.match(/^[,!?;:"“”‘’()\[\]{}，。！？；：]+/u);
    const trailingMatch = token.match(/[,!?;:"“”‘’()\[\]{}，。！？；：]+$/u);
    const leading = leadingMatch ? leadingMatch[0] : "";
    const trailing = trailingMatch ? trailingMatch[0] : "";
    const core = token.slice(leading.length, token.length - trailing.length || undefined);
    return { leading, core, trailing };
}

function reduceLetterElongation(text) {
    return text.replace(/([a-z])\1{2,}/giu, "$1$1");
}

function processLatinSegment(segment) {
    return reduceLetterElongation(segment.toLocaleLowerCase("en"));
}

function processNonWhitespaceToken(token) {
    if (!token || isTag(token) || isMedicalWord(token)) return token;

    const { leading, core, trailing } = removeSentenceEdgePunctuation(token);
    if (core && isMedicalWord(core)) return `${leading}${core}${trailing}`;

    const segments = token.match(
        /<[a-z0-9_]+>|\p{Script=Han}+|[A-Za-z]+(?:[A-Za-z0-9.'’:/+\-]*[A-Za-z0-9])?|[0-9]+|[^\s]/gu
    ) || [token];

    return segments.map((segment) => {
        if (isTag(segment) || containsHanzi(segment) || isMedicalWord(segment)) return segment;
        if (/[A-Za-z]/u.test(segment)) return processLatinSegment(segment);
        return segment;
    }).join("");
}

function normalizeSequenceText(input) {
    const expressionsConverted = convertKnownExpressionsToTags(String(input));
    return expressionsConverted
        .trim()
        .split(/\s+/u)
        .filter(Boolean)
        .map(processNonWhitespaceToken)
        .join(" ");
}

function countCharacterType(text, pattern) {
    return Array.from(text).filter((character) => pattern.test(character)).length;
}

function countTypeTransitions(token) {
    let transitions = 0;
    let previousType = null;
    for (const character of token) {
        const currentType = /[A-Za-z]/u.test(character)
            ? "letter"
            : /[0-9]/u.test(character)
                ? "digit"
                : "other";
        if (previousType && currentType !== previousType && currentType !== "other" && previousType !== "other") {
            transitions += 1;
        }
        previousType = currentType;
    }
    return transitions;
}

function commonBigramRatio(word) {
    const letters = word.toLocaleLowerCase("en").replace(/[^a-z]/gu, "");
    if (letters.length < 2) return 1;
    let commonCount = 0;
    const total = letters.length - 1;
    for (let index = 0; index < total; index += 1) {
        if (COMMON_LATIN_BIGRAMS.has(letters.slice(index, index + 2))) commonCount += 1;
    }
    return commonCount / total;
}

function vowelRatio(word) {
    const letters = word.toLocaleLowerCase("en").replace(/[^a-z]/gu, "");
    if (!letters.length) return 0;
    return (letters.match(/[aeiouy]/gu) || []).length / letters.length;
}

function longestConsonantRun(word) {
    const runs = word.toLocaleLowerCase("en").replace(/[^a-z]/gu, " ").match(/[bcdfghjklmnpqrstvwxz]+/gu) || [];
    return runs.reduce((maximum, run) => Math.max(maximum, run.length), 0);
}

function isKnownLanguageWord(token) {
    const key = normalizeLookupKey(token);
    return isMedicalWord(key) ||
        LANGUAGE_RESOURCES.malayNormalizerMap.has(key) ||
        LANGUAGE_RESOURCES.malayDictionarySet.has(key) ||
        LANGUAGE_RESOURCES.pinyinSet.has(key);
}

function inspectSuspiciousToken(token) {
    const findings = [];
    const cleanToken = String(token || "").replace(/<[a-z0-9_]+>/gu, "");
    const latinLetters = cleanToken.match(/[A-Za-z]/gu) || [];
    const digits = cleanToken.match(/[0-9]/gu) || [];
    const digitGroups = cleanToken.match(/[0-9]+/gu) || [];
    const visibleCharacters = Array.from(cleanToken);
    const symbolCount = visibleCharacters.filter(
        (character) => !/[A-Za-z0-9\p{Script=Han}]/u.test(character)
    ).length;

    if (!cleanToken || isMedicalWord(cleanToken) || containsHanzi(cleanToken) || isKnownLanguageWord(cleanToken)) {
        return findings;
    }

    if (/[\\|`~]/u.test(cleanToken)) {
        findings.push({ score: 2, reason: "invalid symbol inside token" });
    }

    const lettersOnly = cleanToken.replace(/[^A-Za-z]/gu, "");
    const bigramRatio = commonBigramRatio(lettersOnly);
    const tokenVowelRatio = vowelRatio(lettersOnly);
    const consonantRun = longestConsonantRun(lettersOnly);

    if (
        cleanToken.length >= 7 &&
        latinLetters.length >= 4 &&
        digits.length >= 1 &&
        (/^[0-9]/u.test(cleanToken) || countTypeTransitions(cleanToken) >= 2)
    ) {
        findings.push({ score: 3, reason: "random letter-number mixing" });
    }

    if (
        lettersOnly.length >= 7 &&
        tokenVowelRatio < 0.2 &&
        consonantRun >= 4 &&
        bigramRatio < 0.55
    ) {
        findings.push({ score: 4, reason: "improbable consonant sequence" });
    } else if (lettersOnly.length >= 8 && bigramRatio < 0.38) {
        findings.push({ score: 3, reason: "unlikely letter sequence" });
    } else if (lettersOnly.length >= 7 && bigramRatio < 0.3) {
        findings.push({ score: 2, reason: "low natural-language bigram ratio" });
    }

    if (
        cleanToken.length >= 7 &&
        latinLetters.length >= 3 &&
        digits.length >= 2 &&
        digitGroups.length >= 2 &&
        countTypeTransitions(cleanToken) >= 3
    ) {
        findings.push({ score: 3, reason: "multiple random letter-number transitions" });
    }

    if (
        visibleCharacters.length >= 5 &&
        symbolCount >= 3 &&
        symbolCount / visibleCharacters.length >= 0.35
    ) {
        findings.push({ score: 2, reason: "excessive symbol noise" });
    }

    if (/([\[\]{};,.])\1{1,}|(?:\[\]|\{\}){2,}/u.test(cleanToken)) {
        findings.push({ score: 1, reason: "repeated punctuation pattern" });
    }

    return findings;
}

function detectGibberish(normalizedText) {
    if (!normalizedText) return { isGibberish: true, reasons: ["empty input"] };

    const withoutTags = normalizedText.replace(/<[a-z0-9_]+>/gu, " ").trim();
    if (!withoutTags) return { isGibberish: false, reasons: [] };

    const tokens = withoutTags.split(/\s+/u).filter(Boolean);
    const suspiciousTokens = [];
    const reasons = [];
    let totalScore = 0;

    tokens.forEach((token) => {
        const findings = inspectSuspiciousToken(token);
        if (findings.length > 0) {
            suspiciousTokens.push(token);
            findings.forEach((finding) => {
                totalScore += finding.score;
                reasons.push(`${token}: ${finding.reason}`);
            });
        }
    });

    const allCharacters = Array.from(withoutTags);
    const lettersOrHanziOrDigits = countCharacterType(
        withoutTags,
        /[A-Za-z0-9\p{Script=Han}]/u
    );
    const globalSymbolRatio = allCharacters.length
        ? (allCharacters.length - lettersOrHanziOrDigits - countCharacterType(withoutTags, /\s/u)) / allCharacters.length
        : 0;

    if (allCharacters.length >= 8 && globalSymbolRatio > 0.38) {
        totalScore += 2;
        reasons.push("high overall symbol ratio");
    }

    const suspiciousRatio = tokens.length ? suspiciousTokens.length / tokens.length : 0;
    const isGibberish =
        (totalScore >= 3 && suspiciousRatio >= 0.5) ||
        totalScore >= 5;

    return {
        isGibberish,
        reasons: [...new Set(reasons)]
    };
}

function findFullExpressionEntry(token) {
    return EXPRESSION_ENTRIES.find((entry) => {
        return entry.caseInsensitive
            ? token.toLocaleLowerCase("en") === entry.expression.toLocaleLowerCase("en")
            : token === entry.expression;
    }) || null;
}

function tokenizeInputForMonitor(input) {
    const text = String(input || "");
    const tokens = [];
    let index = 0;

    while (index < text.length) {
        if (/\s/u.test(text[index])) {
            index += 1;
            continue;
        }

        const expression = EXPRESSION_ENTRIES.find((entry) => expressionMatchesAt(text, index, entry));
        if (expression) {
            tokens.push(text.slice(index, index + expression.expression.length));
            index += expression.expression.length;
            continue;
        }

        const rest = text.slice(index);
        const pictographicMatch = rest.match(/^\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?/u);
        if (pictographicMatch) {
            tokens.push(pictographicMatch[0]);
            index += pictographicMatch[0].length;
            continue;
        }

        if (/\p{Script=Han}/u.test(text[index])) {
            let endIndex = index + 1;
            while (endIndex < text.length && /\p{Script=Han}/u.test(text[endIndex])) endIndex += 1;
            tokens.push(text.slice(index, endIndex));
            index = endIndex;
            continue;
        }

        if (/[A-Za-z0-9]/u.test(text[index])) {
            let endIndex = index + 1;
            while (
                endIndex < text.length &&
                /[A-Za-z0-9.'’:/+\\\-]/u.test(text[endIndex]) &&
                !EXPRESSION_ENTRIES.some((entry) => expressionMatchesAt(text, endIndex, entry))
            ) {
                endIndex += 1;
            }
            tokens.push(text.slice(index, endIndex));
            index = endIndex;
            continue;
        }

        let endIndex = index + 1;
        while (
            endIndex < text.length &&
            !/\s|[A-Za-z0-9\p{Script=Han}]/u.test(text[endIndex]) &&
            !EXPRESSION_ENTRIES.some((entry) => expressionMatchesAt(text, endIndex, entry))
        ) {
            endIndex += 1;
        }
        tokens.push(text.slice(index, endIndex));
        index = endIndex;
    }

    return tokens.filter(Boolean);
}

function isPhraseToken(token) {
    return /^[A-Za-z0-9][A-Za-z0-9'’\-]*$/u.test(token);
}

function mergeKnownPhrases(tokens) {
    const merged = [];
    let index = 0;
    const maximumWords = Math.max(
        LANGUAGE_RESOURCES.medicalPhraseMaxWords,
        LANGUAGE_RESOURCES.malayNormalizerMaxWords
    );

    while (index < tokens.length) {
        let matched = null;
        const available = Math.min(maximumWords, tokens.length - index);

        for (let size = available; size >= 2; size -= 1) {
            const candidateTokens = tokens.slice(index, index + size);
            if (!candidateTokens.every(isPhraseToken)) continue;
            const display = candidateTokens.join(" ");
            const key = normalizeLookupKey(display);
            const medical = LANGUAGE_RESOURCES.medicalAliasMap.get(key);
            const normalizer = LANGUAGE_RESOURCES.malayNormalizerMap.get(key);

            if (medical) {
                matched = { token: display, forcedMedical: medical, size };
                break;
            }
            if (normalizer) {
                matched = { token: display, forcedNormalizer: normalizer, size };
                break;
            }
        }

        if (matched) {
            merged.push(matched);
            index += matched.size;
        } else {
            merged.push({ token: tokens[index] });
            index += 1;
        }
    }

    return merged;
}

function inferMedicalLanguage(medicalEntry, token) {
    const key = normalizeLookupKey(token);
    if (containsHanzi(token)) return "hanzi";

    const languages = medicalEntry?.sourceLanguages || new Set();
    if (languages.size === 1) return [...languages][0];
    if (languages.has("malay") && LANGUAGE_RESOURCES.malayDictionarySet.has(key)) return "malay";
    if (languages.has("other")) return "other";
    if (languages.has("malay")) return "malay";
    return "other";
}

function detectTokenLanguage(value, medicalEntry = null, informalNormalized = false) {
    const token = String(value || "").trim();
    if (!token) return "other";
    if (containsHanzi(token)) return "hanzi";
    if (informalNormalized) return "malay";
    if (medicalEntry) return inferMedicalLanguage(medicalEntry, token);

    const key = normalizeLookupKey(token);
    const isMalay = LANGUAGE_RESOURCES.malayDictionarySet.has(key);
    const isPinyin = LANGUAGE_RESOURCES.pinyinSet.has(key);

    if (isMalay && !isPinyin) return "malay";
    if (isPinyin && !isMalay) return "pinyin";
    return "pending_mbert";
}

function joinNormalizedValues(values) {
    return values
        .filter((value) => value !== "")
        .join(" ")
        .replace(/([\p{Script=Han}])\s+(?=[\p{Script=Han}])/gu, "$1")
        .replace(/\s+([,!?;:，。！？；：])/gu, "$1")
        .replace(/([\[(（【])\s+/gu, "$1")
        .replace(/\s+([\])）】])/gu, "$1")
        .replace(/\s+/gu, " ")
        .trim();
}

function chooseBetterPinyinSegmentation(current, candidate) {
    if (!current) return candidate;
    if (candidate.length < current.length) return candidate;
    if (candidate.length > current.length) return current;

    // Tie-breaker: prefer a longer syllable earlier in the sequence.
    for (let index = 0; index < candidate.length; index += 1) {
        if (candidate[index].length > current[index].length) return candidate;
        if (candidate[index].length < current[index].length) return current;
    }
    return current;
}

function segmentConnectedPinyinWord(word) {
    const clean = normalizeLookupKey(word).replace(/[^a-z]/gu, "");
    if (!clean) {
        return {
            success: false,
            segments: [],
            segmentedInput: "",
            status: "empty"
        };
    }

    if (LANGUAGE_RESOURCES.pinyinSet.has(clean)) {
        return {
            success: true,
            segments: [clean],
            segmentedInput: clean,
            status: "already_valid_syllable"
        };
    }

    const bestFrom = new Array(clean.length + 1).fill(null);
    bestFrom[clean.length] = [];

    for (let start = clean.length - 1; start >= 0; start -= 1) {
        let best = null;
        for (let end = start + 1; end <= clean.length; end += 1) {
            const syllable = clean.slice(start, end);
            if (!LANGUAGE_RESOURCES.pinyinSet.has(syllable)) continue;
            const suffix = bestFrom[end];
            if (!suffix) continue;
            const candidate = [syllable, ...suffix];
            best = chooseBetterPinyinSegmentation(best, candidate);
        }
        bestFrom[start] = best;
    }

    const segments = bestFrom[0] || [];
    return {
        success: segments.length > 0,
        segments,
        segmentedInput: segments.join(" "),
        status: segments.length > 0 ? "segmented" : "segmentation_failed"
    };
}

function segmentPinyinForHmm(value) {
    const parts = String(value || "")
        .trim()
        .toLocaleLowerCase("en")
        .split(/\s+/u)
        .filter(Boolean);

    if (!parts.length) {
        return {
            success: false,
            segments: [],
            segmentedInput: "",
            status: "empty"
        };
    }

    const allSegments = [];
    let usedSegmentation = false;

    for (const part of parts) {
        const result = segmentConnectedPinyinWord(part);
        if (!result.success) {
            return {
                success: false,
                segments: [],
                segmentedInput: "",
                status: "segmentation_failed",
                failedPart: part
            };
        }
        if (result.status === "segmented") usedSegmentation = true;
        allSegments.push(...result.segments);
    }

    return {
        success: true,
        segments: allSegments,
        segmentedInput: allSegments.join(" "),
        status: usedSegmentation ? "segmented" : "already_segmented"
    };
}

function analyzeToken(item, processingResult) {
    const token = item.token;
    const expression = findFullExpressionEntry(token);

    if (expression) {
        const isUnicodeEmoji = CHATBOT_EMOJI_MAP.has(token) || /\p{Extended_Pictographic}/u.test(token);
        return {
            input: token,
            category: isUnicodeEmoji ? "unicode_emoji" : "text_based_emoticon",
            language: "other",
            label_source: isUnicodeEmoji ? "unicode_emoji_map" : "emoticon_map",
            confidence: null,
            transform_to: expression.tag,
            transform_source: "expression_normalizer",
            normalized_value: expression.tag,
            pinyin_segment: "-",
            thresholdEligible: false,
            model_input: null
        };
    }

    if (/\p{Extended_Pictographic}/u.test(token)) {
        return {
            input: token,
            category: "unicode_emoji",
            language: "other",
            label_source: "unicode_regex",
            confidence: null,
            transform_to: "<emoji>",
            transform_source: "unicode_regex",
            normalized_value: "<emoji>",
            pinyin_segment: "-",
            thresholdEligible: false,
            model_input: null
        };
    }

    if (/^\p{Script=Han}+$/u.test(token)) {
        const medical = item.forcedMedical || getMedicalEntry(token);
        return {
            input: token,
            category: medical ? "medical_word" : "-",
            language: "hanzi",
            label_source: medical ? "medical_wordlist+hanzi_regex" : "hanzi_regex",
            confidence: null,
            transform_to: "-",
            transform_source: "-",
            normalized_value: token,
            pinyin_segment: "-",
            medical_id: medical?.item?.id || null,
            medical_category: medical?.item?.category || null,
            thresholdEligible: true,
            model_input: token
        };
    }

    const punctuationParts = removeSentenceEdgePunctuation(token);
    const core = punctuationParts.core;
    const medical = item.forcedMedical || getMedicalEntry(token) || (core ? getMedicalEntry(core) : null);

    if (medical) {
        return {
            input: token,
            category: "medical_word",
            language: inferMedicalLanguage(medical, core || token),
            label_source: "medical_wordlist",
            confidence: null,
            transform_to: "-",
            transform_source: "-",
            normalized_value: processNonWhitespaceToken(token),
            pinyin_segment: "-",
            medical_id: medical.item?.id || null,
            medical_category: medical.item?.category || null,
            thresholdEligible: true,
            model_input: core || token
        };
    }

    const suspiciousFindings = inspectSuspiciousToken(token);
    if (processingResult.isGibberish && suspiciousFindings.length > 0) {
        return {
            input: token,
            category: "gibberish",
            language: "other",
            label_source: "gibberish_rules",
            confidence: null,
            transform_to: "<gibberish>",
            transform_source: "gibberish_rules",
            normalized_value: "<gibberish>",
            pinyin_segment: "-",
            thresholdEligible: false,
            model_input: null
        };
    }

    const normalizedSequenceToken = processNonWhitespaceToken(token);
    const lookupValue = core || normalizedSequenceToken;
    const key = normalizeLookupKey(lookupValue);
    const informalTransform = item.forcedNormalizer || LANGUAGE_RESOURCES.malayNormalizerMap.get(key);
    const isPinyin = LANGUAGE_RESOURCES.pinyinSet.has(key);

    // V22 Rule 1:
    // A Malay informal-normalizer key that is ALSO valid Pinyin (e.g. ni/de)
    // must be resolved by mBERT before any Malay normalization is applied.
    const normalizerPinyinConflict = Boolean(informalTransform && isPinyin);

    if (informalTransform && !normalizerPinyinConflict) {
        return {
            input: token,
            category: "malay_informal_normalization",
            language: "malay",
            label_source: "asrafulsyifaa_malay_normalizer",
            confidence: null,
            transform_to: informalTransform,
            transform_source: "malay_normalizer",
            normalized_value: informalTransform,
            pinyin_segment: "-",
            malay_normalization: "applied_directly",
            thresholdEligible: true,
            model_input: informalTransform
        };
    }

    if (/^[0-9]+$/u.test(token)) {
        return {
            input: token,
            category: "number",
            language: "other",
            label_source: "number_regex",
            confidence: null,
            transform_to: "-",
            transform_source: "-",
            normalized_value: token,
            pinyin_segment: "-",
            thresholdEligible: false,
            model_input: null
        };
    }

    if (/[A-Za-z]/u.test(token)) {
        const isMalay = LANGUAGE_RESOURCES.malayDictionarySet.has(key);
        const dictionaryMatches = [
            ...(isMalay ? ["malay"] : []),
            ...(isPinyin ? ["pinyin"] : []),
            ...(informalTransform ? ["malay_normalizer"] : [])
        ];

        let language = "pending_mbert";
        let category = "mbert_validation_pending";
        let labelSource = "pending_mBERT";

        const forcedAmbiguous = FORCED_MBERT_AMBIGUOUS_WORDS.has(key);

        if (normalizerPinyinConflict) {
            language = "pending_mbert";
            category = "malay_normalizer_pinyin_conflict";
            labelSource = "pending_mBERT";
        } else if (forcedAmbiguous || (isMalay && isPinyin)) {
            language = "pending_mbert";
            category = "dictionary_conflict";
            labelSource = "pending_mBERT";
        } else if (isMalay && !isPinyin) {
            language = "malay";
            category = "-";
            labelSource = "fakhrullah_malay_dictionary";
        } else if (isPinyin && !isMalay) {
            language = "pinyin";
            category = "-";
            labelSource = "guoyunhe_pinyin_list";
        }

        const transform = normalizedSequenceToken !== token ? normalizedSequenceToken : "-";

        return {
            input: token,
            category,
            language,
            label_source: labelSource,
            confidence: null,
            transform_to: transform,
            transform_source: transform === "-" ? "-" : "sequence_normalizer",
            normalized_value: normalizedSequenceToken,
            dictionary_matches: dictionaryMatches,
            mbert_prediction: null,
            mbert_probabilities: null,
            pending_malay_transform: normalizerPinyinConflict ? informalTransform : null,
            normalizer_pinyin_conflict: normalizerPinyinConflict,
            malay_normalization: normalizerPinyinConflict ? "waiting_for_mBERT" : null,
            pinyin_segment: "-",
            pinyin_segmentation_status: null,
            pinyin_segmentation_source: null,
            pinyin_normalization: language === "pinyin" ? "pending_Pinyin2Hanzi_HMM" : null,
            thresholdEligible: true,
            model_input: lookupValue,
            leading_punctuation: punctuationParts.leading,
            trailing_punctuation: punctuationParts.trailing
        };
    }

    return {
        input: token,
        category: "symbol",
        language: "other",
        label_source: "symbol_regex",
        confidence: null,
        transform_to: "-",
        transform_source: "-",
        normalized_value: token,
        pinyin_segment: "-",
        thresholdEligible: false,
        model_input: null
    };
}


async function enrichTokenAnalysisWithModels(tokenAnalysis, originalInput) {
    const contextRows = tokenAnalysis
        .map((item, index) => ({ item, index }))
        .filter(({ item }) => item.thresholdEligible && item.model_input);

    const modelStatus = {
        mBert: {
            requested: contextRows.some(({ item }) => item.language === "pending_mbert"),
            completed: false,
            usedForLabels: 0,
            error: null,
            modelName: MODEL_RUNTIME.mBert.modelName
        },
        pinyin2HanziHmm: {
            requested: false,
            completed: false,
            convertedTokens: 0,
            segmentedTokens: 0,
            pinyinGroups: 0,
            mixedLanguageInput: false,
            error: null,
            modelName: MODEL_RUNTIME.hmm.modelName
        }
    };

    // Preserve a clearly continuous Pinyin sentence. Some syllables such as
    // "yu" and "ke" also exist in Malay/other dictionaries, so an isolated
    // mBERT decision can otherwise split "wo xiang yao yu yue ke yi ma" into
    // several tiny HMM runs. A run of four or more dictionary-confirmed Pinyin
    // tokens is strong sequence evidence; mBERT still decides shorter/mixed
    // spans where the language really is ambiguous.
    const contextualPinyinIndexes = new Set();
    let pinyinCandidateRun = [];
    const flushPinyinCandidateRun = () => {
        if (pinyinCandidateRun.length >= 4) {
            pinyinCandidateRun.forEach((index) => contextualPinyinIndexes.add(index));
        }
        pinyinCandidateRun = [];
    };
    tokenAnalysis.forEach((item, index) => {
        const isPinyinCandidate = item.thresholdEligible &&
            Array.isArray(item.dictionary_matches) &&
            item.dictionary_matches.includes("pinyin");
        if (isPinyinCandidate) pinyinCandidateRun.push(index);
        else flushPinyinCandidateRun();
    });
    flushPinyinCandidateRun();

    if (contextRows.length > 0 && modelStatus.mBert.requested) {
        try {
            const predictions = await requestMBertPredictions(
                contextRows.map(({ item }) => item.model_input),
                originalInput
            );

            contextRows.forEach(({ item, index }, localIndex) => {
                const prediction = predictions[localIndex];
                if (!prediction) return;

                item.mbert_prediction = prediction.language || "other";
                item.mbert_probabilities = prediction.probabilities || null;

                if (item.language !== "pending_mbert") return;

                let resolvedLanguage = prediction.language === "chinese"
                    ? "hanzi"
                    : ["malay", "pinyin", "other"].includes(prediction.language)
                        ? prediction.language
                        : "other";

                const pinyinSequenceOverride = contextualPinyinIndexes.has(index);
                if (pinyinSequenceOverride) resolvedLanguage = "pinyin";

                item.language = resolvedLanguage;
                item.label_source = pinyinSequenceOverride
                    ? "pinyin_sequence_context_after_mBERT"
                    : "mBERT";
                item.confidence = Number.isFinite(Number(prediction.confidence))
                    ? Number(prediction.confidence)
                    : null;

                if (item.normalizer_pinyin_conflict) {
                    item.category = pinyinSequenceOverride
                        ? "malay_normalizer_pinyin_conflict_resolved_by_pinyin_context"
                        : "malay_normalizer_pinyin_conflict_resolved_by_mbert";

                    // V22 Rule 1: only normalize the informal Malay form AFTER
                    // mBERT resolves the ambiguous token as Malay.
                    if (resolvedLanguage === "malay" && item.pending_malay_transform) {
                        item.transform_to = item.pending_malay_transform;
                        item.transform_source = "malay_normalizer_after_mBERT";
                        item.normalized_value = item.pending_malay_transform;
                        item.malay_normalization = "applied_after_mBERT";
                    } else if (resolvedLanguage === "pinyin") {
                        item.malay_normalization = "skipped_mBERT_identified_pinyin";
                    } else {
                        item.malay_normalization = "skipped_mBERT_non_malay";
                    }
                } else {
                    item.category = pinyinSequenceOverride
                        ? "dictionary_conflict_resolved_by_pinyin_context"
                        : item.dictionary_matches?.filter((value) => value !== "malay_normalizer").length > 1
                            ? "dictionary_conflict_resolved_by_mbert"
                            : "mbert_validation";
                }

                item.pinyin_normalization = resolvedLanguage === "pinyin"
                    ? "pending_Pinyin2Hanzi_HMM"
                    : null;
                modelStatus.mBert.usedForLabels += 1;
            });

            modelStatus.mBert.completed = true;
            modelStatus.mBert.modelName = MODEL_RUNTIME.mBert.modelName;
        } catch (error) {
            MODEL_RUNTIME.mBert.lastError = error.message;
            modelStatus.mBert.error = error.message;

            tokenAnalysis.forEach((item) => {
                if (item.language !== "pending_mbert") return;
                item.language = "other";
                item.category = item.normalizer_pinyin_conflict
                    ? "malay_normalizer_pinyin_conflict_unresolved"
                    : item.dictionary_matches?.length > 1
                        ? "dictionary_conflict_unresolved"
                        : "mbert_unavailable_fallback";
                item.label_source = "fallback_other_model_unavailable";
                item.confidence = null;
                if (item.normalizer_pinyin_conflict) {
                    item.malay_normalization = "not_applied_model_unavailable";
                }
            });
        }
    } else {
        modelStatus.mBert.completed = !modelStatus.mBert.requested;
    }

    const pinyinRows = tokenAnalysis
        .map((item, index) => ({ item, index }))
        .filter(({ item }) => item.thresholdEligible && item.language === "pinyin" && item.model_input);

    if (pinyinRows.length > 0) {
        modelStatus.pinyin2HanziHmm.requested = true;

        const nonPinyinEligibleCount = tokenAnalysis.filter(
            (item) => item.thresholdEligible && item.language !== "pinyin"
        ).length;
        modelStatus.pinyin2HanziHmm.mixedLanguageInput = nonPinyinEligibleCount > 0;

        // V22 Rule 2:
        // If mBERT identifies a connected Latin token as Pinyin, segment it
        // with the Guoyunhe Pinyin inventory before HMM decoding.
        pinyinRows.forEach(({ item }) => {
            const segmentation = segmentPinyinForHmm(item.model_input);
            item.pinyin_segmentation_status = segmentation.status;
            item.pinyin_segmentation_source = "guoyunhe_pinyin_list";

            if (segmentation.success) {
                item.pinyin_segment = segmentation.segmentedInput;
                item.hmm_input = segmentation.segmentedInput;
                if (segmentation.status === "segmented") {
                    modelStatus.pinyin2HanziHmm.segmentedTokens += 1;
                    if (item.category === "mbert_validation") {
                        item.category = "mbert_pinyin_segmented";
                    } else if (item.category === "malay_normalizer_pinyin_conflict_resolved_by_mbert") {
                        item.category = "mbert_pinyin_conflict_segmented";
                    }
                }
            } else {
                item.pinyin_segment = "-";
                item.hmm_input = null;
                item.pinyin_normalization = "pinyin_segmentation_failed";
                item.transform_source = item.transform_source === "-"
                    ? "segmentation_failed"
                    : item.transform_source;
            }
        });

        const convertibleRows = pinyinRows.filter(({ item }) => item.hmm_input);

        // V22 Rule 3:
        // In mixed-language input, split out ONLY each contiguous Pinyin span
        // and send that span to HMM. Malay/Other/Hanzi tokens are boundaries.
        const pinyinGroups = [];
        let currentGroup = [];
        convertibleRows.forEach((row) => {
            const previous = currentGroup.at(-1);
            if (!previous || row.index === previous.index + 1) {
                currentGroup.push(row);
            } else {
                pinyinGroups.push(currentGroup);
                currentGroup = [row];
            }
        });
        if (currentGroup.length > 0) pinyinGroups.push(currentGroup);
        modelStatus.pinyin2HanziHmm.pinyinGroups = pinyinGroups.length;

        try {
            for (let groupIndex = 0; groupIndex < pinyinGroups.length; groupIndex += 1) {
                const group = pinyinGroups[groupIndex];
                const groupInput = group.map(({ item }) => item.hmm_input).join(" ");
                const conversions = await requestHmmConversions(
                    group.map(({ item }) => item.hmm_input)
                );

                group.forEach(({ item }, localIndex) => {
                    const conversion = conversions[localIndex];
                    const converted = String(conversion?.output || "").trim();

                    item.pinyin_group = groupIndex + 1;
                    item.pinyin_group_input = groupInput;
                    item.hmm_segmented_input = conversion?.segmented_input || item.hmm_input;

                    if (!converted) {
                        item.pinyin_normalization = conversion?.status === "unknown_pinyin"
                            ? "Pinyin2Hanzi_HMM_unknown_pinyin"
                            : conversion?.status === "segmentation_failed"
                                ? "Pinyin2Hanzi_HMM_segmentation_failed"
                                : "Pinyin2Hanzi_HMM_returned_empty";
                        return;
                    }

                    const hmmModelName = String(
                        conversion?.model_name || MODEL_RUNTIME.hmm.modelName || "Pinyin2Hanzi HMM"
                    );
                    const isMockHmm = MODEL_RUNTIME.mockMode || /mock/iu.test(hmmModelName);
                    const sourcePinyin = String(item.hmm_input || "").trim().toLocaleLowerCase("en");
                    const outputComparable = converted.trim().toLocaleLowerCase("en");
                    const isPassThrough = isMockHmm && outputComparable === sourcePinyin;

                    item.transform_to = converted;
                    item.transform_source = isMockHmm
                        ? "MOCK Pinyin2Hanzi HMM (UI test only)"
                        : "Pinyin2Hanzi HMM";
                    item.normalized_value = `${item.leading_punctuation || ""}${converted}${item.trailing_punctuation || ""}`;
                    item.pinyin_normalization = isPassThrough
                        ? "mock_Pinyin2Hanzi_passthrough"
                        : isMockHmm
                            ? "converted_by_MOCK_Pinyin2Hanzi_HMM"
                            : "converted_by_Pinyin2Hanzi_HMM";
                    item.hmm_model = hmmModelName;
                    if (!isPassThrough && outputComparable !== sourcePinyin) {
                        modelStatus.pinyin2HanziHmm.convertedTokens += 1;
                    }
                });
            }

            modelStatus.pinyin2HanziHmm.completed = true;
            modelStatus.pinyin2HanziHmm.modelName = MODEL_RUNTIME.hmm.modelName;
        } catch (error) {
            MODEL_RUNTIME.hmm.lastError = error.message;
            modelStatus.pinyin2HanziHmm.error = error.message;
            convertibleRows.forEach(({ item }) => {
                item.pinyin_normalization = "Pinyin2Hanzi_HMM_unavailable_token_kept_as_pinyin";
                item.transform_source = item.transform_source === "-"
                    ? "model_unavailable"
                    : item.transform_source;
            });
        }
    } else {
        modelStatus.pinyin2HanziHmm.completed = true;
    }

    return modelStatus;
}


function summarizeLanguageThreshold(input, tokenAnalysis) {
    const eligible = tokenAnalysis.filter((item) => item.thresholdEligible);
    const counts = { malay: 0, chinese: 0, other: 0 };

    eligible.forEach((item) => {
        if (item.language === "malay") counts.malay += 1;
        else if (item.language === "hanzi" || item.language === "pinyin") counts.chinese += 1;
        else counts.other += 1;
    });

    const total = eligible.length;
    const thresholds = total
        ? {
            malay: counts.malay / total,
            chinese: counts.chinese / total,
            other: counts.other / total
        }
        : { malay: 0, chinese: 0, other: 1 };

    let replyRoute = "ask_preference_reply";
    if (thresholds.malay >= 0.7) replyRoute = "malay_reply_conversation";
    else if (thresholds.chinese >= 0.7) replyRoute = "chinese_reply_conversation";

    return {
        input,
        token_count: total,
        counts,
        malay: thresholds.malay,
        chinese: thresholds.chinese,
        other: thresholds.other,
        malay_percent: `${Math.round(thresholds.malay * 100)}%`,
        chinese_percent: `${Math.round(thresholds.chinese * 100)}%`,
        other_percent: `${Math.round(thresholds.other * 100)}%`,
        reply_route: replyRoute
    };
}

async function preprocessUserInput(input) {
    const originalInput = String(input ?? "");
    const sequenceNormalizedText = normalizeSequenceText(originalInput);
    const gibberishResult = detectGibberish(sequenceNormalizedText);
    const baseTokens = tokenizeInputForMonitor(originalInput);
    const mergedTokens = mergeKnownPhrases(baseTokens);

    const preliminary = {
        originalInput,
        sequenceNormalizedText,
        isGibberish: gibberishResult.isGibberish,
        gibberishReasons: gibberishResult.reasons
    };

    const tokenAnalysis = mergedTokens.map((item) => analyzeToken(item, preliminary));
    let modelStatus = {
        mBert: { requested: false, completed: true, usedForLabels: 0, error: null },
        pinyin2HanziHmm: { requested: false, completed: true, convertedTokens: 0, error: null }
    };

    if (!gibberishResult.isGibberish) {
        modelStatus = await enrichTokenAnalysisWithModels(tokenAnalysis, originalInput);
    }

    const finalNormalizedText = gibberishResult.isGibberish
        ? "<gibberish>"
        : joinNormalizedValues(tokenAnalysis.map((item) => item.normalized_value));

    const languageSummary = summarizeLanguageThreshold(originalInput.trim(), tokenAnalysis);
    if (gibberishResult.isGibberish) languageSummary.reply_route = "reject_reply";

    const isHmmConversion = (item) => [
        "converted_by_Pinyin2Hanzi_HMM",
        "converted_by_MOCK_Pinyin2Hanzi_HMM"
    ].includes(item.pinyin_normalization);
    const convertedCount = tokenAnalysis.filter(isHmmConversion).length;
    const segmentedCount = tokenAnalysis.filter(
        (item) => item.pinyin_segmentation_status === "segmented"
    ).length;
    const remainingPinyinCount = tokenAnalysis.filter(
        (item) => item.language === "pinyin" && !isHmmConversion(item)
    ).length;

    return {
        originalInput,
        sequenceNormalizedText,
        candidateNormalizedText: finalNormalizedText,
        normalizedText: finalNormalizedText,
        isGibberish: gibberishResult.isGibberish,
        gibberishReasons: gibberishResult.reasons,
        tokenAnalysis,
        languageSummary,
        modelStatus,
        modelRuntime: modelRuntimeSnapshot(),
        pinyinNormalizationStatus: convertedCount > 0
            ? `${convertedCount} pinyin token(s) converted by Pinyin2Hanzi HMM; ${segmentedCount} connected token(s) segmented; ${remainingPinyinCount} remained pinyin`
            : remainingPinyinCount > 0
                ? `${remainingPinyinCount} pinyin token(s) remained; ${segmentedCount} connected token(s) were segmented before HMM`
                : "No pinyin token required conversion",
        status: gibberishResult.isGibberish
            ? "request_reentry"
            : modelStatus.mBert.error || modelStatus.pinyin2HanziHmm.error
                ? "language_layer_completed_with_model_fallback"
                : "language_layer_completed_with_mBERT_and_Pinyin2Hanzi_HMM"
    };
}

function selectAskPreferenceLanguage(summary) {
    if (summary.chinese > summary.malay) return "zh";
    if (summary.malay > summary.chinese) return "ms";
    if (containsHanzi(summary.input)) return "zh";
    if (replyConversation === "chinese") return "zh";
    if (replyConversation === "malay") return "ms";
    return "ms";
}

async function routeReply(processingResult) {
    const directInput = String(processingResult.originalInput || processingResult.normalizedText || "").trim();
    const identityInput = [
        processingResult.originalInput,
        processingResult.normalizedText,
        processingResult.sequenceNormalizedText
    ].filter(Boolean).join(" ").trim();
    const language = bookingLanguage(processingResult);
    let preclassifiedIntentionDecision = null;

    // Gibberish is a hard safety gate. It must never reach XLM-R, RAG or SQL,
    // even when a statistical model predicts ask_info with high confidence.
    if (processingResult.isGibberish) {
        const rejectLanguage = containsHanzi(processingResult.originalInput) ? "zh" : "ms";
        processingResult.intentionDecision = {
            intention: "gibberish", source: "gibberish_hard_gate", confidence: 1, probabilities: null
        };
        return {
            ...createReplyObject("reject_reply", rejectLanguage),
            route: "reject_reply",
            replyConversation
        };
    }

    // Identity questions must be handled before introduction extraction because
    // Chinese questions such as "我叫什么名字" contain the substring "我叫".
    if (isIdentityRecallQuestion(identityInput)) {
        if (cancellationInteractionActive()) resetCancellationFlow();
        awaitingCheckBookingIc = false;
        processingResult.intentionDecision = {
            intention: "user_identity_recall", source: "conversation_state_rule", confidence: 1, probabilities: null
        };
        rememberConversationLanguage(language);
        return {
            type: "intent_reply", language, id: "user_identity_recall",
            text: conversationState.userName
                ? (language === "zh"
                    ? `记得，您是 ${conversationState.userName}。请问今天有什么可以帮到您？`
                    : `Ya, saya ingat. Nama anda ${conversationState.userName}. Apa yang boleh saya bantu hari ini?`)
                : (language === "zh"
                    ? "您还没有告诉我名字。您可以说“我叫……”来让我记住您。"
                    : "Anda belum memberitahu nama anda. Anda boleh berkata “Nama saya ...” supaya saya boleh mengingatinya."),
            route: "user_identity_recall", replyConversation,
            conversationState: { ...conversationState }
        };
    }

    const introducedName = processingResult.llmIntroducedName || extractIntroducedName(directInput);
    if (introducedName) {
        if (cancellationInteractionActive()) resetCancellationFlow();
        awaitingCheckBookingIc = false;
        const rememberedName = rememberUserName(introducedName);
        processingResult.intentionDecision = {
            intention: "user_intro", source: "conversation_state_rule", confidence: 1, probabilities: null
        };
        rememberConversationLanguage(language);
        return {
            type: "intent_reply", language, id: "user_intro_remembered",
            text: language === "zh"
                ? `你好，${rememberedName}！我记住您了。请问有什么可以帮到您？`
                : `Hai, ${rememberedName}! Saya akan ingat nama anda. Apa yang boleh saya bantu?`,
            route: "user_intro", replyConversation,
            conversationState: { ...conversationState }
        };
    }

    if (awaitingCancellationIc) {
        const icDigits = directInput.replace(/\D/gu, "");
        const looksLikeIcAttempt = /^[\d\s-]+$/u.test(directInput) && icDigits.length > 0;
        if (icDigits.length !== 12 && !looksLikeIcAttempt) {
            const interruption = await classifyStateInterruption(processingResult);
            const interruptedIntent = interruption.intention;
            if (["cancel_booking", "reschedule_booking"].includes(interruptedIntent)) {
                cancellationMode = interruptedIntent === "reschedule_booking" ? "rebook" : "cancel";
                processingResult.intentionDecision = interruption;
                return {
                    type: "cancel_prompt", language, id: "cancellation_mode_updated",
                    text: language === "zh"
                        ? `好的，已切换为${cancellationMode === "rebook" ? "重新预约" : "取消预约"}。请在准备好时输入预约所使用的 12 位 IC；若想问其他问题，可直接输入问题。`
                        : `Baik, aliran telah ditukar kepada ${cancellationMode === "rebook" ? "penjadualan semula" : "pembatalan"}. Masukkan 12 digit IC apabila bersedia, atau terus taip soalan lain.`,
                    route: interruptedIntent, replyConversation
                };
            }
            resetCancellationFlow();
            preclassifiedIntentionDecision = interruption;
        }
    }

    if (awaitingCheckBookingIc) {
        const patientIc = directInput.replace(/[^0-9]/gu, "");
        const looksLikeIcAttempt = /^[\d\s-]+$/u.test(directInput) && patientIc.length > 0;
        if (!/^\d{12}$/u.test(patientIc)) {
            if (!looksLikeIcAttempt) {
                const interruption = await classifyStateInterruption(processingResult);
                if (interruption.intention !== "check_booking") {
                    awaitingCheckBookingIc = false;
                    preclassifiedIntentionDecision = interruption;
                } else {
                    processingResult.intentionDecision = interruption;
                    return {
                        type: "intent_reply", language, id: "check_booking_ic_prompt_repeated",
                        text: language === "zh"
                            ? "为了查询有效预约，请输入预约时使用的12位 IC；如果想问其他问题，也可以直接输入问题。"
                            : "Untuk menyemak janji temu aktif, masukkan 12 digit IC yang digunakan semasa tempahan. Jika anda mahu bertanya perkara lain, terus taip soalan tersebut.",
                        route: "check_booking", replyConversation
                    };
                }
            }
        }
        if (awaitingCheckBookingIc && !/^\d{12}$/u.test(patientIc)) {
            return {
                type: "intent_reply", language, id: "check_booking_ic_invalid",
                text: language === "zh"
                    ? "IC 格式不正确。请输入12位数字；连字符可以省略。"
                    : "Format IC tidak betul. Masukkan 12 digit; tanda sempang boleh ditinggalkan.",
                route: "check_booking", replyConversation
            };
        }
        if (awaitingCheckBookingIc) {
            awaitingCheckBookingIc = false;
            try {
                const result = await bookingApiJson("/appointments/latest", { params: { patient_ic: patientIc } });
                const appointment = result.appointment || {};
                return {
                    type: "intent_reply", language, id: "check_booking_latest_result",
                    text: language === "zh"
                        ? `您最新的有效预约：\n预约号码：${appointment.appointment_id}\n日期：${appointment.appointment_date}\n时间：${appointment.start_time}–${appointment.end_time}\n医生：${appointment.doctor_name}`
                        : `Janji temu aktif terkini anda:\nID: ${appointment.appointment_id}\nTarikh: ${appointment.appointment_date}\nMasa: ${appointment.start_time}–${appointment.end_time}\nDoktor: ${appointment.doctor_name}`,
                    route: "check_booking", replyConversation,
                    retrieval: { source: "SQLite", endpoint: "/appointments/latest", result }
                };
            } catch (error) {
                return {
                    type: "intent_reply", language, id: "check_booking_not_found",
                    text: language === "zh"
                        ? `无法找到该 IC 的有效预约：${error.message}`
                        : `Janji temu aktif untuk IC tersebut tidak ditemui: ${error.message}`,
                    route: "check_booking", replyConversation,
                    retrieval: { source: "SQLite", endpoint: "/appointments/latest", error: error.message }
                };
            }
        }
    }

    if (awaitingAppointmentId) {
        const appointmentId = directInput.toUpperCase().replace(/\s+/gu, "");
        if (!/^A\d{6}$/u.test(appointmentId)) {
            return {
                type: "cancel_id_reject",
                language,
                id: "cancel_id_reject",
                text: language === "zh"
                    ? "预约号码格式不正确。请输入 A 加 6 位数字，例如 A000001。"
                    : "Format ID janji temu tidak betul. Masukkan A diikuti 6 digit, contoh A000001.",
                route: cancellationMode === "rebook" ? "reschedule_booking" : "cancel_booking",
                replyConversation
            };
        }

        awaitingAppointmentId = false;
        pendingCancellationId = appointmentId;
        if (lastBookingContext?.appointmentId === appointmentId && lastBookingContext.patientIc) {
            pendingCancellationIc = lastBookingContext.patientIc;
        }
        // Legacy compatibility branch; the V22 flow verifies IC first.
        // If IC is not already known from this session, ask for it only after user confirms with 1.
        awaitingCancelConfirmation = true;
        return {
            type: "cancel_confirmation",
            language,
            id: "cancel_confirmation",
            text: language === "zh"
                ? (cancellationMode === "rebook"
                    ? `确定重新预约${appointmentId}吗？ 输入1 = 确定，2 = 取消。`
                    : `确定取消${appointmentId}吗？ 输入1 = 确定，2 = 取消。`)
                : `Sahkan ${cancellationMode === "rebook" ? "penjadualan semula" : "pembatalan"} ${appointmentId}: masukkan 1 = Yes, 2 = No.`,
            route: "confirmation",
            replyConversation
        };
    }

    if (awaitingCancellationIc) {
        const digits = directInput.replace(/\D/gu, "");
        if (digits.length !== 12) {
            return {
                type: "cancel_ic_reject",
                language,
                id: "cancel_ic_reject",
                text: language === "zh"
                    ? "IC 格式不正确。请输入完整的 12 位数字，例如 010101101234。"
                    : "Format IC tidak betul. Masukkan 12 digit lengkap, contoh 010101101234.",
                route: cancellationMode === "rebook" ? "reschedule_booking" : "cancel_booking",
                replyConversation
            };
        }
        pendingCancellationIc = formatIcInput(digits);
        try {
            const result = await bookingApiJson("/appointments/latest", {
                params: { patient_ic: pendingCancellationIc }
            });
            const appointment = result.appointment || {};
            pendingCancellationId = appointment.appointment_id;
            awaitingCancellationIc = false;
            const details = language === "zh"
                ? `预约号码：${appointment.appointment_id}\n日期：${appointment.appointment_date}\n时间：${appointment.start_time}–${appointment.end_time}\n医生：${appointment.doctor_name}`
                : `ID: ${appointment.appointment_id}\nTarikh: ${appointment.appointment_date}\nMasa: ${appointment.start_time}–${appointment.end_time}\nDoktor: ${appointment.doctor_name}`;

            if (cancellationMode === "rebook") {
                pendingRebooking = {
                    oldAppointmentId: appointment.appointment_id,
                    patientIc: pendingCancellationIc,
                    appointment,
                    reason: "Patient requested reschedule through chatbot."
                };
                const savedMode = cancellationMode;
                resetCancellationFlow();
                cancellationMode = savedMode;
                return {
                    type: "reschedule_form", language, appointment,
                    text: language === "zh"
                        ? `已找到您原本的预约：\n${details}\n请在预填表格中选择新的医生、日期和时间。确认成功后，旧时段会自动释放。`
                        : `Janji temu asal anda ditemui:\n${details}\nPilih doktor, tarikh dan masa baharu dalam borang yang telah diisi. Slot lama akan dilepaskan selepas pengesahan berjaya.`,
                    route: "reschedule_booking", replyConversation
                };
            }

            awaitingCancelConfirmation = true;
            return {
                type: "cancel_confirmation", language,
                text: language === "zh"
                    ? `请确认是否取消以下预约：\n${details}\n输入 1 = Yes，2 = No。`
                    : `Sahkan pembatalan janji temu berikut:\n${details}\nMasukkan 1 = Yes, 2 = No.`,
                route: "confirmation", replyConversation
            };
        } catch (error) {
            awaitingCancellationIc = false;
            resetCancellationFlow();
            return {
                type: "intent_reply", language,
                text: language === "zh"
                    ? `找不到该 IC 的有效预约：${error.message}`
                    : `Janji temu aktif untuk IC tersebut tidak ditemui: ${error.message}`,
                route: cancellationMode === "rebook" ? "reschedule_booking" : "cancel_booking",
                replyConversation
            };
        }
    }

    if (awaitingCancelConfirmation) {
        if (directInput !== "1" && directInput !== "2") {
            return {
                type: "cancel_confirmation_reject",
                language,
                id: "cancel_confirmation_reject",
                text: language === "zh"
                    ? "请输入 1 = 确定 或 2 = 取消。"
                    : "Masukkan 1 = Yes atau 2 = No.",
                route: "confirmation",
                replyConversation
            };
        }

        if (directInput === "2") {
            resetCancellationFlow();
            return {
                type: "cancel_kept",
                language,
                id: "cancel_kept",
                text: language === "zh"
                    ? "取消操作已停止。有其他问题可以随时找我！=D"
                    : "Pembatalan dihentikan. Jika ada soalan lain, anda boleh hubungi saya bila-bila masa! =D",
                route: "confirmation",
                replyConversation
            };
        }

        const payload = {
            appointmentId: pendingCancellationId,
            patientIc: pendingCancellationIc,
            mode: cancellationMode
        };
        resetCancellationFlow();
        return {
            type: "cancel_submit",
            language,
            id: "cancel_submit",
            payload,
            text: "",
            route: payload.mode === "rebook" ? "reschedule_booking" : "cancel_booking",
            replyConversation
        };
    }

    if (awaitingBookingConfirmation) {
        const answer = directInput.toLocaleLowerCase("en").trim();
        const yesAnswer = /^(?:1|yes|y|ya|ye|boleh|要|是|可以|好|好的)$/iu.test(answer);
        const noAnswer = /^(?:2|no|n|tidak|tak|tak mahu|不要|不|否|不用|不需要)$/iu.test(answer);
        if (yesAnswer) {
            awaitingBookingConfirmation = false;
            return {
                type: "booking_form", language, id: "clinical_booking_confirmed",
                text: language === "zh"
                    ? "好的，请填写以下预约表格。"
                    : "Baik, sila lengkapkan borang janji temu berikut.",
                route: "booking", replyConversation
            };
        }
        if (noAnswer) {
            awaitingBookingConfirmation = false;
            resetClinicalStateForNextBooking();
            return {
                type: "intent_reply", language, id: "clinical_booking_declined",
                text: language === "zh"
                    ? "好的。如有其他需要，请告诉我有什么可以帮到您。"
                    : "Baik. Jika ada keperluan lain, beritahu saya bagaimana saya boleh membantu anda.",
                route: "confirmation", replyConversation
            };
        }
        return {
            type: "intent_reply", language, id: "clinical_booking_confirmation_required",
            text: language === "zh"
                ? "请问您是否要现在预约医生？请输入 Yes 或 No。"
                : "Adakah anda mahu membuat janji temu doktor sekarang? Masukkan Yes atau No.",
            route: "confirmation", replyConversation
        };
    }

    const preferenceCandidate = processingResult.normalizedText.trim();

    if (awaitingLanguagePreference && (preferenceCandidate === "0" || preferenceCandidate === "1")) {
        replyConversation = preferenceCandidate === "0" ? "malay" : "chinese";
        explicitLanguagePreference = replyConversation;
        awaitingLanguagePreference = false;
        const language = replyConversation === "malay" ? "ms" : "zh";
        return {
            ...createReplyObject("preference_confirmation_reply", language),
            route: replyConversation === "malay"
                ? "malay_reply_conversation"
                : "chinese_reply_conversation",
            replyConversation
        };
    }

    // Enforce the original dominant-language requirement before intent routing.
    // English/Other or mixed input below 0.70 asks for Malay/Chinese preference.
    if (!explicitLanguagePreference && processingResult.languageSummary.reply_route === "ask_preference_reply") {
        awaitingLanguagePreference = true;
        const askLanguage = selectAskPreferenceLanguage(processingResult.languageSummary);
        processingResult.intentionDecision = {
            intention: "request_language_preference",
            source: "dominant_language_threshold",
            confidence: Math.max(
                Number(processingResult.languageSummary.malay || 0),
                Number(processingResult.languageSummary.chinese || 0)
            ),
            probabilities: null
        };
        return {
            ...createReplyObject("ask_preference_reply", askLanguage),
            route: "ask_preference_reply",
            replyConversation: null
        };
    }

    // Business intents are evaluated against both the final normalized text and
    // the user's original text. This keeps an explicit booking/cancel/thanks
    // request reachable even when an upstream Pinyin/gibberish flag is stale.
    const intentInput = `${processingResult.normalizedText || ""} ${processingResult.sequenceNormalizedText || ""} ${processingResult.originalInput || ""}`;
    let intentionDecision = preclassifiedIntentionDecision;
    let earlyIntent = intentionDecision?.intention
        || (isDoctorDirectoryFollowUp(directInput) ? "ask_info" : detectKeywordIntent(intentInput));
    if (!intentionDecision && earlyIntent) {
        intentionDecision = { intention: earlyIntent, source: "keyword_rule", confidence: 1, probabilities: null };
    }
    if (!earlyIntent) {
        try {
            intentionDecision = await classifyMainIntention(
                String(processingResult.normalizedText || processingResult.originalInput || "").trim()
            );
            earlyIntent = intentionDecision.intention;
        } catch (error) {
            console.error("XLM-R intention classification failed:", error);
            earlyIntent = "uncertain";
            intentionDecision = { intention: "uncertain", source: "fallback", confidence: null, probabilities: null };
        }
    }
    processingResult.intentionDecision = intentionDecision;

    // If the chatbot explicitly asked for missing clinical information, treat the next
    // ordinary answer as part of the description workflow unless the user clearly
    // starts another business action (booking/cancel/reschedule/thanks).
    if (lastClinicalAction && ["ask_symptoms", "ask_duration", "ask_severity"].includes(lastClinicalAction)
        && earlyIntent === "uncertain") {
        earlyIntent = "description";
    }

    const greetingLanguage = detectGreetingLanguage(processingResult.normalizedText);
    if (greetingLanguage) {
        processingResult.intentionDecision = {
            intention: "greeting", source: "keyword_rule", confidence: 1, probabilities: null
        };
        replyConversation = greetingLanguage === "zh" ? "chinese" : "malay";
        const greeting = createReplyObject("greeting_reply2", greetingLanguage);
        if (conversationState.userName) {
            greeting.text = greetingLanguage === "zh"
                ? `你好，${conversationState.userName}！欢迎回来，请问有什么可以帮到您？`
                : `Hai, ${conversationState.userName}! Selamat kembali. Apa yang boleh saya bantu?`;
        }
        return {
            ...greeting,
            route: replyConversation === "malay"
                ? "malay_reply_conversation"
                : "chinese_reply_conversation",
            replyConversation
        };
    }

    const intent = earlyIntent;
    conversationState.lastIntent = intent || "unknown";
    saveConversationState();
    if (["thanks", "goodbye", "greeting", "user_intro", "cancel_booking", "reschedule_booking", "booking", "check_booking", "check_availability_query", "description", "ask_info", "confirmation", "uncertain"].includes(intent)) {
        rememberConversationLanguage(language);
    }

    if (intent === "user_intro") {
        const name = rememberUserName(extractIntroducedName(directInput));
        return {
            type: "intent_reply", language, id: "user_intro_greeting",
            text: language === "zh"
                ? `你好，${name || conversationState.userName || "您好"}，我会记住您。有什么可以帮到您？`
                : `Hai, ${name || conversationState.userName || "anda"}. Saya akan mengingati anda. Apa yang boleh saya bantu?`,
            route: "user_intro", replyConversation,
            conversationState: { ...conversationState }
        };
    }

    if (intent === "goodbye") {
        return {
            type: "intent_reply", language, id: "goodbye_rule",
            text: language === "zh" ? "再见，祝您身体健康！需要帮助时欢迎再来。" : "Jumpa lagi. Semoga anda sentiasa sihat!",
            route: "goodbye", replyConversation
        };
    }

    if (intent === "thanks") {
        if (!ratingOffered) {
            ratingOffered = true;
            return {
                type: "thanks",
                language,
                id: "thanks_rating_once",
                text: language === "zh"
                    ? "不客气！希望这能帮助到您。也欢迎您为 Klinik Chong 的服务评分。"
                    : "Sama-sama! Semoga ini membantu anda. Anda juga dialu-alukan untuk menilai perkhidmatan Klinik Chong.",
                showRating: true,
                route: "thanks",
                replyConversation
            };
        }
        return {
            type: "intent_reply",
            language,
            id: "thanks_after_rating_offer",
            text: language === "zh"
                ? (ratingSubmitted ? "不客气！希望下次再使用 Klinik Chong Booking Chatbot！" : "不客气！希望这能帮助到您！")
                : (ratingSubmitted ? "Sama-sama! Semoga anda menggunakan Klinik Chong Booking Chatbot lagi pada masa akan datang!" : "Sama-sama! Semoga ini membantu anda!"),
            route: "thanks",
            replyConversation
        };
    }

    if (intent === "cancel_booking" || intent === "reschedule_booking") {
        resetCancellationFlow();
        cancellationMode = intent === "reschedule_booking" ? "rebook" : "cancel";
        if (intent === "reschedule_booking") recentCancelledAppointment = null;
        awaitingCancellationIc = true;
        return {
            type: "cancel_prompt",
            language,
            id: intent === "reschedule_booking" ? "reschedule_keyword" : "cancel_keyword",
            text: language === "zh"
                ? (intent === "reschedule_booking"
                    ? "为了安全读取并修改您原本的预约，请输入预约时使用的 12 位 IC；连字符可省略。如果您想先问其他问题，可以直接输入问题。"
                    : "为了安全查找您自己的预约，请输入预约时使用的 12 位 IC；连字符可省略。如果您想先问其他问题，可以直接输入问题。")
                : (intent === "reschedule_booking"
                    ? "Untuk membaca dan mengubah janji temu anda dengan selamat, masukkan 12 digit IC yang digunakan semasa tempahan. Jika mahu bertanya perkara lain dahulu, terus taip soalan anda."
                    : "Untuk mencari janji temu anda dengan selamat, masukkan 12 digit IC yang digunakan semasa tempahan. Jika mahu bertanya perkara lain dahulu, terus taip soalan anda."),
            route: intent,
            replyConversation
        };
    }

    if (intent === "booking") {
        // A booking request may also contain useful clinical details (for example,
        // "我咳嗽三天了想预约"). Extract them before checking prerequisites.
        try {
            const extracted = await extractClinicalFromUser(directInput);
            const containsClinicalDetail = Boolean(
                (Array.isArray(extracted?.symptoms) && extracted.symptoms.length)
                || extracted?.duration
                || (Number.isInteger(Number(extracted?.severity)) && Number(extracted.severity) >= 0 && Number(extracted.severity) <= 10)
            );
            mergeClinicalExtraction(extracted, containsClinicalDetail ? directInput : "");
        } catch (error) {
            console.warn("Clinical details inside booking request could not be extracted:", error);
        }
        const missing = clinicalMissingFields();
        if (missing.length) {
            pendingBookingAfterClinical = true;
            const text = clinicalPromptReply(language, missing);
            return {
                type: missing[0] === "severity" ? "clinical_severity" : "intent_reply",
                language,
                id: `booking_requires_${missing[0]}`,
                text: language === "zh"
                    ? `在显示预约表格前，我需要先完成基本症状资料。${text}`
                    : `Sebelum borang janji temu dipaparkan, saya perlu melengkapkan maklumat simptom asas terlebih dahulu. ${text}`,
                route: "booking",
                replyConversation
            };
        }

        if (Number(clinicalState.severity) >= 9) {
            pendingBookingAfterClinical = true;
            return {
                type: "emergency_severity",
                language,
                severity: Number(clinicalState.severity),
                id: "booking_severity_9_10_override",
                text: language === "zh"
                    ? "您报告的严重程度为 9–10，情况十分严重，不建议继续普通 chatbot 预约流程。"
                    : "Tahap keterukan anda ialah 9–10. Keadaan ini sangat serius dan aliran tempahan chatbot biasa tidak disarankan.",
                route: "emergency_override",
                replyConversation
            };
        }

        pendingBookingAfterClinical = false;
        try {
            if (!clinicalState.clinicalNote) await refreshClinicalNote();
        } catch (error) {
            console.error("Clinical note generation before booking failed:", error);
        }

        if (recentCancelledAppointment && !pendingRebooking) {
            pendingRebooking = {
                oldAppointmentId: recentCancelledAppointment.oldAppointmentId,
                patientIc: recentCancelledAppointment.patientIc,
                reason: recentCancelledAppointment.reason
            };
        }

        return {
            type: "booking_form",
            language,
            id: pendingRebooking ? "rebooking_form_ready" : "booking_form_ready",
            text: language === "zh"
                ? (pendingRebooking
                    ? "原预约已经取消。您的症状资料也已完成，请填写以下预约表格来选择新的预约。"
                    : "您的症状、持续时间和严重程度都已记录。请填写以下预约表格以预约您的看诊时间。")
                : (pendingRebooking
                    ? "Janji temu asal telah dibatalkan dan maklumat simptom anda telah lengkap. Sila isi borang di bawah untuk memilih janji temu baharu."
                    : "Simptom, tempoh dan tahap keterukan anda telah direkodkan. Sila isi borang janji temu di bawah untuk membuat tempahan."),
            route: pendingRebooking ? "reschedule_booking" : "booking",
            replyConversation
        };
    }

    if (intent === "description") {
        return await processDescriptionIntent(processingResult, language);
    }

    if (intent === "ask_info") {
        return await processAskInfoIntent(processingResult, language);
    }

    if (intent === "check_booking") {
        awaitingCheckBookingIc = true;
        return {
            type: "intent_reply", language, id: "check_booking_ic_prompt",
            text: language === "zh"
                ? "为了查询您最新的有效预约，请输入预约时使用的12位 IC；连字符可以省略。"
                : "Untuk menyemak janji temu aktif terkini, masukkan 12 digit IC yang digunakan semasa tempahan; tanda sempang boleh ditinggalkan.",
            route: "check_booking", replyConversation,
            retrieval: { source: "SQLite", status: "awaiting_patient_ic" }
        };
    }

    if (intent === "check_availability_query") {
        const routed = await processAskInfoIntent(processingResult, language);
        routed.route = intent;
        routed.id = `${intent}_database_router`;
        return routed;
    }

    if (intent === "confirmation") {
        const text = confirmationReply(language);
        return {
            type: lastClinicalAction === "ask_severity" ? "clinical_severity" : "intent_reply",
            language,
            id: "confirmation_contextual",
            text,
            route: "confirmation",
            replyConversation
        };
    }

    if (intent === "unrelated") {
        return {
            type: "intent_reply",
            language,
            id: "unrelated_scope",
            text: language === "zh"
                ? "抱歉，这个问题超出了我的服务范围。我只能协助 Klinik Chong 的医疗看诊资讯、症状资料、预约、取消预约和重新预约相关问题。"
                : "Maaf, soalan itu di luar skop perkhidmatan saya. Saya hanya boleh membantu dengan maklumat perubatan Klinik Chong, simptom, tempahan, pembatalan dan penjadualan semula janji temu.",
            route: "unrelated",
            replyConversation
        };
    }

    if (intent === "uncertain") {
        return {
            type: "intent_reply",
            language,
            id: "uncertain_clarify",
            text: language === "zh"
                ? "我还不确定您的意思。您可以再说明一点吗？例如您想询问诊所资讯、描述症状、预约、取消预约，还是重新预约？"
                : "Saya masih belum pasti maksud anda. Boleh jelaskan sedikit lagi? Contohnya, adakah anda mahu bertanya maklumat klinik, menerangkan simptom, membuat tempahan, membatalkan atau menjadualkan semula janji temu?",
            route: "uncertain",
            replyConversation
        };
    }

    const route = processingResult.languageSummary.reply_route;

    if (route === "malay_reply_conversation") {
        replyConversation = "malay";
        awaitingLanguagePreference = false;
        return {
            type: "processing_result",
            language: "ms",
            id: "ready_for_next_layer_ms",
            text: processingResult.normalizedText,
            route,
            replyConversation
        };
    }

    if (route === "chinese_reply_conversation") {
        replyConversation = "chinese";
        awaitingLanguagePreference = false;
        return {
            type: "processing_result",
            language: "zh",
            id: "ready_for_next_layer_zh",
            text: processingResult.normalizedText,
            route,
            replyConversation
        };
    }

    if (explicitLanguagePreference) {
        replyConversation = explicitLanguagePreference;
        const language = replyConversation === "malay" ? "ms" : "zh";
        return {
            type: "processing_result",
            language,
            id: "ready_for_next_layer_explicit_preference",
            text: processingResult.normalizedText,
            route: replyConversation === "malay"
                ? "malay_reply_conversation"
                : "chinese_reply_conversation",
            replyConversation,
            explicitPreferenceUsed: true
        };
    }

    awaitingLanguagePreference = true;
    const askLanguage = selectAskPreferenceLanguage(processingResult.languageSummary);
    return {
        ...createReplyObject("ask_preference_reply", askLanguage),
        route: "ask_preference_reply",
        replyConversation: null
    };
}

function displayRoutedReply(reply, processingResult) {
    if (reply.type === "reschedule_form") {
        if (reply.text) createBotMessage(reply.text);
        window.setTimeout(() => showBookingForm(reply.language, {
            rescheduleAppointment: reply.appointment
        }), 250);
        return;
    }
    if (reply.type === "booking_form") {
        if (reply.text) createBotMessage(reply.text);
        window.setTimeout(() => showBookingForm(reply.language), 250);
        return;
    }
    if (reply.type === "clinical_severity") {
        if (reply.text) createBotMessage(reply.text);
        showSeverityCard(reply.language);
        return;
    }
    if (reply.type === "emergency_severity") {
        if (reply.text) createBotMessage(reply.text);
        showEmergencyAlertCard({
            language: reply.language,
            severity: reply.severity,
            onCompleted: async () => {
                try {
                    if (clinicalIsComplete()) await refreshClinicalNote();
                } catch (error) {
                    console.error("Clinical note generation after emergency call failed:", error);
                }
                if (pendingBookingAfterClinical && clinicalIsComplete()) {
                    pendingBookingAfterClinical = false;
                    createBotMessage(reply.language === "zh"
                        ? "紧急通知已完成。若您仍需一般预约，可继续填写以下表格。"
                        : "Notifikasi kecemasan selesai. Jika anda masih memerlukan janji temu biasa, teruskan dengan borang berikut.");
                    window.setTimeout(() => showBookingForm(reply.language), 250);
                } else {
                    createBotMessage(reply.language === "zh"
                        ? "如仍有生命危险，请立即拨打 999。您也可以继续输入其他问题。"
                        : "Jika masih mengancam nyawa, hubungi 999 segera. Anda juga boleh meneruskan dengan soalan lain.");
                }
            }
        });
        return;
    }
    if (reply.type === "thanks") {
        if (reply.text) createBotMessage(reply.text);
        if (reply.showRating) showRatingCard(reply.language);
        return;
    }
    if (reply.type === "cancel_submit") { cancelAppointment(reply.payload, reply.language); return; }
    if (reply.type === "processing_result") {
        createProcessingResultMessage(processingResult, reply);
        return;
    }
    createBotMessage(reply.text);
}

async function sendMessage(options = {}) {
    const rawUserMessage = typeof options.rawUserMessage === "string"
        ? options.rawUserMessage
        : messageInput.value;
    if (rawUserMessage.trim() === "" || (sendButton.disabled && !options.resume)) return;

    const originalDisplayInput = rawUserMessage.trim();
    if (!options.skipUserBubble) createUserMessage(originalDisplayInput);

    messageInput.value = "";
    setComposerDisabled(true);

    if (!options.skipEmergency) {
        const isEmergency = await detectEmergencyBeforeIntent(originalDisplayInput);
        if (isEmergency) {
            setComposerDisabled(false);
            showEmergencyAlertCard({
                userInput: originalDisplayInput,
                onCompleted: () => sendMessage({
                    rawUserMessage,
                    skipUserBubble: true,
                    skipEmergency: true,
                    resume: true
                })
            });
            return;
        }
    }

    const typingIndicator = showTypingIndicator();
    const processingStartedAt = Date.now();

    try {
        const processingResult = await preprocessUserInput(rawUserMessage);
        processingResult.llmIntroducedName = processingResult.isGibberish
            ? ""
            : await updateConversationStateWithLlm(
                rawUserMessage,
                processingResult.normalizedText
            );
        const reply = await applyEmotionAwareReply(
            await routeReply(processingResult),
            rawUserMessage
        );

        const turn = reserveMonitorTurn({
            originalInput: rawUserMessage,
            displayedInput: originalDisplayInput,
            normalizedInput: processingResult.normalizedText,
            candidateNormalizedInput: processingResult.candidateNormalizedText,
            processingStatus: processingResult.status,
            isGibberish: processingResult.isGibberish,
            gibberishReasons: processingResult.gibberishReasons,
            detectedLanguage: reply.language,
            detectedIntent: processingResult.isGibberish
                ? "gibberish"
                : reply.type === "greeting_reply2"
                    ? "greeting"
                    : reply.type === "ask_preference_reply"
                        ? "request_language_preference"
                        : (reply.route || "unknown"),
            intentionDecision: processingResult.intentionDecision || null,
            detectedEmotion: reply.emotion || reply.rag?.emotion || detectTurnEmotion(rawUserMessage),
            retrievalTrace: reply.rag?.retrieval || reply.retrieval || null,
            retrievedContext: reply.rag?.compiled_context || null,
            tokenAnalysis: processingResult.tokenAnalysis,
            languageSummary: processingResult.languageSummary,
            replyRoute: reply.route,
            replyConversation: reply.replyConversation,
            pinyinNormalizationStatus: processingResult.pinyinNormalizationStatus,
            modelStatus: processingResult.modelStatus,
            modelRuntime: processingResult.modelRuntime,
            layerStatus: {
                sequenceTextProcessing: "completed",
                informalMalayNormalization: "completed",
                dictionaryLanguageDetection: "completed",
                mBertLanguageDetection: processingResult.modelStatus.mBert,
                pinyinSegmentation: {
                    source: "guoyunhe_pinyin_list",
                    segmentedTokens: processingResult.modelStatus.pinyin2HanziHmm.segmentedTokens || 0,
                    pinyinGroups: processingResult.modelStatus.pinyin2HanziHmm.pinyinGroups || 0,
                    mixedLanguageInput: Boolean(processingResult.modelStatus.pinyin2HanziHmm.mixedLanguageInput)
                },
                pinyin2HanziHmm: processingResult.modelStatus.pinyin2HanziHmm
            },
            reply: {
                type: reply.type,
                language: reply.language,
                id: reply.id,
                route: reply.route,
                replyConversation: reply.replyConversation,
                text: null
            }
        });

        const elapsed = Date.now() - processingStartedAt;
        if (elapsed < 350) {
            await new Promise((resolve) => window.setTimeout(resolve, 350 - elapsed));
        }

        typingIndicator.remove();
        displayRoutedReply(reply, processingResult);
        appendConversationHistory("user", rawUserMessage);
        appendConversationHistory("assistant", reply.text || "");
        finalizeMonitorTurn(turn, reply);
    } catch (error) {
        console.error("Version 22 processing failed:", error);
        typingIndicator.remove();
        createBotMessage("Processing failed. Please check the model server and try again. 🤔");
    } finally {
        setComposerDisabled(false);
    }
}

function insertEmojiIntoInput(emoji) {
    if (!emoji) return;
    const currentValue = messageInput.value;
    const selectionStart = messageInput.selectionStart ?? currentValue.length;
    const selectionEnd = messageInput.selectionEnd ?? currentValue.length;
    const maxLength = Number(messageInput.maxLength) || Infinity;
    const updatedValue = currentValue.slice(0, selectionStart) + emoji + currentValue.slice(selectionEnd);

    if (updatedValue.length > maxLength) {
        messageInput.focus();
        return;
    }

    messageInput.value = updatedValue;
    const newCursorPosition = selectionStart + emoji.length;
    messageInput.setSelectionRange(newCursorPosition, newCursorPosition);
    messageInput.dispatchEvent(new Event("input", { bubbles: true }));
    messageInput.focus();
    if (emojiPicker) emojiPicker.open = false;
}

emojiButtons.forEach((button) => {
    button.addEventListener("click", () => {
        const emoji = button.dataset.emoji || button.querySelector("span")?.textContent.trim();
        insertEmojiIntoInput(emoji);
    });
});

emojiCloseButton?.addEventListener("click", () => {
    if (emojiPicker) emojiPicker.open = false;
    messageInput?.focus();
});

developerViewButton?.addEventListener("click", showDeveloperView);
footerDeveloperButton?.addEventListener("click", showDeveloperView);
updateFooterClock();
window.setInterval(updateFooterClock, 1000);
developerBackButton?.addEventListener("click", showChatbotView);

window.addEventListener("message", (event) => {
    if (event.data?.type === "show-klinik-chatbot") showChatbotView();
});

sendButton.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

window.addEventListener("DOMContentLoaded", async () => {
    await Promise.all([
        loadReplyData(),
        loadLanguageResources(),
        checkModelBackend()
    ]);
    const greetingReply = getInitialGreetingReply();
    initializeMonitorSession(greetingReply);
    window.setTimeout(() => replyWithTyping(greetingReply.text, BOT_REPLY_DELAY), 500);
});

window.KlinikChongViewSwitcher = {
    showDeveloper: showDeveloperView,
    showChatbot: showChatbotView
};

window.KlinikChongTextProcessor = {
    loadLanguageResources,
    checkModelBackend,
    preprocessUserInput,
    normalizeSequenceText,
    detectGibberish,
    summarizeLanguageThreshold,
    segmentConnectedPinyinWord,
    segmentPinyinForHmm,
    routeReply,
    resources: LANGUAGE_RESOURCES,
    modelRuntime: MODEL_RUNTIME,
    getReplyConversation: () => replyConversation,
    getExplicitLanguagePreference: () => explicitLanguagePreference
};
