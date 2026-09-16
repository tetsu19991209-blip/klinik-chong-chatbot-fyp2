// Klinik Chong Chatbot - Version 22 model API configuration
window.KLINIK_CHONG_MODEL_CONFIG = {
  // The frontend and model/developer API are served by the same V22 server.
  modelApiBase: "",
  // Booking traffic uses the same-origin proxy. RunChatbot supplies the current
  // Cloudflare target to model_server_ver22.py through an environment variable.
  bookingApiBase: "",
  // The same-origin proxy injects the runtime key; do not store it in frontend code.
  developerApiKey: "",
  endpoints: {
    health: "/api/health",
    languageDetect: "/api/language-detect",
    pinyinToHanzi: "/api/pinyin-to-hanzi",
    intentionClassify: "/api/intention-classify",
    ragQuery: "/api/rag-query",
    conversationState: "/api/conversation-state",
    emotionAdjust: "/api/emotion-adjust",
    clinicalExtract: "/api/clinical/extract",
    clinicalNote: "/api/clinical/note",
    clinicalRating: "/api/clinical/rating"
  },
  bookingEndpoints: {
    health: "/health",
    doctors: "/doctors",
    availableSlots: "/available-slots",
    patients: "/patients",
    appointments: "/appointments",
    completeRebooking: "/appointments/complete-rebooking"
  },
  developerEndpoints: {
    databaseSchema: "/developer/database/schema",
    doctorLeave: "/developer/database/doctor-leave"
  },
  requestTimeoutMs: 300000,
  bookingRequestTimeoutMs: 30000,
  mBertName: "mBert_language_detect_model/best_model",
  hmmName: "pinyin2hanzi_hmm_model",
  xlmrIntentName: "models/xlmr"
};
