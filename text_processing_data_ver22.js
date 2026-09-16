// Klinik Chong Chatbot - Version 22 text-processing configuration
// Exact canonical filenames in data/. Numeric upload suffixes such as (1) are intentionally excluded.
window.KLINIK_CHONG_TEXT_DATA = {
  emoticons: window.KLINIK_CHONG_RESOURCE_DATA?.emoticons?.emoticons || [],
  resourcePaths: {
    emoticons: "data/emoticons.json",
    medicalTerms: "data/medical_terms_wordlist.json",
    malayNormalizer: "data/asrafulsyifaa_malay_normalizer.json",
    pinyinList: "data/guoyunhe_pinyin_list.json",
    malayDictionary: "data/fakhrullah_malay_dictionary.dic"
  }
};

