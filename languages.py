"""languages.py — All 22 official Indian language configs."""

LANGUAGES = {
    "hindi":     {"native":"हिन्दी",  "script":"devanagari",   "easyocr":["hi","en"], "tesseract":"hin","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"medium"},
    "bengali":   {"native":"বাংলা",   "script":"bengali",      "easyocr":["bn","en"], "tesseract":"ben","unicode":(0x0980,0x09FF),"direction":"LTR","shirorekha":False,"difficulty":"medium"},
    "tamil":     {"native":"தமிழ்",  "script":"tamil",        "easyocr":["ta","en"], "tesseract":"tam","unicode":(0x0B80,0x0BFF),"direction":"LTR","shirorekha":False,"difficulty":"hard"},
    "telugu":    {"native":"తెలుగు", "script":"telugu",       "easyocr":["te","en"], "tesseract":"tel","unicode":(0x0C00,0x0C7F),"direction":"LTR","shirorekha":False,"difficulty":"hard"},
    "kannada":   {"native":"ಕನ್ನಡ", "script":"kannada",      "easyocr":["kn","en"], "tesseract":"kan","unicode":(0x0C80,0x0CFF),"direction":"LTR","shirorekha":False,"difficulty":"hard"},
    "malayalam": {"native":"മലയാളം", "script":"malayalam",    "easyocr":["ml","en"], "tesseract":"mal","unicode":(0x0D00,0x0D7F),"direction":"LTR","shirorekha":False,"difficulty":"very_hard"},
    "punjabi":   {"native":"ਪੰਜਾਬੀ", "script":"gurmukhi",    "easyocr":["pa","en"], "tesseract":"pan","unicode":(0x0A00,0x0A7F),"direction":"LTR","shirorekha":False,"difficulty":"medium"},
    "gujarati":  {"native":"ગુજરાતી","script":"gujarati",     "easyocr":["gu","en"], "tesseract":"guj","unicode":(0x0A80,0x0AFF),"direction":"LTR","shirorekha":False,"difficulty":"medium"},
    "odia":      {"native":"ଓଡ଼ିଆ",   "script":"odia",         "easyocr":None,        "tesseract":"ori","unicode":(0x0B00,0x0B7F),"direction":"LTR","shirorekha":False,"difficulty":"hard"},
    "marathi":   {"native":"मराठी",  "script":"devanagari",   "easyocr":["mr","en"], "tesseract":"mar","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"medium"},
    "urdu":      {"native":"اردو",   "script":"perso_arabic", "easyocr":["ur","en"], "tesseract":"urd","unicode":(0x0600,0x06FF),"direction":"RTL","shirorekha":False,"difficulty":"very_hard"},
    "assamese":  {"native":"অসমীয়া","script":"bengali",      "easyocr":None,        "tesseract":"asm","unicode":(0x0980,0x09FF),"direction":"LTR","shirorekha":False,"difficulty":"medium"},
    "maithili":  {"native":"मैथिली", "script":"devanagari",   "easyocr":None,        "tesseract":"mai","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"medium"},
    "sanskrit":  {"native":"संस्कृत","script":"devanagari",   "easyocr":None,        "tesseract":"san","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"hard"},
    "nepali":    {"native":"नेपाली", "script":"devanagari",   "easyocr":["ne","en"], "tesseract":"nep","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"medium"},
    "dogri":     {"native":"डोगरी",  "script":"devanagari",   "easyocr":None,        "tesseract":"dog","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"hard"},
    "konkani":   {"native":"कोंकणी", "script":"devanagari",   "easyocr":None,        "tesseract":"kok","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"hard"},
    "bodo":      {"native":"বোড়ো",   "script":"devanagari",   "easyocr":None,        "tesseract":"brx","unicode":(0x0900,0x097F),"direction":"LTR","shirorekha":True, "difficulty":"hard"},
    "sindhi":    {"native":"سنڌي",   "script":"perso_arabic", "easyocr":None,        "tesseract":"snd","unicode":(0x0600,0x06FF),"direction":"RTL","shirorekha":False,"difficulty":"very_hard"},
    "kashmiri":  {"native":"كشميري","script":"perso_arabic",  "easyocr":None,        "tesseract":"kas","unicode":(0x0600,0x06FF),"direction":"RTL","shirorekha":False,"difficulty":"very_hard"},
    "manipuri":  {"native":"মণিপুরী","script":"meitei_mayek", "easyocr":None,        "tesseract":"mni","unicode":(0xABC0,0xABFF),"direction":"LTR","shirorekha":False,"difficulty":"very_hard"},
    "santali":   {"native":"ᱥᱟᱱᱛᱟᱲᱤ","script":"ol_chiki",  "easyocr":None,        "tesseract":"sat","unicode":(0x1C50,0x1C7F),"direction":"LTR","shirorekha":False,"difficulty":"very_hard"},
}

ALL_LANG_NAMES = sorted(LANGUAGES.keys())
RTL_LANGUAGES  = {k for k,v in LANGUAGES.items() if v["direction"]=="RTL"}

def get(name):             return LANGUAGES.get(name.lower(), {})
def is_rtl(name):          return get(name).get("direction") == "RTL"
def needs_shirorekha(name):return get(name).get("shirorekha", False)
def easyocr_codes(name):   return get(name).get("easyocr") or ["en"]
def tesseract_code(name):  return get(name).get("tesseract", "eng")
