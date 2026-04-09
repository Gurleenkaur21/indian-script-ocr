

import os
import sys
import time
import urllib.request
import zipfile
import shutil
import json

# ── Folder setup ─────────────────────────────────────────────────
BASE = "data/raw"
LANGS = [
    "hindi","bengali","tamil","telugu","kannada","malayalam",
    "punjabi","gujarati","odia","marathi","urdu","assamese",
    "maithili","sanskrit","nepali","dogri","konkani","bodo",
    "sindhi","kashmiri","manipuri","santali"
]
for lang in LANGS:
    os.makedirs(f"{BASE}/{lang}/images", exist_ok=True)

print("="*60)
print("  INDIAN OCR — Dataset Downloader")
print("  All 22 Languages — Image Datasets")
print("="*60)
print()

# ── Helper functions ──────────────────────────────────────────────
def show(msg): print(f"  {msg}")
def ok(msg):   print(f"  ✓  {msg}")
def err(msg):  print(f"  ✗  {msg}")
def head(msg): print(f"\n{'─'*55}\n  {msg}\n{'─'*55}")

def count_images(lang):
    folder = f"{BASE}/{lang}/images"
    return len([f for f in os.listdir(folder)
                if f.lower().endswith((".png",".jpg",".jpeg"))])

def save_labels(lang, records):
    import pandas as pd
    df = pd.DataFrame(records, columns=["filename","text"])
    df.to_csv(f"{BASE}/{lang}/labels.csv", index=False, encoding="utf-8")

# ═══════════════════════════════════════════════════════════════
# PART 1 — KAGGLE DATASETS
# ═══════════════════════════════════════════════════════════════
head("PART 1 — Kaggle Image Datasets")
show("Checking if Kaggle API is set up...")

KAGGLE_DATASETS = {
    "hindi":   [
        ("rishianand/devanagari-character-set",   "devanagari_chars"),
        ("aneesh2312/hindi-handwritten-word-recognition", "hindi_words"),
    ],
    "bengali": [
        ("ayanfazlul/bangla-handwritten-character","bengali_chars"),
    ],
    "tamil":   [
        ("ashwin2610/tamil-handwriting",           "tamil_chars"),
    ],
    "kannada": [
        ("vivovinco/kannada-mnist",                "kannada_digits"),
    ],
    "punjabi": [
        ("gurpreetsingh99/gurmukhi-characters",    "punjabi_chars"),
    ],
    "telugu":  [
        ("nageshsingh/iiit5k",                     "telugu_words"),
    ],
}

kaggle_ok = False
try:
    import kaggle
    kaggle_ok = True
    ok("Kaggle API found")
except ImportError:
    err("Kaggle not installed — run: pip install kaggle")

if not kaggle_ok:
    show("Skipping Kaggle downloads.")
    show("To enable: pip install kaggle")
    show("Then: set up ~/.kaggle/kaggle.json")
else:
    for lang, datasets in KAGGLE_DATASETS.items():
        for slug, name in datasets:
            show(f"Downloading {lang} — {name}...")
            out = f"{BASE}/{lang}"
            try:
                os.system(f"kaggle datasets download -d {slug} --path {out} -q")
                # Unzip
                for f in os.listdir(out):
                    if f.endswith(".zip"):
                        with zipfile.ZipFile(f"{out}/{f}") as z:
                            z.extractall(f"{out}/{name}_extracted")
                        os.remove(f"{out}/{f}")
                ok(f"{lang} — {name} downloaded")
            except Exception as e:
                err(f"{lang} — {name} failed: {e}")

# ═══════════════════════════════════════════════════════════════
# PART 2 — AI4BHARAT (HuggingFace)
# ═══════════════════════════════════════════════════════════════
head("PART 2 — AI4Bharat IndicOCR (Printed Text Images)")
show("Checking HuggingFace datasets library...")

HF_LANGS = {
    "hindi":     "hi",
    "bengali":   "bn",
    "tamil":     "ta",
    "telugu":    "te",
    "kannada":   "kn",
    "malayalam": "ml",
    "punjabi":   "pa",
    "gujarati":  "gu",
    "odia":      "or",
    "marathi":   "mr",
    "urdu":      "ur",
    "assamese":  "as",
    "maithili":  "mai",
}

hf_ok = False
try:
    from datasets import load_dataset
    hf_ok = True
    ok("HuggingFace datasets found")
except ImportError:
    err("datasets not installed — run: pip install datasets")

if not hf_ok:
    show("Skipping AI4Bharat downloads.")
    show("To enable: pip install datasets")
else:
    for lang, code in HF_LANGS.items():
        show(f"Downloading AI4Bharat {lang} ({code})...")
        img_dir = f"{BASE}/{lang}/images"
        try:
            from PIL import Image as PILImage
            ds = load_dataset("ai4bharat/IndicOCR", code,
                              split="train", trust_remote_code=True)
            records = []
            limit = min(len(ds), 5000)  # max 5000 per language
            for i in range(limit):
                sample = ds[i]
                fname  = f"ai4b_{lang}_{i:06d}.png"
                path   = os.path.join(img_dir, fname)
                img    = sample["image"]
                if not isinstance(img, PILImage.Image):
                    img = PILImage.fromarray(img)
                img.save(path)
                records.append([fname, sample["text"]])
                if (i+1) % 1000 == 0:
                    show(f"  {lang}: {i+1}/{limit} images saved")
            save_labels(lang, records)
            ok(f"{lang}: {len(records)} images saved")
        except Exception as e:
            err(f"{lang} AI4Bharat failed: {e}")

# ═══════════════════════════════════════════════════════════════
# PART 3 — SYNTHETIC IMAGES (fonts + Wikipedia text)
# ═══════════════════════════════════════════════════════════════
head("PART 3 — Synthetic Images from Wikipedia Text")
show("Generating synthetic training images from fonts...")
show("(This works for ALL 22 languages, no download needed)")

WIKI_LANGS = {
    "hindi":     "hi",  "bengali":   "bn",  "tamil":     "ta",
    "telugu":    "te",  "kannada":   "kn",  "malayalam": "ml",
    "punjabi":   "pa",  "gujarati":  "gu",  "odia":      "or",
    "marathi":   "mr",  "urdu":      "ur",  "assamese":  "as",
    "nepali":    "ne",  "sanskrit":  "sa",  "maithili":  "mai",
    "dogri":     "dty", "konkani":   "kok", "bodo":      "bpy",
    "sindhi":    "sd",  "kashmiri":  "ks",  "manipuri":  "mni",
    "santali":   "sat",
}

SAMPLE_TEXTS = {
    "hindi":    ["नमस्ते","भारत","हिंदी","दिल्ली","मुंबई","कोलकाता","पानी","खाना","घर","किताब","विद्यालय","परिवार","दोस्त","प्यार","शांति"],
    "bengali":  ["নমস্কার","বাংলাদেশ","কলকাতা","ভালোবাসা","বই","পানি","খাবার","বাড়ি","বন্ধু","পরিবার","শান্তি","আনন্দ","স্কুল","মা","বাবা"],
    "tamil":    ["வணக்கம்","தமிழ்நாடு","சென்னை","அன்பு","நீர்","உணவு","வீடு","நண்பன்","குடும்பம்","அமைதி","மகிழ்ச்சி","பள்ளி","அம்மா","அப்பா","புத்தகம்"],
    "telugu":   ["నమస్కారం","తెలంగాణ","హైదరాబాద్","ప్రేమ","నీరు","ఆహారం","ఇల్లు","స్నేహితుడు","కుటుంబం","శాంతి","ఆనందం","పాఠశాల","అమ్మ","నాన్న","పుస్తకం"],
    "kannada":  ["ನಮಸ್ಕಾರ","ಕರ್ನಾಟಕ","ಬೆಂಗಳೂರು","ಪ್ರೀತಿ","ನೀರು","ಆಹಾರ","ಮನೆ","ಗೆಳೆಯ","ಕುಟುಂಬ","ಶಾಂತಿ","ಸಂತೋಷ","ಶಾಲೆ","ಅಮ್ಮ","ಅಪ್ಪ","ಪುಸ್ತಕ"],
    "malayalam":["നമസ്കാരം","കേരളം","തിരുവനന്തപുരം","സ്നേഹം","വെള്ളം","ഭക്ഷണം","വീട്","സുഹൃത്ത്","കുടുംബം","സമാധാനം","സന്തോഷം","സ്കൂൾ","അമ്മ","അച്ഛൻ","പുസ്തകം"],
    "punjabi":  ["ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ","ਪੰਜਾਬ","ਅੰਮ੍ਰਿਤਸਰ","ਪਿਆਰ","ਪਾਣੀ","ਖਾਣਾ","ਘਰ","ਦੋਸਤ","ਪਰਿਵਾਰ","ਸ਼ਾਂਤੀ","ਖੁਸ਼ੀ","ਸਕੂਲ","ਮਾਂ","ਪਿਤਾ","ਕਿਤਾਬ"],
    "gujarati": ["નમસ્તે","ગુજરાત","અમદાવાદ","પ્રેમ","પાણી","ખોરાક","ઘર","મિત્ર","પરિવાર","શાંતિ","આનંદ","શાળા","માતા","પિતા","પુસ્તક"],
    "odia":     ["ନମସ୍କାର","ଓଡ଼ିଶା","ଭୁବନେଶ୍ୱର","ଭଲ","ଜଳ","ଖାଦ୍ୟ","ଘର","ବନ୍ଧୁ","ପରିବାର","ଶାନ୍ତି","ଆନନ୍ଦ","ବିଦ୍ୟାଳୟ","ମା","ବାପା","ପୁସ୍ତକ"],
    "marathi":  ["नमस्कार","महाराष्ट्र","मुंबई","प्रेम","पाणी","अन्न","घर","मित्र","कुटुंब","शांती","आनंद","शाळा","आई","बाबा","पुस्तक"],
    "urdu":     ["السلام علیکم","پاکستان","کراچی","محبت","پانی","کھانا","گھر","دوست","خاندان","امن","خوشی","اسکول","ماں","باپ","کتاب"],
    "assamese": ["নমস্কাৰ","অসম","গুৱাহাটী","মৰম","পানী","খাদ্য","ঘৰ","বন্ধু","পৰিয়াল","শান্তি","আনন্দ","বিদ্যালয়","মা","দেউতা","কিতাপ"],
    "maithili": ["प्रणाम","मिथिला","दरभंगा","प्रेम","पानि","खाना","घर","दोस्त","परिवार","शांति","आनंद","विद्यालय","माय","बाबूजी","किताब"],
    "sanskrit": ["नमस्ते","भारतवर्षः","संस्कृतम्","प्रेम","जलम्","अन्नम्","गृहम्","मित्रम्","परिवारः","शान्तिः","आनन्दः","विद्यालयः","माता","पिता","पुस्तकम्"],
    "nepali":   ["नमस्ते","नेपाल","काठमाडौं","प्रेम","पानी","खाना","घर","साथी","परिवार","शान्ति","आनन्द","विद्यालय","आमा","बुवा","किताब"],
    "dogri":    ["नमस्ते","जम्मू","डोगरी","प्यार","पानी","खाना","घर","दोस्त","परिवार","शांति","खुशी","स्कूल","माँ","पिताजी","किताब"],
    "konkani":  ["नमस्कार","गोवा","पणजी","मोग","उदक","जेवण","घर","दोस्त","कुटुंब","शांती","आनंद","शाळा","आवय","बापूय","पुस्तक"],
    "bodo":     ["नमस्कार","असम","बड़ो","गोसो","जिउ","बिबार","नखर","बेसेन","सुबुंसोनाय","गोरलैखा","फोरखांथि","बिद्यालय","अमा","अबा","बि"],
    "sindhi":   ["جي آيا","سنڌ","ڪراچي","پيار","پاڻي","کاڌو","گهر","دوست","خاندان","امن","خوشي","اسڪول","ماء","پيء","ڪتاب"],
    "kashmiri": ["السلام علیکم","کشمیر","سرینگر","محبت","پانی","کھانا","گھر","دوست","خاندان","امن","خوشی","اسکول","ماں","باپ","کتاب"],
    "manipuri": ["ꯍꯥꯏꯕꯤꯔꯨ","ꯃꯅꯤꯄꯨꯔ","ꯏꯝꯐꯥꯜ","ꯌꯥꯏꯐꯥꯕꯥ","ꯏꯁꯤꯡ","ꯆꯥꯅꯕꯥ","ꯌꯨꯝ","ꯃꯤꯠꯔꯨ","ꯄꯥꯝꯒꯤ","ꯁꯟꯇꯤ","ꯈꯨꯗꯣꯡꯆꯥꯕꯥ","ꯁ꯭ꯀꯨꯜ","ꯑꯏꯆꯥꯟ","ꯑꯄꯥ","ꯄꯨꯊꯣꯛ"],
    "santali":  ["ᱡᱳᱦᱟᱨ","ᱡᱷᱟᱨᱠᱷᱚᱸᱰ","ᱟᱹᱰᱤ","ᱯᱤᱨᱤᱛ","ᱫᱟᱜ","ᱦᱟᱹᱴᱤᱧ","ᱡᱟᱶᱟ","ᱮᱱᱮᱡ","ᱮᱴᱟᱜ","ᱥᱟᱸᱛᱤ","ᱥᱮᱫ","ᱤᱥᱠᱩᱞ","ᱟᱭᱳ","ᱛᱤᱱᱤᱧ","ᱯᱳᱛᱷᱤ"],
}

def generate_synthetic(lang, texts, count=500):
    """Generate synthetic images from text using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
        import numpy as np

        img_dir = f"{BASE}/{lang}/images"
        records = []

        # Try to find a font — use default if none available
        font_paths = [
            f"fonts/Noto{lang.capitalize()}-Regular.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "C:/Windows/Fonts/Calibri.ttf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, 32)
                    break
                except:
                    pass
        if font is None:
            font = ImageFont.load_default()

        for i in range(count):
            text = random.choice(texts)

            # Image settings
            bg_color = (random.randint(240,255), random.randint(238,255), random.randint(235,255))
            ink_color = (random.randint(0,40), random.randint(0,40), random.randint(0,40))

            # Measure text
            dummy = Image.new("RGB", (1,1))
            draw  = ImageDraw.Draw(dummy)
            try:
                bbox = draw.textbbox((0,0), text, font=font)
                tw, th = bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1
            except:
                tw, th = len(text)*18, 36

            px, py = random.randint(8,20), random.randint(6,14)
            img    = Image.new("RGB", (max(tw+2*px,60), max(th+2*py,40)), color=bg_color)
            draw   = ImageDraw.Draw(img)
            try:
                draw.text((px, py), text, font=font, fill=ink_color)
            except:
                draw.text((px, py), text, fill=ink_color)

            # Small rotation
            angle = random.uniform(-3, 3)
            img   = img.rotate(angle, fillcolor=bg_color, expand=False)

            fname = f"syn_{lang}_{i:05d}.png"
            img.save(os.path.join(img_dir, fname))
            records.append([fname, text])

        return records
    except Exception as e:
        return []

# Generate synthetic for all 22 languages
for lang, texts in SAMPLE_TEXTS.items():
    show(f"Generating synthetic images for {lang}...")
    records = generate_synthetic(lang, texts, count=500)
    if records:
        # Append to existing labels if any
        csv_path = f"{BASE}/{lang}/labels.csv"
        if os.path.exists(csv_path):
            import pandas as pd
            existing = pd.read_csv(csv_path, encoding="utf-8")
            new_df   = pd.DataFrame(records, columns=["filename","text"])
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(csv_path, index=False, encoding="utf-8")
        else:
            save_labels(lang, records)
        ok(f"{lang}: {len(records)} synthetic images created")
    else:
        err(f"{lang}: synthetic generation failed")

# ═══════════════════════════════════════════════════════════════
# PART 4 — SCRAPE WIKIPEDIA TEXT → GENERATE MORE IMAGES
# ═══════════════════════════════════════════════════════════════
head("PART 4 — Wikipedia Text Scraper → More Synthetic Images")
show("Scraping Wikipedia in all 22 languages...")

import urllib.request
import urllib.parse
import json
import random

def scrape_wiki(lang_code, num=50):
    """Get random sentences from Wikipedia in any language."""
    sentences = []
    try:
        api = f"https://{lang_code}.wikipedia.org/w/api.php"
        params = urllib.parse.urlencode({
            "action":"query","list":"random",
            "rnnamespace":0,"rnlimit":20,"format":"json"
        })
        req = urllib.request.Request(
            f"{api}?{params}",
            headers={"User-Agent":"IndianOCR/1.0 (Educational)"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        titles = [p["title"] for p in data["query"]["random"]]

        for title in titles[:5]:
            params2 = urllib.parse.urlencode({
                "action":"query","titles":title,
                "prop":"extracts","explaintext":True,"format":"json"
            })
            req2 = urllib.request.Request(
                f"{api}?{params2}",
                headers={"User-Agent":"IndianOCR/1.0 (Educational)"}
            )
            with urllib.request.urlopen(req2, timeout=8) as r:
                data2 = json.loads(r.read())
            pages = data2["query"]["pages"]
            text  = next(iter(pages.values())).get("extract","")
            for s in text.split("\n"):
                s = s.strip()
                if 4 <= len(s) <= 60:
                    sentences.append(s)
            time.sleep(0.5)
    except Exception as e:
        pass
    return sentences[:num]

for lang, code in WIKI_LANGS.items():
    show(f"Wikipedia scraping: {lang} ({code})...")
    try:
        sentences = scrape_wiki(code, num=30)
        if sentences:
            records = generate_synthetic(lang, sentences, count=200)
            if records:
                csv_path = f"{BASE}/{lang}/labels.csv"
                if os.path.exists(csv_path):
                    import pandas as pd
                    existing = pd.read_csv(csv_path, encoding="utf-8")
                    new_df   = pd.DataFrame(records, columns=["filename","text"])
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined.to_csv(csv_path, index=False, encoding="utf-8")
                else:
                    save_labels(lang, records)
                ok(f"{lang}: +{len(records)} images from Wikipedia")
            else:
                ok(f"{lang}: text scraped but image generation skipped")
        else:
            show(f"{lang}: no text scraped (Wikipedia may be small)")
    except Exception as e:
        err(f"{lang}: {e}")
    time.sleep(1)

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
head("SUMMARY — Dataset Download Complete")
print()
total_images = 0
total_labels = 0
print(f"  {'Language':12s} {'Images':8s} {'Labels':8s}")
print(f"  {'─'*12} {'─'*8} {'─'*8}")
for lang in LANGS:
    imgs = count_images(lang)
    csv_path = f"{BASE}/{lang}/labels.csv"
    lbls = 0
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            lbls = len(pd.read_csv(csv_path))
        except:
            pass
    total_images += imgs
    total_labels += lbls
    status = "✓" if imgs > 0 else "○"
    print(f"  {status} {lang:12s} {imgs:8d} {lbls:8d}")

print()
print(f"  TOTAL:        {total_images:8d} images")
print(f"  TOTAL LABELS: {total_labels:8d} rows")
print()
print("  All data saved in: data/raw/{language}/")
print("  Each folder has:   images/  +  labels.csv")
print()
print("  Next step — preprocess and train:")
print("  python train/trainer.py")
print()
print("="*55)