#!/usr/bin/env python3
"""cli/ocr.py — Command line OCR tool.
Usage:
  python cli/ocr.py image.jpg
  python cli/ocr.py image.jpg --language tamil --backend tesseract
  python cli/ocr.py folder/ --batch --output results.csv
  python cli/ocr.py --list-languages
"""

import argparse, os, sys, importlib, importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")

def _load(alias, fname):
    spec=importlib.util.spec_from_file_location(alias,os.path.join(_SRC,fname))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

_lang_mod = _load("_cli_lang","languages.py")
_eng_mod  = _load("_cli_eng", "ocr_engine.py")
ALL_LANG_NAMES = _lang_mod.ALL_LANG_NAMES
LANGUAGES      = _lang_mod.LANGUAGES
OCREngine      = _eng_mod.OCREngine

G="\033[92m"; Y="\033[93m"; R="\033[91m"; B="\033[94m"; C="\033[96m"; X="\033[0m"; D="\033[1m"

def show(r, path=""):
    if r.error: print(f"{R}Error: {r.error}{X}"); return
    cc = G if r.confidence>=70 else Y if r.confidence>=40 else R
    print(f"\n{D}File      :{X} {os.path.basename(path)}")
    print(f"{D}Language  :{X} {r.language}   {D}Backend:{X} {r.backend}")
    print(f"{D}Text      :{X} {C}{r.text}{X}")
    print(f"{D}Confidence:{X} {cc}{r.confidence:.1f}%{X}   {D}Time:{X} {r.time_ms:.0f}ms")

def do_list():
    print(f"\n{D}{'Language':14s} {'Native':16s} {'Script':15s} EasyOCR  Tesseract{X}")
    print("─"*65)
    for n,c in sorted(LANGUAGES.items()):
        e=f"{G}Yes{X}" if c["easyocr"] else f"{R}No {X}"
        t=f"{G}Yes{X}" if c["tesseract"] else f"{R}No {X}"
        print(f"  {n:12s} {c['native']:16s} {c['script']:15s}  {e}      {t}")
    print()

def main():
    ap=argparse.ArgumentParser(description="Indian OCR Command Line Tool")
    ap.add_argument("image",nargs="?",default=None)
    ap.add_argument("--language","-l",default="hindi",choices=ALL_LANG_NAMES,metavar="LANG")
    ap.add_argument("--backend", "-b",default="easyocr",choices=["easyocr","tesseract","custom"])
    ap.add_argument("--batch",   action="store_true")
    ap.add_argument("--output",  "-o",default=None)
    ap.add_argument("--list-languages",action="store_true")
    ap.add_argument("--model",default="models/checkpoints/best_model.pth")
    ap.add_argument("--vocab",default="data/vocab/vocab.json")
    ap.add_argument("--gpu",  action="store_true")
    args=ap.parse_args()

    print(f"\n{B}{D}Indian OCR — 22 Languages{X}\n")

    if args.list_languages: do_list(); return
    if not args.image: ap.print_help(); return

    eng=OCREngine(args.language,args.backend,args.model,args.vocab,args.gpu)

    if args.batch:
        results=eng.predict_folder(args.image,save_csv=args.output or "results/batch.csv")
        ok=sum(1 for r in results if not r.error)
        avg=sum(r.confidence for r in results if not r.error)/max(ok,1)
        print(f"\n{G}Done: {ok}/{len(results)} | avg confidence {avg:.1f}%{X}")
    else:
        r=eng.predict(args.image); show(r,args.image)
        if args.output:
            with open(args.output,"w",encoding="utf-8") as f: f.write(r.text)
            print(f"\n{G}Saved → {args.output}{X}")

if __name__=="__main__": main()
