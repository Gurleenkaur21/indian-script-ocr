"""setup.py — Run this first to check everything is installed.
Usage: python setup.py
"""
import os, sys

def check():
    print("\n  Indian OCR — Setup Check\n" + "="*40)

    # Create folders
    folders=[
        "data/raw","data/vocab",
        "data/processed/train/images",
        "data/processed/val/images",
        "data/processed/test/images",
        "models/checkpoints",
        "results/plots","results/predictions","fonts",
    ]
    for f in folders: os.makedirs(f,exist_ok=True)
    print("✓  Project folders created\n")

    # Check libraries
    libs=[
        ("cv2",          "opencv-python"),
        ("numpy",        "numpy"),
        ("PIL",          "Pillow"),
        ("pandas",       "pandas"),
        ("sklearn",      "scikit-learn"),
        ("torch",        "torch"),
        ("easyocr",      "easyocr"),
        ("pytesseract",  "pytesseract"),
        ("editdistance", "editdistance"),
        ("fastapi",      "fastapi"),
        ("uvicorn",      "uvicorn[standard]"),
        ("albumentations","albumentations"),
    ]
    missing=[]
    for lib,pkg in libs:
        try:
            __import__(lib)
            print(f"  ✓  {pkg}")
        except ImportError:
            print(f"  ✗  {pkg}  ← run: pip install {pkg}")
            missing.append(pkg)

    import torch
    print(f"\n  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  GPU     : {torch.cuda.is_available()}")

    # Check Tesseract
    import subprocess
    try:
        r=subprocess.run(["tesseract","--version"],capture_output=True,text=True)
        print(f"  Tesseract: {r.stdout.splitlines()[0]}")
    except FileNotFoundError:
        print("  Tesseract: NOT FOUND — install from github.com/UB-Mannheim/tesseract/wiki")

    # Check src files
    src_files=["languages.py","preprocess.py","vocabulary.py",
               "model.py","dataset.py","ocr_engine.py"]
    print("\n  src/ files:")
    for f in src_files:
        p=os.path.join("src",f)
        ok="✓" if os.path.exists(p) else "✗  MISSING"
        print(f"    {ok}  {p}")

    print()
    if missing:
        print(f"Run these commands to fix:\n")
        for p in missing: print(f"  pip install {p}")
    else:
        print("All libraries installed!")
        print("\nQuick start:")
        print("  python src/ocr_engine.py image.jpg --language hindi")
        print("  python cli/ocr.py image.jpg --language tamil")
        print("  uvicorn api.server:app --reload --port 8000")
    print()

if __name__=="__main__": check()
