"""api/server.py — FastAPI REST API.  Run: uvicorn api.server:app --port 8000"""

import io, os, sys, uuid, zipfile
from typing import Optional
import cv2, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

import importlib, importlib.util
_SRC  = os.path.join(_ROOT, "src")
def _load(alias, fname):
    spec=importlib.util.spec_from_file_location(alias,os.path.join(_SRC,fname))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

_lang_mod  = _load("_srv_lang", "languages.py")
_eng_mod   = _load("_srv_eng",  "ocr_engine.py")
LANGUAGES  = _lang_mod.LANGUAGES
ALL_LANG_NAMES = _lang_mod.ALL_LANG_NAMES
OCREngine  = _eng_mod.OCREngine

app = FastAPI(title="Indian OCR API", version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

_engines={}; _jobs={}

def _eng(lang,backend):
    k=f"{lang}:{backend}"
    if k not in _engines: _engines[k]=OCREngine(language=lang,backend=backend)
    return _engines[k]

class Resp(BaseModel):
    text:str; confidence:float; language:str; backend:str
    time_ms:float; char_count:int; word_count:int
    status:str="success"; error:Optional[str]=None

@app.get("/")
def root(): return {"status":"running","languages":len(ALL_LANG_NAMES)}

@app.get("/health")
def health(): return {"status":"healthy"}

@app.get("/languages")
def langs(): return sorted([{"name":n,"native":c["native"],"script":c["script"],
    "direction":c["direction"],"easyocr":c["easyocr"] is not None,
    "tesseract":c["tesseract"] is not None}
    for n,c in LANGUAGES.items()],key=lambda x:x["name"])

@app.post("/ocr",response_model=Resp)
async def ocr_single(file:UploadFile=File(...),
                     language:str=Form(default="hindi"),
                     backend:str=Form(default="easyocr")):
    if language not in LANGUAGES: raise HTTPException(400,f"Unknown language: {language}")
    if not file.content_type.startswith("image/"): raise HTTPException(400,"Need image file")
    contents=await file.read()
    tmp=f"/tmp/ocr_{uuid.uuid4().hex[:8]}.png"
    try:
        img=cv2.imdecode(np.frombuffer(contents,np.uint8),cv2.IMREAD_COLOR)
        if img is None: raise HTTPException(400,"Cannot decode image")
        cv2.imwrite(tmp,img)
        r=_eng(language,backend).predict(tmp)
        return Resp(text=r.text,confidence=r.confidence,language=r.language,
                    backend=r.backend,time_ms=r.time_ms,
                    char_count=len(r.text.replace(" ","")),
                    word_count=len(r.text.split()),error=r.error)
    finally:
        if os.path.exists(tmp): os.remove(tmp)

@app.post("/ocr/batch")
async def ocr_batch(bg:BackgroundTasks,file:UploadFile=File(...),
                    language:str=Form(default="hindi"),backend:str=Form(default="easyocr")):
    if not file.filename.endswith(".zip"): raise HTTPException(400,"Upload .zip file")
    jid=uuid.uuid4().hex[:12]; content=await file.read()
    _jobs[jid]={"status":"queued","total":0,"done":0,"results":[],"error":None}
    bg.add_task(_run_batch,jid,content,language,backend)
    return {"job_id":jid,"poll_url":f"/ocr/status/{jid}"}

@app.get("/ocr/status/{jid}")
def status(jid:str):
    if jid not in _jobs: raise HTTPException(404,f"Job {jid} not found")
    return _jobs[jid]

def _run_batch(jid,zb,lang,backend):
    import tempfile,shutil; job=_jobs[jid]
    try:
        d=tempfile.mkdtemp()
        with zipfile.ZipFile(io.BytesIO(zb)) as z: z.extractall(d)
        imgs=[os.path.join(r,f) for r,_,fs in os.walk(d)
              for f in fs if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))]
        job.update({"status":"running","total":len(imgs)})
        e=_eng(lang,backend)
        for p in sorted(imgs):
            r=e.predict(p)
            job["results"].append({"filename":os.path.basename(p),"text":r.text,
                "confidence":r.confidence,"time_ms":r.time_ms,"error":r.error})
            job["done"]=len(job["results"])
        job["status"]="done"; shutil.rmtree(d,ignore_errors=True)
    except Exception as ex:
        job.update({"status":"error","error":str(ex)})