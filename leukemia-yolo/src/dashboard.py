"""Local research dashboard. Launch with one Uvicorn worker."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .config import load_config
from .integrated_pipeline import SmearPipeline

app = FastAPI(title="Leukemia smear research dashboard")
_lock = threading.Lock()
_pipeline = None
_root = None


def setup():
    global _pipeline, _root
    if _pipeline is None:
        cfg = load_config()
        _root = cfg["output_path"] / "inference"
        _root.mkdir(parents=True, exist_ok=True)
        completed = sorted((cfg["output_path"] / "classifier/full_finetune").glob("full_*/completed.json"),
                           key=lambda path: path.stat().st_mtime_ns)
        classifier_checkpoint = (completed[-1].parent / "best_model.pth" if completed
                                 else cfg["output_path"] / "classifier/runs/efficientnet_b3_20261013T110511Z_67988092/best_model.pth")
        _pipeline = SmearPipeline(
            cfg["output_path"] / "checkpoints/main_20261013T090007Z_a25a7934/weights/best.pt",
            classifier_checkpoint,
            device=cfg["device"], confidence=cfg["confidence"], iou=cfg["iou"], gradcam=True)


def artifact(run_id: str, name: str) -> Path:
    setup()
    if "/" in run_id or "\\" in run_id or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "Invalid artifact path")
    path = (_root / run_id / name).resolve()
    if path.parent != (_root / run_id).resolve() or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return path


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html><html><head><meta charset='utf-8'><title>Smear analysis</title>
<style>body{font:16px system-ui;max-width:1050px;margin:2em auto;padding:0 1em;background:#f6f8fb;color:#17202a}button{padding:.6em 1em}img{max-width:100%;max-height:460px}#cells{display:flex;gap:1em;flex-wrap:wrap}.cell{background:white;padding:1em;border:1px solid #ccd;border-radius:8px;width:180px;cursor:pointer}.cell img{height:120px}.panel{background:white;padding:1em;margin:1em 0;border-radius:8px}</style></head>
<body><h1>Blood-smear research analysis</h1><p>For research and assistive screening only; not a standalone medical diagnosis.</p>
<form id='form'><input name='file' type='file' accept='image/png,image/jpeg,image/bmp' required><button>Analyze</button></form>
<p id='status'></p><div class='panel'><img id='preview'></div><div id='result'></div>
<script>
const form=document.querySelector('#form'),status=document.querySelector('#status'),preview=document.querySelector('#preview'),result=document.querySelector('#result');
form.file.onchange=()=>{if(form.file.files[0])preview.src=URL.createObjectURL(form.file.files[0]);};
form.onsubmit=async(e)=>{e.preventDefault();status.textContent='Analyzing image…';result.textContent='';
try{const r=await fetch('/analyze',{method:'POST',body:new FormData(form)});const data=await r.json();if(!r.ok)throw Error(data.detail||r.statusText);
status.textContent='Complete';const a=(name)=>'/artifact/'+encodeURIComponent(data.run_id)+'/'+encodeURIComponent(name);
const s=data.report;result.innerHTML='<div class="panel"><h2>Counts</h2><p>Detected WBCs: '+s.total_detected_wbc+' | Classified: '+s.total_classified_wbc+' | Normal: '+s.predicted_normal+' | Blast: '+s.predicted_blast+' | Unclassified: '+s.unclassified+'</p><p>Estimated blast fraction among accepted classified WBCs: '+(s.estimated_blast_fraction_percent===null?'Unavailable':s.estimated_blast_fraction_percent.toFixed(1)+'%')+'</p><img src="'+a('annotated.png')+'"><p><a href="'+a('report.json')+'" download>JSON report</a> · <a href="'+a('cells.csv')+'" download>CSV cells</a></p><p>'+s.warnings.join(' ')+'</p></div><h2>Detected cells</h2><div id="cells"></div><div class="panel" id="selected">Select a cell to inspect its crop and visual attribution.</div>';
const cells=document.querySelector('#cells');for(const [i,c] of s.cells.entries()){const item=document.createElement('div');item.className='cell';item.innerHTML='<img src="'+a(c.crop_filename)+'"><p>Cell '+(i+1)+': '+c.label+'</p>';item.onclick=()=>{document.querySelector('#selected').innerHTML='<h3>Cell '+(i+1)+'</h3><p>Blast model score: '+c.blast_score.toFixed(3)+' (uncalibrated)</p><img src="'+a(c.crop_filename)+'">'+(c.gradcam_filename?'<img src="'+a(c.gradcam_filename)+'">':'<p>No Grad-CAM for an unclassified cell.</p>');};cells.append(item);}
}catch(error){status.textContent='Analysis failed: '+error.message;}};
</script></body></html>"""


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/bmp"):
        raise HTTPException(415, "Upload PNG, JPEG or BMP")
    data = file.file.read(15 * 1024 * 1024 + 1)
    if not _lock.acquire(blocking=False):
        raise HTTPException(503, "An analysis is already running")
    try:
        setup()
        run, report = _pipeline.analyze(data, _root)
        return {"run_id": run.name, "report": report}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        _lock.release()


@app.get("/artifact/{run_id}/{name}")
def get_artifact(run_id: str, name: str):
    return FileResponse(artifact(run_id, name))


@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    setup()
    if "/" in run_id or "\\" in run_id or not run_id.startswith("smear_"):
        raise HTTPException(400, "Invalid run ID")
    folder = (_root / run_id).resolve()
    if folder.parent != _root.resolve() or not folder.is_dir():
        raise HTTPException(404, "Run not found")
    for path in folder.iterdir():
        if path.is_file():
            path.unlink()
    folder.rmdir()
    return {"deleted": run_id}
