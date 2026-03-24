from typing import Optional
from pathlib import Path
import time, threading
import asyncio
from collections import deque

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from google.oauth2 import service_account
from googleapiclient.discovery import build
from fastapi.responses import JSONResponse
from datetime import datetime
import os

GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "1n6Q5kY7AG3d1bf4C6RIhh4JNao2JubObIeZAKaOoiJc")
GOOGLE_SHEETS_RANGE = os.getenv("GOOGLE_SHEETS_RANGE", "MMIReports!A:Q")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", r"c:\project\secret\focal-sight-490506-e2-e0cd6f286f88.json")

class MMIReportRequest(BaseModel):
    user_uid: str | None = None
    user_lat: float
    user_lon: float

    mmi_value: int
    mmi_code: str
    feeling_text: str | None = None

    event_time_th: str | None = None
    event_time_utc: str | None = None
    event_lat: float
    event_lon: float
    event_mag: float | None = None
    event_depth_km: float | None = None
    event_changwat: str | None = None

    distance_km: float | None = None
    estimated_pga_percent_g: float | None = None
    source: str | None = "manual_user_report"

from .logic import (
    fetch_latest_event_in_thailand,
    compute_overlay_from_event,
    simulate_event,
    get_soil_info,
    predict_mmi_ai,
)
# ==== ดึงฟังก์ชันจาก logic.py ของโปรเจกคุณ ====
# - fetch_latest_event_in_thailand() : ดึงเมตาเหตุการณ์ล่าสุด
# - compute_overlay_from_event(ev)   : คำนวณ/สร้างผลลัพธ์แผนที่จาก ev
# - simulate_event(lat, lon, depth_km, mag) : จำลองเหตุการณ์


app = FastAPI(title="SHAKEMAP API", version="1.3.0")

from .logic import debug_vs30_paths

@app.get("/api/soil_debug")
def soil_debug():
    return debug_vs30_paths()


# CORS (เปิดกว้างสำหรับ dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ================== In-memory Cache (คำนวณครั้งแรก) ==================
_CACHE_LOCK = threading.Lock()
_CACHE = {
    "data": None,        # JSON ผลลัพธ์เต็ม (รวม data URL/HTML/เมตา)
    "event_key": None,   # คีย์อ้างอิงเหตุการณ์ล่าสุดที่คำนวณแล้ว
    "ts": 0.0,           # เวลาที่คำนวณ (epoch)
}

# ตั้ง TTL ถ้าอยากให้รีเฟรชอัตโนมัติเมื่อพ้นเวลา; None = ไม่หมดอายุเอง
CACHE_TTL_SEC: Optional[int] = None  # เช่น 600 = 10 นาที

def _make_event_key(meta: dict) -> str:
    return f"{meta.get('time_utc') or meta.get('time_th')}|{meta.get('lat')}|{meta.get('lon')}|{meta.get('mag')}|{meta.get('depth_km')}"

def _get_cached_ok() -> bool:
    if _CACHE["data"] is None:
        return False
    if CACHE_TTL_SEC is None:
        return True
    return (time.time() - (_CACHE["ts"] or 0)) < CACHE_TTL_SEC

def _compute_and_store() -> dict:
    """คำนวณผลจากเหตุการณ์ล่าสุด แล้วเก็บลงแคช (ต้องเรียกภายใต้ LOCK)"""
    ev = fetch_latest_event_in_thailand()
    data = compute_overlay_from_event(ev)
    meta = data.get("meta", {})
    _CACHE["data"] = data
    _CACHE["event_key"] = _make_event_key(meta)
    _CACHE["ts"] = time.time()
    return data

def _get_or_compute(force: bool = False) -> dict:
    # 1) บังคับคำนวณใหม่
    if force:
        with _CACHE_LOCK:
            return _compute_and_store()

    # 2) ถ้ายังไม่มีแคช → คำนวณใหม่
    if _CACHE["data"] is None:
        with _CACHE_LOCK:
            if _CACHE["data"] is None:
                return _compute_and_store()
            return _CACHE["data"]

    # 3) มีแคชแล้ว → เช็กว่ามี "เหตุการณ์ใหม่" ไหม (เปรียบเทียบ event_key)
    try:
        ev = fetch_latest_event_in_thailand()  # ดึง meta ล่าสุด (ไม่เรนเดอร์ภาพ)
        if ev:
            # สร้างคีย์จาก meta ล่าสุด
            new_key = _make_event_key({
                "time_utc":  ev.get("time_utc"),
                "time_th":   ev.get("time_th"),
                "lat":       ev.get("lat"),
                "lon":       ev.get("lon"),
                "mag":       ev.get("mag"),
                "depth_km":  ev.get("depth"),
            })
            # ถ้าไม่ใช่เหตุการณ์เดิม → คำนวณใหม่ด้วย ev นี้
            if new_key and new_key != _CACHE["event_key"]:
                with _CACHE_LOCK:
                    data = compute_overlay_from_event(ev)
                    _CACHE["data"] = data
                    _CACHE["event_key"] = new_key
                    _CACHE["ts"] = time.time()
                    return data
    except Exception:
        # ถ้าเช็ก meta ล้มเหลว ให้ใช้ของเดิมไปก่อน
        pass

    # 4) เหตุการณ์เดิม → ใช้แคช
    return _CACHE["data"]



# ================== Queue: จำกัดผู้ใช้พร้อมกัน 10 คน ==================
MAX_ACTIVE = 10           # จำกัดผู้ใช้งานพร้อมกัน
HEARTBEAT_TIMEOUT = 45.0     # วินาที: ไม่ส่ง heartbeat เกินนี้ = หลุด
PROMOTE_BATCH = 5            # โปรโมตทีละกี่คนจากคิว (กันกรณีพุ่งพร้อมกัน)

_q_lock = asyncio.Lock()
_active = {}         # key=(client_id, tab_id) -> last_seen_ts
_queue = deque()     # item=(client_id, tab_id, enq_ts)

def _now() -> float:
    return time.time()

def _queue_position(client_id: str, tab_id: str):
    pos = 1
    for (c, t, _) in _queue:
        if c == client_id and t == tab_id:
            return pos
        pos += 1
    return None

async def _maintain_and_promote():
    """ลบ active ที่หมดอายุ และโปรโมตจากคิวตามช่องว่าง"""
    now = _now()
    # ตัด active หมดอายุ
    expired = [k for k, ts in list(_active.items()) if (now - ts) > HEARTBEAT_TIMEOUT]
    for k in expired:
        _active.pop(k, None)

    # โปรโมตจากคิว
    slots = max(0, MAX_ACTIVE - len(_active))
    moved = 0
    while slots > 0 and _queue and moved < PROMOTE_BATCH:
        c_id, t_id, _ = _queue[0]
        key = (c_id, t_id)
        if key in _active:
            _queue.popleft()
            continue
        _active[key] = _now()
        _queue.popleft()
        slots -= 1
        moved += 1

def append_mmi_report_to_google_sheet(payload: MMIReportRequest):
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        raise RuntimeError("Missing GOOGLE_SHEETS_SPREADSHEET_ID")
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_FILE")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=scopes,
    )

    service = build("sheets", "v4", credentials=creds)

    values = [[
        datetime.now().isoformat(),
        payload.user_uid,
        payload.user_lat,
        payload.user_lon,
        payload.mmi_value,
        payload.mmi_code,
        payload.feeling_text,
        payload.event_time_th,
        payload.event_time_utc,
        payload.event_lat,
        payload.event_lon,
        payload.event_mag,
        payload.event_depth_km,
        payload.event_changwat,
        payload.distance_km,
        payload.estimated_pga_percent_g,
        payload.source,
    ]]

    body = {"values": values}

    service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEETS_SPREADSHEET_ID,
        range=GOOGLE_SHEETS_RANGE,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

# ================== Routes ==================
@app.get("/")
def index():
    # เสิร์ฟหน้าเว็บ
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/api/report_mmi")
def api_report_mmi(payload: MMIReportRequest):
    try:
        append_mmi_report_to_google_sheet(payload)
        return {"ok": True}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "detail": str(e)}
        )

# --------- Queue APIs ----------
@app.post("/api/queue/enter")
async def queue_enter(req: Request):
    body = await req.json()
    client_id = body.get("client_id")
    tab_id    = body.get("tab_id")
    if not client_id or not tab_id:
        return JSONResponse({"error": "missing client_id/tab_id"}, status_code=400)

    async with _q_lock:
        await _maintain_and_promote()
        key = (client_id, tab_id)

        # อยู่ active แล้ว
        if key in _active:
            _active[key] = _now()
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}

        # ยังมีช่องว่าง
        if len(_active) < MAX_ACTIVE:
            _active[key] = _now()
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}

        # เต็ม -> เข้าคิวถ้ายังไม่อยู่
        if _queue_position(client_id, tab_id) is None:
            _queue.append((client_id, tab_id, _now()))
        pos = _queue_position(client_id, tab_id)
        return {"state": "queued", "position": pos, "active": len(_active), "limit": MAX_ACTIVE}

@app.get("/api/queue/status")
async def queue_status(client_id: str, tab_id: str):
    async with _q_lock:
        await _maintain_and_promote()
        key = (client_id, tab_id)
        if key in _active:
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}
        pos = _queue_position(client_id, tab_id)
        if pos is not None:
            return {"state": "queued", "position": pos, "active": len(_active), "limit": MAX_ACTIVE}
        return {"state": "none", "active": len(_active), "limit": MAX_ACTIVE}

@app.post("/api/queue/heartbeat")
async def queue_heartbeat(req: Request):
    body = await req.json()
    client_id = body.get("client_id")
    tab_id    = body.get("tab_id")
    if not client_id or not tab_id:
        return JSONResponse({"error": "missing client_id/tab_id"}, status_code=400)

    async with _q_lock:
        await _maintain_and_promote()
        key = (client_id, tab_id)
        if key in _active:
            _active[key] = _now()
            return {"state": "active", "active": len(_active), "limit": MAX_ACTIVE}
        pos = _queue_position(client_id, tab_id)
        if pos is not None:
            return {"state": "queued", "position": pos, "active": len(_active), "limit": MAX_ACTIVE}
        return {"state": "none", "active": len(_active), "limit": MAX_ACTIVE}

@app.post("/api/queue/leave")
async def queue_leave(req: Request):
    body = await req.json()
    client_id = body.get("client_id")
    tab_id    = body.get("tab_id")
    if not client_id or not tab_id:
        return JSONResponse({"error": "missing client_id/tab_id"}, status_code=400)

    async with _q_lock:
        key = (client_id, tab_id)
        _active.pop(key, None)
        # ลบจากคิวหากมี
        for i, (c, t, ts) in enumerate(list(_queue)):
            if c == client_id and t == tab_id:
                try:
                    _queue.remove((c, t, ts))
                except Exception:
                    pass
                break
        await _maintain_and_promote()
        return {"ok": True, "active": len(_active), "limit": MAX_ACTIVE}


# --------- Data APIs (เดิม) ----------
# GET สำหรับเปิดในเบราว์เซอร์/เทส
@app.get("/api/run")
def api_run_get():
    return JSONResponse({
        "ok": True,
        "mode": "simulate_only",
        "message": "ใช้ POST /api/run หรือ POST /api/simulate พร้อม lat/lon/depth/mag"
    })

@app.get("/api/run")
def api_run_get():
    return JSONResponse({
        "ok": True,
        "mode": "simulate_only",
        "message": "ใช้ POST /api/run พร้อม lat/lon/depth/mag"
    })

@app.post("/api/run")
def api_run(body: dict = Body(...)):
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
        depth = float(body["depth"])
        mag = float(body["mag"])

        data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/simulate")
def api_simulate(body: dict = Body(...)):
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
        depth = float(body["depth"])
        mag = float(body["mag"])

        data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/simulate")
def api_simulate(body: dict = Body(...)):
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
        depth = float(body["depth"])
        mag = float(body["mag"])

        data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# รีเฟรชเหตุการณ์ล่าสุดแบบบังคับ (สำหรับแอดมิน/DevTools)
@app.post("/api/refresh")
def api_refresh():
    try:
        data = _get_or_compute(force=True)
        return JSONResponse({"ok": True, "meta": data.get("meta", {}), "event_key": _CACHE["event_key"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ดูสถานะแคช (debug)
@app.get("/api/cache_state")
def api_cache_state():
    return {
        "has_cache": _CACHE["data"] is not None,
        "event_key": _CACHE["event_key"],
        "ts": _CACHE["ts"],
        "ttl_sec": CACHE_TTL_SEC,
    }



# ชั้นดิน/ประเภทชั้นดิน (Site Class จาก Vs30)
@app.get("/api/soil")
def api_soil(lat: float, lon: float):
    try:
        return JSONResponse(get_soil_info(lat=lat, lon=lon))
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e), "lat": lat, "lon": lon}, status_code=500)

@app.post("/api/simulate")
def api_simulate(body: dict = Body(...)):
    try:
        lat   = float(body["lat"])
        lon   = float(body["lon"])
        depth = float(body["depth"])
        mag   = float(body["mag"])
        data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
from pydantic import BaseModel

class MMIPredictRequest(BaseModel):
    dist: float
    pga: float
    mag: float
    lat: float | None = None
    lon: float | None = None


@app.post("/api/predict_mmi")
def api_predict_mmi(payload: MMIPredictRequest):
    try:
        result = predict_mmi_ai(
        dist=payload.dist,
        pga=payload.pga / 100.0,   # payload มาจากเว็บเป็น %g
        mag=payload.mag,
        lat=payload.lat,
        lon=payload.lon,
    )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    