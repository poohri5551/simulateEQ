import io
import re
import math
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from scipy.interpolate import griddata   # ใช้เท่านี้พอ
from .features import make_features
import os
import ctypes
from ctypes import wintypes

from pathlib import Path

# --- AI MMI models (regional joblib) ---
MMI_MODEL = None
MMI_MODELS = {}

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODEL_DIR / "mmi_model.joblib"      # north
_MODEL_WEST_PATH = _MODEL_DIR / "mmi_west.joblib"
_MODEL_SOUTH_PATH = _MODEL_DIR / "mmi_south.joblib"


def _ensure_model_shims():
    import sys
    try:
        from . import features as _app_features
        sys.modules.setdefault("features", _app_features)
    except Exception:
        pass

    try:
        import __main__
        if not hasattr(__main__, "make_features"):
            from .features import make_features as _mf
            setattr(__main__, "make_features", _mf)
    except Exception:
        pass


def _load_one_model(path: Path, label: str):
    try:
        import joblib
        if path.exists():
            m = joblib.load(path)
            print(f"[MMI_MODEL] loaded {label}: {path}")
            return m
        print(f"[MMI_MODEL] not found {label}: {path}")
        return None
    except Exception as e:
        print(f"[MMI_MODEL] load failed {label}: {repr(e)}")
        return None


try:
    _ensure_model_shims()

    north_model = _load_one_model(_MODEL_PATH, "north/default")
    west_model  = _load_one_model(_MODEL_WEST_PATH, "west")
    south_model = _load_one_model(_MODEL_SOUTH_PATH, "south")

    MMI_MODELS = {
        "north": north_model,
        "west": west_model,
        "south": south_model,
    }

    # fallback หลัก
    MMI_MODEL = north_model or west_model or south_model

except Exception as e:
    print("[MMI_MODEL] init failed:", repr(e))
    MMI_MODEL = None
    MMI_MODELS = {}

def _thai_region_from_epicenter(lat: float, lon: float) -> str:
    """
    แบ่งโซนแบบง่ายด้วย bounding box สำหรับประเทศไทย
    ใช้ lat/lon ของจุดศูนย์กลางเหตุการณ์
    """

    lat = float(lat)
    lon = float(lon)

    # ภาคใต้: คาบสมุทรไทย
    if lat < 11.0:
        return "south"

    # ภาคตะวันตก: กาญจนบุรี/ตาก/ราชบุรี/เพชรบุรีฝั่งตะวันตกประมาณนี้
    # ใช้กรอบกว้างก่อนเพื่อให้ route โมเดลได้
    if 11.0 <= lat <= 19.5 and 98.0 <= lon <= 99.8:
        return "west"

    # ที่เหลือโยนเข้า north/default
    return "north"


def _get_mmi_model_for_region(lat: float | None, lon: float | None):
    """
    คืน (model, region_name)
    fallback ไป MMI_MODEL ถ้ายังหาโมเดลเฉพาะภาคไม่ได้
    """
    if lat is None or lon is None:
        return MMI_MODEL, "default"

    region = _thai_region_from_epicenter(lat, lon)
    model = MMI_MODELS.get(region)

    if model is None:
        model = MMI_MODEL

    return model, region


def _win_short_path(path: str) -> str:
    """แปลง path เป็น 8.3 short path บน Windows เพื่อให้ lib C (netCDF4/HDF5) เปิดได้ง่ายขึ้น"""
    if os.name != "nt":
        return path
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        GetShortPathNameW.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(4096)
        r = GetShortPathNameW(path, buf, len(buf))
        return buf.value if r else path
    except Exception:
        return path

# ====== Interpolation switches ======
USE_INTERPOLATION  = True          # บังคับใช้ interpolation
TARGET_SPACING_DEG = 0.025         # ความละเอียดกริด “ปลายทาง” หลัง interpolate
SAMPLE_STEP        = 2             # เลือกจุดตัวอย่างห่าง ๆ
SOFTMASK_SIGMA_KM  = 50.0          # sigma ของ Gaussian soft mask
TMD_URL = "https://earthquake.tmd.go.th/inside.html"

# ====== HTTP headers (รวมที่เดียว) ======
_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (QuakePGA/1.0; +local)",
    "Accept-Language": "th,th-TH;q=0.9,en;q=0.8",
}

# ====== Custom PGA color bins (discrete) ======
# เปิด/ปิดโหมดกำหนดช่วงสีเอง
USE_CUSTOM_PGA_COLORS = True

# ขอบเขตช่วง PGA (หน่วย %g) มี N+1 ค่า สำหรับสี N ช่วง (ตัวสุดท้ายใช้ inf ได้)
PGA_BOUNDS = [0.0, 0.05, 0.3, 2.8, 6.2, 12, 22, 40, 75, 139, float("inf")]

# สีของแต่ละช่วง (HEX) ยาวเท่ากับจำนวนช่วง = len(PGA_BOUNDS) - 1
# โทนอ่อนเย็น → เขียว/ฟ้า เมื่อ PGA ต่ำ และร้อนขึ้นเมื่อค่าสูง
PGA_COLORS = [
    "#eaf2ff",  # <0.05
    "#cfe3ff",  # 0.05–0.3
    "#a6d8ff",  # 0.3–2.8
    "#b7f0b7",  # 2.8–6.2
    "#e9f57f",  # 6.2–12
    "#ffe066",  # 12–22
    "#ff9b40",  # 22–40
    "#ff4d3a",  # 40–75
    "#d8222a",  # 75–139
    "#990000",  # ≥139
]


# -------------------- Utilities --------------------
def _clean_num(s: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", s))

def _parse_latlon(lat_s: str, lon_s: str):
    lat = _clean_num(lat_s)
    lon = _clean_num(lon_s)
    s_lat = lat_s.strip().upper()
    s_lon = lon_s.strip().upper()
    if s_lat.endswith(("S", "ใต้")):
        lat = -lat
    if s_lon.endswith(("W", "ตะวันตก")):
        lon = -lon
    return lat, lon

def _looks_region_th(s: str) -> bool:
    return any(k in s for k in ("ประเทศไทย", "จ.", "อ.", "ต.", "ตำบล", "อำเภอ", "จังหวัด"))

def _parse_datetime_th_block(block_before_mag: list[str]) -> tuple[str, str]:
    re_dt = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
    time_th, time_utc = "", ""
    for s in block_before_mag[-6:][::-1]:
        if re_dt.search(s):
            if "UTC" in s.upper():
                m = re_dt.search(s)
                if m:
                    time_utc = m.group(0)
            else:
                m = re_dt.search(s)
                if m and not time_th:
                    time_th = m.group(0)
        if time_th and time_utc:
            break
    return time_th, time_utc

def reverse_geocode_th(lat: float, lon: float, timeout=15):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": str(lat),
        "lon": str(lon),
        "accept-language": "th",
        "zoom": 12,
        "addressdetails": 1,
    }
    headers = {"User-Agent": _HTTP_HEADERS["User-Agent"]}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        addr = js.get("address", {}) or {}
    except Exception:
        addr = {}

    tambon = (
        addr.get("subdistrict") or
        addr.get("town") or
        addr.get("village") or
        addr.get("suburb") or
        ""
    )
    amphoe = (addr.get("district") or addr.get("county") or "")
    changwat = (addr.get("province") or addr.get("state") or "")
    return {"tambon": tambon, "amphoe": amphoe, "changwat": changwat}

def _parse_tambon_from_text(s: str):
    tambon = ""
    m_a = re.search(r"(อ\.|อำเภอ)\s*([ก-๛A-Za-z\.\-\s]+)", s)
    m_c = re.search(r"(จ\.|จังหวัด)\s*([ก-๛A-Za-z\.\-\s]+)", s)
    amphoe = (m_a.group(2).strip() if m_a else "")
    changwat = (m_c.group(2).strip() if m_c else "")
    m_t = re.search(r"(ต\.|ตำบล)\s*([ก-๛A-Za-z\.\-\s]+)", s)
    tambon = (m_t.group(2).strip() if m_t else "")
    return tambon, amphoe, changwat

def fetch_latest_event_in_thailand():
    html = requests.get(TMD_URL, timeout=20, headers=_HTTP_HEADERS).text
    soup = BeautifulSoup(html, "html.parser")
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]

    events = []
    i = 0
    re_mag = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
    re_deg = re.compile(r"^[\-]?[0-9]+(?:\.[0-9]+)?\s*°\s*[NSEW]?$", re.IGNORECASE)
    re_num = re.compile(r"^[\-]?[0-9]+(?:\.[0-9]+)?$")

    while i < len(lines) - 7:
        s_mag = lines[i]
        if re_mag.fullmatch(s_mag):
            lat_s = lines[i+1] if i+1 < len(lines) else ""
            lon_s = lines[i+2] if i+2 < len(lines) else ""
            dep_s = lines[i+3] if i+3 < len(lines) else ""
            cand4 = lines[i+4] if i+4 < len(lines) else ""
            cand5 = lines[i+5] if i+5 < len(lines) else ""
            region_s = cand5 if re_num.fullmatch(cand4 or "") else cand4
            pre_block = lines[max(0, i-6):i]
            time_th, time_utc = _parse_datetime_th_block(pre_block)
            dep_num_s = re.sub(r"[^0-9.\-]", "", dep_s) or "0"
            if re_deg.fullmatch(lat_s) and re_deg.fullmatch(lon_s) and re_num.fullmatch(dep_num_s):
                try:
                    mag = float(s_mag)
                    lat, lon = _parse_latlon(lat_s, lon_s)
                    depth = float(dep_num_s)
                    events.append(dict(
                        mag=mag, lat=lat, lon=lon, depth=depth, region=region_s,
                        time_th=time_th, time_utc=time_utc
                    ))
                    i += 6
                    continue
                except Exception:
                    pass
        i += 1

    if not events:
        return None
    for ev in events:
        if _looks_region_th(ev["region"]):
            return ev
    return events[0]

def _fmt_th_datetime(s: str) -> str:
    try:
        dt = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return s

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

def _fmt_num(x, digits=3):
    s = f"{float(x):.{digits}f}"
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"



# ================= Soil (Vs30 / NEHRP Site Class) =================
# แนวคิด: ถ้ามีไฟล์ Vs30 (GeoTIFF/GRD) ใน static/data จะ sample เพื่อแปลงเป็น "Site Class"
# - GeoTIFF แนะนำ: vs30_thailand.tif หรือ vs30_global.tif
# - GMT/NetCDF GRD: vs30_global.grd

_VS30_CACHE = {"kind": None, "ds": None, "band": None, "path": None, "last_error": None}


def _site_class_from_vs30(vs30_ms: float) -> str:
    """NEHRP Site Class (A–E) จาก Vs30 (m/s)."""
    if vs30_ms is None or not np.isfinite(vs30_ms) or vs30_ms <= 0:
        return ""
    if vs30_ms > 1500:
        return "A"
    if vs30_ms >= 760:
        return "B"
    if vs30_ms >= 360:
        return "C"
    if vs30_ms >= 180:
        return "D"
    return "E"

def _site_class_desc_th(sc: str) -> str:
    return {
        "A": "หินแข็งมาก",
        "B": "หินแข็ง",
        "C": "ดินแน่น/หินผุ",
        "D": "ดินอ่อน–ปานกลาง",
        "E": "ดินอ่อนมาก",
    }.get(sc, "")

def _load_vs30_dataset():
    """โหลด Vs30 dataset แบบ lazy (ถ้ามี). คืน True/False"""
    if _VS30_CACHE.get("ds") is not None:
        return True

    data_dir = Path(__file__).resolve().parent / "static" / "data"
    tif_candidates = ["vs30_thailand.tif", "vs30_global.tif", "global_vs30.tif", "vs30.tif"]
    grd_candidates = ["vs30_global.grd", "global_vs30.grd", "vs30.grd"]

    # 1) GeoTIFF (ใช้ rasterio)
    for name in tif_candidates:
        p = data_dir / name
        if p.exists():
            try:
                import rasterio  # type: ignore
                ds = rasterio.open(p)
                _VS30_CACHE.update({"kind": "tif", "ds": ds, "band": 1, "path": str(p), "last_error": None})
                return True
            except Exception as e:
                _VS30_CACHE["last_error"] = f"read_error:{p.name}:{e}"
                # ลองไฟล์ถัดไป
                continue

        # 2) GRD/NetCDF (ใช้ xarray)
    for name in grd_candidates:
        p = data_dir / name
        if p.exists():
            try:
                import xarray as xr  # type: ignore

                errors = {}

                # 1) บังคับลอง netcdf4 ก่อน (ไฟล์คุณเป็น NetCDF4)
                path_str = str(p)
                try:
                    ds = xr.open_dataset(path_str, engine="netcdf4")
                except Exception as e:
                    # ถ้าเจอ Errno 2 ให้ลอง short path (ช่วยเรื่อง path ไทย/ยาวบน Windows)
                    if "Errno 2" in str(e):
                        shortp = _win_short_path(path_str)
                        if shortp != path_str:
                            ds = xr.open_dataset(shortp, engine="netcdf4")
                            path_str = shortp
                        else:
                            raise
                    else:
                        raise

                _VS30_CACHE.update({"kind": "grd", "ds": ds, "band": None, "path": path_str, "last_error": None})
                return True


                # 2) ลอง h5netcdf ถัดไป
                try:
                    ds = xr.open_dataset(p, engine="h5netcdf")
                    _VS30_CACHE.update({"kind": "grd", "ds": ds, "band": None, "path": str(p), "last_error": None})
                    return True
                except Exception as e:
                    errors["h5netcdf"] = str(e)

                # 3) ค่อยลอง default (เผื่อบางเครื่องใช้ได้)
                try:
                    ds = xr.open_dataset(p)
                    _VS30_CACHE.update({"kind": "grd", "ds": ds, "band": None, "path": str(p), "last_error": None})
                    return True
                except Exception as e:
                    errors["default"] = str(e)

                # 4) สุดท้ายค่อย scipy (แต่ถ้าไฟล์เป็น NetCDF4 มักไม่ผ่าน)
                try:
                    ds = xr.open_dataset(p, engine="scipy")
                    _VS30_CACHE.update({"kind": "grd", "ds": ds, "band": None, "path": str(p), "last_error": None})
                    return True
                except Exception as e:
                    errors["scipy"] = str(e)

                # ถ้าล้มหมด ให้เก็บ error จริงทุก engine (สำคัญมาก)
                _VS30_CACHE["last_error"] = f"read_error:{p.name}: " + " | ".join([f"{k}={v}" for k, v in errors.items()])
                continue

            except Exception as e:
                _VS30_CACHE["last_error"] = f"read_error:{p.name}:{e}"
                continue
            
    _VS30_CACHE["last_error"] = _VS30_CACHE.get("last_error") or "no_local_vs30_dataset"
    return False





def _sample_vs30(lat: float, lon: float):
    """คืนค่า (vs30_ms, source_text) หรือ (None, reason)."""
    if not _load_vs30_dataset():
        return None, (_VS30_CACHE.get("last_error") or "no_local_vs30_dataset")


    kind = _VS30_CACHE.get("kind")
    ds = _VS30_CACHE.get("ds")
    path = _VS30_CACHE.get("path") or ""

    if kind == "tif":
        try:
            val = list(ds.sample([(float(lon), float(lat))]))[0]
            v = float(val[0]) if val is not None and len(val) else float("nan")
            if not np.isfinite(v):
                return None, f"nodata:{Path(path).name}"
            return v, f"local_tif:{Path(path).name}"
        except Exception:
            return None, f"read_error:{Path(path).name}"

    if kind == "grd":
        try:
            import xarray as xr  # type: ignore
            import numpy as _np
            # เลือก data variable แรก (ส่วนมากคือ vs30)
            var_name = next(iter(ds.data_vars))
            da = ds[var_name]

            # หาชื่อแกน lat/lon ได้ทั้ง coords และ dims
            lat_candidates = ["lat", "latitude", "y"]
            lon_candidates = ["lon", "longitude", "x"]

            lat_name = next((n for n in lat_candidates if n in da.coords), None)
            lon_name = next((n for n in lon_candidates if n in da.coords), None)

            if lat_name is None:
                lat_name = next((n for n in lat_candidates if n in da.dims), None)
            if lon_name is None:
                lon_name = next((n for n in lon_candidates if n in da.dims), None)

            if lon_name is None or lat_name is None:
                return None, f"bad_coords:{Path(path).name}"

            # handle lon domain 0..360
            lon_q = float(lon)
            lon_vals = da[lon_name].values
            try:
                lon_min = float(_np.nanmin(lon_vals))
                lon_max = float(_np.nanmax(lon_vals))
                if lon_min >= 0 and lon_max > 180 and lon_q < 0:
                    lon_q = lon_q + 360.0
            except Exception:
                pass

            # ใช้ sel แบบ nearest (ไม่ต้องคำนวณ index เอง)
            v = da.sel({lat_name: float(lat), lon_name: float(lon_q)}, method="nearest").values
            v = float(_np.array(v).ravel()[0])

            if not _np.isfinite(v):
                return None, f"nodata:{Path(path).name}"
            return v, f"local_grd:{Path(path).name}"
        except Exception as e:
            return None, f"read_error:{Path(path).name}:{e}"


def get_soil_info(lat: float, lon: float) -> dict:
    """ข้อมูลชั้นดิน (Site Class) จากพิกัด"""
    vs30, src = _sample_vs30(lat, lon)
    sc = _site_class_from_vs30(vs30) if vs30 is not None else ""
    return {
        "lat": float(lat),
        "lon": float(lon),
        "vs30_ms": float(vs30) if vs30 is not None and np.isfinite(vs30) else None,
        "site_class": sc or None,
        "desc_th": _site_class_desc_th(sc) if sc else None,
        "source": src,
        "status": "ok" if sc else "no_data",
    }


# ---------- คำนวณ GMPE (CY08) ที่ “จุดที่กำหนด” ----------
def _pga_cy08_at_points(mag, depth, src_lat, src_lon, lat_arr, lon_arr):
    """คำนวณ PGA (หน่วย %g) ที่ชุดพิกัด lat_arr/lon_arr ตาม CY08"""
    strike = 0.0
    dip    = 45.0
    rake   = 0.0

    a_km2 = 10 ** ((mag - 4.07) / 0.98)
    w_km  = math.sqrt(a_km2 / 2.0)
    l_km  = 2.0 * w_km
    ztor  = depth - (w_km * math.sin(dip * math.pi/180.0) / 2.0)
    ztor  = max(0.0, ztor)

    c2 = 1.06; c3 = 3.45; c4 = -2.1; c4a = -0.5; crb = 50.0; chm = 3.0; cy3 = 4.0
    c1 = -1.2687; c1a = 0.1; c1b = -0.255; cn = 2.996; cm = 4.184; c5 = 6.16
    c6 = 0.4893; c7 = 0.0512; c7a = 0.086; c9 = 0.79; c9a = 1.5005
    c10 = -0.3218; cy1 = -0.00804; cy2 = -0.00785

    Frv = 1.0 if (30 <= rake <= 150) else 0.0
    Fnm = 1.0 if (-120 <= rake <= -60) else 0.0
    Ass = 0.0
    ff1 = c1 + (c1a * Frv + c1b * Fnm + c7 * (ztor - 4.0)) * (1.0 - Ass) + (c10 + c7a * (ztor - 4.0)) * Ass

    max1 = max(mag - chm, 0.0)
    xx = c6 * max1
    coshxx = (np.exp(xx) + np.exp(-xx)) / 2.0
    ff2 = c2 * (mag - 6.0) + ((c2 - c3) / cn) * np.log1p(np.exp(cn * (cm - mag)))

    DEG2RAD = math.pi / 180.0
    coslat = np.cos(lat_arr * DEG2RAD)
    dx_km = (src_lon - lon_arr) * coslat * 111.0
    dy_km = (src_lat - lat_arr) * 111.0
    dist_km = np.sqrt(dx_km**2 + dy_km**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ff33 = c4 * np.log(dist_km + c5 * coshxx)
        ff43 = (c4a - c4) * np.log(np.sqrt(dist_km**2 + crb**2))

    max2 = max(mag - cy3, 0.0)
    xxx = max2
    coshxxx = (np.exp(xxx) + np.exp(-xxx)) / 2.0
    ff53 = (cy1 + cy2 / (coshxxx)) * dist_km

    lnPGA_g = ff1 + ff2 + ff33 + ff43 + ff53
    PGA = np.exp(lnPGA_g) * 100.0
    return PGA

def compute_overlay_from_event(ev: dict):
    """
    รับ ev = {lat, lon, mag, depth, region, time_th, time_utc}
    คืนผล JSON-ready โดยเพิ่มขั้นตอน Interpolation (griddata)
    """
    lat = float(ev["lat"]); lon = float(ev["lon"]); mag = float(ev["mag"]); depth = float(ev["depth"])
    region_text = ev.get("region", "")
    time_th = ev.get("time_th", "")
    time_utc = ev.get("time_utc", "")

    # reverse geocode
    geo = reverse_geocode_th(lat, lon)
    tambon, amphoe, changwat = geo["tambon"], geo["amphoe"], geo["changwat"]
    if not (amphoe or changwat):
        t2, a2, c2 = _parse_tambon_from_text(region_text)
        tambon = tambon or t2
        amphoe = amphoe or a2
        changwat = changwat or c2

    # -------------------- กริดปลายทาง & กริดตัวอย่าง --------------------
    half_box_deg = 2.0 + 1.0 * max(0.0, mag - 4.0)
    latmin = lat - half_box_deg
    latmax = lat + half_box_deg
    lonmin = lon - half_box_deg
    lonmax = lon + half_box_deg

    tgt_spacing = float(TARGET_SPACING_DEG)
    n_lat_t = int(round((latmax - latmin) / tgt_spacing))
    n_lon_t = int(round((lonmax - lonmin) / tgt_spacing))
    if n_lat_t % 2 == 0: n_lat_t += 1
    if n_lon_t % 2 == 0: n_lon_t += 1
    lat_vals_t = latmin + tgt_spacing * np.arange(n_lat_t)
    lon_vals_t = lonmin + tgt_spacing * np.arange(n_lon_t)
    LAT_T, LON_T = np.meshgrid(lat_vals_t, lon_vals_t, indexing="ij")  # (n_lat, n_lon)

    base_spacing = tgt_spacing * max(1, int(SAMPLE_STEP))
    n_lat_s = int(round((latmax - latmin) / base_spacing))
    n_lon_s = int(round((lonmax - lonmin) / base_spacing))
    lat_vals_s = latmin + base_spacing * np.arange(n_lat_s)
    lon_vals_s = lonmin + base_spacing * np.arange(n_lon_s)
    LAT_S, LON_S = np.meshgrid(lat_vals_s, lon_vals_s, indexing="ij")

    PGA_S = _pga_cy08_at_points(mag, depth, lat, lon, LAT_S, LON_S)

    # -------------------- Interpolation (griddata) --------------------
    if USE_INTERPOLATION:
        points = np.column_stack([LON_S.ravel(), LAT_S.ravel()])
        values = PGA_S.ravel().astype(float)
        PGA_T = griddata(points, values, (LON_T, LAT_T), method="linear")
        nanmask = ~np.isfinite(PGA_T)
        if np.any(nanmask):
            PGA_T2 = griddata(points, values, (LON_T, LAT_T), method="nearest")
            PGA_T[nanmask] = PGA_T2[nanmask]
    else:
        PGA_T = _pga_cy08_at_points(mag, depth, lat, lon, LAT_T, LON_T)

    # Soft mask
    dist_ep = _haversine(lat, lon, LAT_T, LON_T)
    soft_mask = np.exp(-0.5 * (dist_ep / float(SOFTMASK_SIGMA_KM))**2)
    masked_pga = PGA_T * soft_mask
    masked_pga = np.where(np.isnan(masked_pga), 0.0, masked_pga)
    masked_pga = np.where(masked_pga < 0, 0.0, masked_pga)
    
      # --- AI: predict MMI over grid (optional) ---
    mmi_pred_grid = None
    mmi_level_grid = None

    grid_model, grid_model_region = _get_mmi_model_for_region(lat, lon)
    print(f"[MMI_GRID] selected region={grid_model_region} lat={lat} lon={lon}")

    if grid_model is not None:
        try:
            # masked_pga เป็น %g -> แปลงเป็น g
            pga_g = masked_pga / 100.0

            Xg_raw = pd.DataFrame({
                "dist": dist_ep.ravel(),
                "pga": pga_g.ravel(),
                "mag": np.full(masked_pga.size, float(mag)),
                "lat": LAT_T.ravel(),
                "lon": LON_T.ravel(),
            })

            Xg_pred_full = make_features(Xg_raw)

            if hasattr(grid_model, "feature_names_in_"):
                expected_cols = list(grid_model.feature_names_in_)
                for c in expected_cols:
                    if c not in Xg_pred_full.columns:
                        Xg_pred_full[c] = 0.0
                Xg_pred = Xg_pred_full[expected_cols]
            else:
                Xg_pred = Xg_pred_full

            pred = grid_model.predict(Xg_pred).reshape(masked_pga.shape)

            print("[DEBUG] grid before guardrail min/max =", float(np.nanmin(pred)), float(np.nanmax(pred)))

            pred_guarded = np.vectorize(apply_physical_guardrail)(
                pred,
                pga_g,
                dist_ep,
                float(mag),
            )
            print("[DEBUG] grid after guardrail min/max =", float(np.nanmin(pred_guarded)), float(np.nanmax(pred_guarded)))

            pred_guarded = np.clip(pred_guarded, 1.0, 12.0)

            # table level จาก PGA (%g)
            table_level_grid = np.zeros_like(masked_pga, dtype=int)
            for i in range(masked_pga.shape[0]):
                for j in range(masked_pga.shape[1]):
                    table_level_grid[i, j] = int(mmi_from_pga(float(masked_pga[i, j]))["bin_index"] + 1)

            model_level_grid = np.clip(np.rint(pred_guarded), 1, 12).astype(int)
            diff_grid = np.abs(model_level_grid - table_level_grid)

            # กฎใหม่: ถ้า PGA >= 40 %g ให้ใช้ table ทันที
            force_table_grid = masked_pga >= 40.0

            mmi_level_grid = np.where(
                force_table_grid | (diff_grid >= 2),
                table_level_grid,
                model_level_grid
            )

            mmi_pred_grid = np.where(
                force_table_grid | (diff_grid >= 2),
                table_level_grid.astype(float),
                pred_guarded
)
        except Exception as e:
            print("[MMI_MODEL] predict grid failed:", repr(e))
            mmi_pred_grid = None
            mmi_level_grid = None

            # ใส่ physical guardrail กลับ
            pred_guarded = np.vectorize(apply_physical_guardrail)(
                pred,
                pga_g,
                dist_ep,
                float(mag),
            )
            print("[DEBUG] grid after guardrail min/max =", float(np.nanmin(pred_guarded)), float(np.nanmax(pred_guarded)))

            pred_guarded = np.clip(pred_guarded, 1.0, 12.0)

            mmi_pred_grid = pred_guarded
            mmi_level_grid = np.clip(np.rint(pred_guarded), 1, 12).astype(int)

        

            
    pga_max = float(np.nanmax(masked_pga))

    # ----- Legend/Colormap -----
        # ----- Legend/Colormap -----
    if USE_CUSTOM_PGA_COLORS:
        # จัดการขอบเขตที่เป็นอินฟินิตี้สำหรับการแสดงผล legend
        finite_bounds = [float(b) for b in PGA_BOUNDS if np.isfinite(b)]
        # กำหนดช่วงแสดงผลสำหรับ legend
        vmin_show = float(finite_bounds[0])
        vmax_show = float(finite_bounds[-1]) if finite_bounds else float(pga_max) or 1e-6

        cmap = ListedColormap(PGA_COLORS)
        norm_show = BoundaryNorm(PGA_BOUNDS, cmap.N)

        # mid-points เพื่อความเข้ากันได้กับโค้ดเดิมที่ใช้ stops
        mids = []
        for i in range(len(PGA_COLORS)):
            lo = float(PGA_BOUNDS[i])
            hi = PGA_BOUNDS[i+1]
            if not np.isfinite(hi):
                hi = vmax_show * 1.2   # ใช้ค่าสูงกว่าเล็กน้อยเพื่อคำนวณ mid
            mids.append((float(lo) + float(hi)) / 2.0)

        # **สำคัญ**: ทำ bounds ให้ JSON-safe (ห้ามมี inf)
        bounds_out = []
        for b in PGA_BOUNDS:
            if np.isfinite(b):
                bounds_out.append(float(b))
            else:
                bounds_out.append(float(vmax_show))  # แทน inf ด้วย vmax_show

        legend = {
            "units": "%g",
            "min": float(vmin_show),
            "max": float(vmax_show),
            "mode": "discrete",
            "bounds": bounds_out,      # ไม่มี inf แล้ว → JSON-safe
            "colors": PGA_COLORS,
            "stops": [{"value": float(m), "color": PGA_COLORS[i]} for i, m in enumerate(mids)],
            "open_ended_last": True    # เผื่อ front-end อยากรู้ว่าช่วงสุดท้ายเดิมเป็น open-ended
        }
    else:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
        vmin_show = 0.0
        vmax_show = float(pga_max)
        if not np.isfinite(vmax_show) or vmax_show <= vmin_show:
            vmax_show = vmin_show + 1e-6
        norm_show = Normalize(vmin=vmin_show, vmax=vmax_show)

        tick_vals = np.linspace(vmin_show, vmax_show, 7)
        def _to_hex(rgb):
            r,g,b,a = rgb
            return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        legend = {
            "units": "%g",
            "min": float(vmin_show),
            "max": float(vmax_show),
            "stops": [{"value": float(v), "color": _to_hex(cmap(norm_show(v)))} for v in tick_vals],
        }


    # --- PGA grid points for nearest lookup ---
    pga_points = []
    for i in range(LAT_T.shape[0]):
        for j in range(LAT_T.shape[1]):
            pt = {
                "lat": float(LAT_T[i, j]),
                "lon": float(LON_T[i, j]),
                "pga": float(masked_pga[i, j]),
            }

            # ✅ เพิ่มตรงนี้ (แนบผล AI ถ้ามี)
            if mmi_level_grid is not None:
                pt["mmi_level"] = int(mmi_level_grid[i, j])   # 1..12
                pt["mmi_pred"]  = float(mmi_pred_grid[i, j])  # ค่าทศนิยม

            pga_points.append(pt)

    # วาดภาพโปร่งใสเป็น PNG (ใช้ cmap/norm จากด้านบน)
    # === บันทึกรูปแบบที่กรอบภาพ = กรอบพิกัดเป๊ะ ===
    fig = plt.figure(figsize=(8, 8), dpi=150)

    # แทน subplots ใช้ axes กินพื้นที่เต็มเฟรม 100% (ไม่มีขอบ/ไม่มี padding)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.imshow(
        masked_pga,
        extent=[lonmin, lonmax, latmin, latmax],   # x=lon, y=lat
        origin="lower",
        cmap=cmap,
        norm=norm_show,
        alpha=0.55,
        interpolation="bicubic"   # จะเปลี่ยนเป็น "bilinear" ก็ได้ถ้าภาพเบลอเกิน
    )

    # ล็อคกรอบแกนให้ตรงกับ extent (กัน Matplotlib แอบขยับ)
    ax.set_xlim(lonmin, lonmax)
    ax.set_ylim(latmin, latmax)

    # บันทึกแบบโปร่งใส โดย "ไม่ใช้" bbox_inches="tight"
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        transparent=True,
        facecolor="none",
        edgecolor="none",
        pad_inches=0
    )
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    time_th_fmt = _fmt_th_datetime(time_th) if time_th else "-"

    # --- Soil info at epicenter (Vs30 / Site Class) ---
    soil = None
    try:
        soil = get_soil_info(lat, lon)
    except Exception:
        soil = None

    popup_html = f"""
<div style="line-height:1.5; font-size:1.2em; color:red; padding:4px; text-align:center;">
  <strong>แผ่นดินไหว</strong><br>
</div>
<div style="line-height:1.2; font-size:1.05em">
  วันเวลา: <b>{time_th_fmt} น.</b><br>
  ขนาด: <b>{_fmt_num(mag, 2)}</b><br>
  จุดศูนย์กลาง: <b>{region_text}</b><br>
  ค่าระดับการสั่นสะเทือนสูงสุด: <b><span style="color:red;">{_fmt_num(pga_max, 4)}</span> %g</b><br>
  ชั้นดิน: <b>{(soil.get('site_class') if soil else None) or '-'}</b> {(('('+soil.get('desc_th')+')') if soil and soil.get('desc_th') else '')}<br>
  Vs30: <b>{(_fmt_num(soil.get('vs30_ms'), 0) if soil and soil.get('vs30_ms') is not None else '-')}</b> m/s<br>
</div>
""".strip()

    meta = {
        "time_th": time_th_fmt,
        "time_utc": time_utc or "-",
        "region_text": region_text,
        "tambon": tambon or "-",
        "amphoe": amphoe or "-",
        "changwat": changwat or "-",
        "lat": float(round(lat, 1)),
        "lon": float(round(lon, 1)),
        "mag": float(mag),
        "mag_api": float(mag),
        "depth_km": float(depth),
        "pga_max": float(round(pga_max, 2)),
        "mmi_model_region": grid_model_region,
    }

    return {
        "bounds": [[float(latmin), float(lonmin)], [float(latmax), float(lonmax)]],
        "epicenter": [float(lat), float(lon)],
        "image_data_url": data_url,
        "popup_html": popup_html,
        "meta": meta,
        "pga_points": pga_points,
        "legend": legend
    }

def run_pipeline():
    ev = fetch_latest_event_in_thailand()
    if not ev:
        raise RuntimeError("ไม่พบเหตุการณ์จาก TMD หรือโครงสร้างหน้าเว็บเปลี่ยนไป")
    result = compute_overlay_from_event(ev)
    return result


# ===================== เพิ่มสำหรับ "เหตุการณ์จำลอง" =====================
def _now_th_str():
    """เวลาปัจจุบันโซนไทย (UTC+7) ในฟอร์แมต YYYY-MM-DD HH:MM:SS"""
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")

def _now_utc_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def simulate_event(lat: float, lon: float, depth_km: float, mag: float, region_text: str | None = None):
    """
    สร้างเหตุการณ์จำลอง แล้วเรียก compute_overlay_from_event() คืนผลลัพธ์รูปแบบเดียวกับ /api/run
    """
    ev = {
        "lat": float(lat),
        "lon": float(lon),
        "depth": float(depth_km),
        "mag": float(mag),
        "region": region_text or f"จำลอง (Lat {lat:.4f}, Lon {lon:.4f})",
        "time_th": _now_th_str(),
        "time_utc": _now_utc_str()
    }
    return compute_overlay_from_event(ev)

def run_pipeline_manual(lat: float, lon: float, depth_km: float, mag: float, region_text: str | None = None):
    """ฟังก์ชันสะดวก ๆ สำหรับเรียกเหตุการณ์จำลองตรง ๆ"""
    return simulate_event(lat=lat, lon=lon, depth_km=depth_km, mag=mag, region_text=region_text)

# PGA (%g) → MMI (Worden+2012)
PGA_THRESH = [0.05, 0.3, 2.8, 6.2, 12, 22, 40, 75, 139]
MMI_CODES  = ["I","II–III","IV","V","VI","VII","VIII","IX","X","X+"]
TH_SHAKING = ["ไม่รู้สึก","อ่อนมาก–อ่อน","อ่อน","ปานกลาง","ค่อนข้างแรง",
              "แรงมาก","รุนแรง","รุนแรงมาก","รุนแรงมาก (Violent)","รุนแรงที่สุด (Extreme)"]
TH_DAMAGE  = ["ไม่มี","ไม่มี","ไม่มี","มีรอยแตกของผนังและกระจกหน้าต่างบ้างเล็กน้อย",
    "ผนัง/เพดานร่วงหล่นบางจุด หน้าต่างบานเกร็ดแตกหรือร้าว",
    "บ้าน/ตึกแถวที่ก่อสร้างไม่ได้มาตรฐานเริ่มเสียหายหนัก (ผนังแตกร้าว, เสาปูนบางต้นร้าว)"
    ,"อาคารที่สร้างไม่แข็งแรง (บ้านก่ออิฐไม่เสริมเหล็ก, ตึกแถวเก่า) เสียหายหนัก บางส่วนอาจพังทลาย"
    ,"อาคาร/บ้านเรือนที่ไม่ได้ออกแบบให้ทนแผ่นดินไหวเสียหายหนัก"
    ,"อาคาร/บ้านเรือนพังถล่มเกือบทั้งหมด","อาคาร/บ้านเรือนพังถล่ม"]

def mmi_from_pga(pga_percent_g: float):
    """รับค่า PGA เป็นหน่วย %g แล้วคืน dict ข้อมูล MMI"""
    idx = 0
    while idx < len(PGA_THRESH) and pga_percent_g >= PGA_THRESH[idx]:
        idx += 1
    code   = MMI_CODES[idx]
    shake  = TH_SHAKING[idx]
    damage = TH_DAMAGE[idx]
    lower  = 0 if idx == 0 else PGA_THRESH[idx-1]
    upper  = float("inf") if idx == len(PGA_THRESH) else PGA_THRESH[idx]
    range_text = f"< {PGA_THRESH[0]}%g" if idx == 0 else (f"≥ {lower}%g" if upper == float("inf") else f"{lower}–{upper}%g")
    return {
        "code": code,
        "shake_th": shake,
        "damage_th": damage,
        "range_text": range_text,
        "bin_index": idx
    }

def debug_vs30_paths():
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent / "static" / "data"
    candidates = [
        "vs30_global.grd",
        "global_vs30.grd",
        "vs30.grd",
        "vs30_thailand.tif",
        "vs30_global.tif",
        "global_vs30.tif",
        "vs30.tif",
    ]
    exists = {name: (data_dir / name).exists() for name in candidates}
    listing = []
    try:
        listing = sorted([p.name for p in data_dir.iterdir()])
    except Exception as e:
        listing = [f"(listdir error) {e}"]

    return {
        "logic_file": str(Path(__file__).resolve()),
        "data_dir": str(data_dir),
        "data_dir_exists": data_dir.exists(),
        "listing": listing,
        "candidate_exists": exists,
    }

def apply_physical_guardrail(pred: float, pga_g: float, dist_km: float, mag: float) -> float:
    # ถ้าแมกนิจูดต่ำมาก และ PGA ก็ต่ำจริง ให้เป็น MMI 1 ไปเลย
    if mag < 2.0 and pga_g <= 0.001:
        return 1.0

    pred = cap_mmi_from_pga_g(pred, pga_g)

    if dist_km >= 150 and pga_g <= 0.002:
        pred = min(pred, 2.5)
    if dist_km >= 100 and pga_g <= 0.005:
        pred = min(pred, 3.0)

    return pred

def cap_mmi_from_pga_g(pred: float, pga_g: float) -> float:
    """
    คุมเพดาน MMI จากค่า PGA โดยตรง
    pga_g เป็นหน่วย g
    """

    # threshold ชุดเริ่มต้นสำหรับกันเคสเวอร์
    if pga_g <= 0.0001:      # <= 0.01 %g
        return min(pred, 1.0)
    elif pga_g <= 0.0003:    # <= 0.03 %g
        return min(pred, 2.5)
    elif pga_g <= 0.0030:    # <= 0.30 %g
        return min(pred, 3.0)
    elif pga_g <= 0.0280:    # <= 2.80 %g
        return min(pred, 4.0)

    return pred

def _hybrid_mmi_from_model_and_table(pred_mmi: float, pga_percent_g: float):
    """
    pred_mmi      : ค่าทศนิยมจากโมเดล (1..12)
    pga_percent_g : PGA หน่วย %g

    กติกา:
    - ถ้า PGA >= 40 %g -> ยึดค่าจากตารางทันที (VIII / IX / X+)
    - ถ้า model กับ table ต่างกัน >= 2 ระดับ -> ใช้ table
    - ถ้าต่างกัน 0 หรือ 1 ระดับ -> ใช้ model
    """
    pred_mmi = float(np.clip(pred_mmi, 1.0, 12.0))
    model_level = int(np.clip(np.rint(pred_mmi), 1, 12))

    table_info = mmi_from_pga(float(pga_percent_g))
    table_level = int(table_info["bin_index"] + 1)

    diff = abs(model_level - table_level)

    # กฎใหม่: PGA สูงมากให้ยึดตารางเลย
    if float(pga_percent_g) >= 40.0:
        final_level = table_level
        final_pred = float(table_level)
        source = "table_forced_high_pga"

    elif diff >= 2:
        final_level = table_level
        final_pred = float(table_level)
        source = "table"

    else:
        final_level = model_level
        final_pred = pred_mmi
        source = "model"

    return {
        "mmi_pred": float(final_pred),
        "mmi_level": int(final_level),
        "model_level": int(model_level),
        "table_level": int(table_level),
        "diff": int(diff),
        "source": source,
        "table": table_info,
    }

def predict_mmi_ai(dist: float, pga: float, mag: float = None, lat: float = None, lon: float = None):
    """
    predict MMI จาก dist/pga/(mag)/lat/lon
    - pga ที่รับเข้ามาเป็นหน่วย g
    - รองรับโมเดลที่ train มาไม่เท่ากันเรื่อง feature columns
    """
    model, model_region = _get_mmi_model_for_region(lat, lon)
    print(f"[MMI_POINT] selected region={model_region} lat={lat} lon={lon}")

    if model is None:
        return None

    row = {
        "dist": float(dist),
        "pga": float(pga),
    }

    if mag is not None:
        row["mag"] = float(mag)
    if lat is not None:
        row["lat"] = float(lat)
    if lon is not None:
        row["lon"] = float(lon)

    try:
        X_raw = pd.DataFrame([row])
        X_pred_full = make_features(X_raw)

        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            for c in expected_cols:
                if c not in X_pred_full.columns:
                    X_pred_full[c] = 0.0
            X_pred = X_pred_full[expected_cols]
        else:
            X_pred = X_pred_full

        pred = float(model.predict(X_pred)[0])
        print("[DEBUG] before guardrail =", pred)

        pred = apply_physical_guardrail(
            pred=pred,
            pga_g=float(pga),
            dist_km=float(dist),
            mag=float(mag) if mag is not None else 5.0,
        )
        print("[DEBUG] after guardrail =", pred)

        pred = float(np.clip(pred, 1.0, 12.0))

        hybrid = _hybrid_mmi_from_model_and_table(
            pred_mmi=pred,
            pga_percent_g=float(pga) * 100.0,   # เพราะ pga ในฟังก์ชันนี้เป็นหน่วย g
        )

        return hybrid

    except Exception as e:
        print("[MMI_MODEL] predict point failed:", repr(e))
        return None
    
