#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Cyftee
darkpfp_gui.py — GUI для стилю darkpfp із покращеними контурами, пресетами (INI) і пакетною обробкою папок.

Фішки:
- Контури: Fusion / Canny / HED (опційно: deploy.prototxt + hed_pretrained_bsds.caffemodel)
- Foreground підсилення: saliency (opencv-contrib) + опційна сегментація (U2Net ONNX / GrabCut)
- Thinning (скелетизація)
- Апскейл 1×/2×/4× перед стилізацією; квадрат: пад або розтяг
- Стилі: Neon (darkpfp) / Line Art (біле на чорному)
- Пресети: збереження/видалення у INI, пам'ять останніх шляхів
- **НОВЕ**: Пакетна обробка папки → вибір вхідної та вихідної директорій, масове збереження

Встановлення (рекомендовано):
    pip uninstall -y opencv-python
    pip install opencv-contrib-python PyQt5 pillow numpy

(Опційно для HED) — покласти у папку скрипта:
    - hed_pretrained_bsds.caffemodel
    - deploy.prototxt (або hed.prototxt)

(Опційно для Segmentation) — покласти у папку скрипта: u2net.onnx (або u2netp.onnx)

Запуск:
    python darkpfp_gui.py
"""
import sys, os
from pathlib import Path
import configparser
import numpy as np
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

# ======= Конфіг/INI =======
SCRIPT_DIR = Path(__file__).resolve().parent
INI_PATH = SCRIPT_DIR / "darkpfp_gui.ini"

def _bool(v, default=False):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1","true","yes","y","on"): return True
    if s in ("0","false","no","n","off"): return False
    return default

def load_ini():
    cfg = configparser.ConfigParser()
    if INI_PATH.exists():
        try:
            cfg.read(INI_PATH, encoding="utf-8")
        except Exception:
            cfg = configparser.ConfigParser()
    return cfg

def save_ini(cfg):
    with open(INI_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)

def load_user_presets(cfg):
    presets = {}
    for sect in cfg.sections():
        if sect.startswith("preset:"):
            name = sect.split("preset:",1)[1]
            p = {}
            p["size"] = int(cfg.get(sect,"size", fallback="1024"))
            p["square"] = _bool(cfg.get(sect,"square", fallback="True"), True)
            p["stretch"] = _bool(cfg.get(sect,"stretch", fallback="False"), False)
            p["tint"] = cfg.get(sect,"tint", fallback="#E7C2FF")
            p["line"] = float(cfg.get(sect,"line", fallback="1.8"))
            p["glow"] = float(cfg.get(sect,"glow", fallback="12.0"))
            p["glow_gain"] = float(cfg.get(sect,"glow_gain", fallback="1.35"))
            p["softness"] = float(cfg.get(sect,"softness", fallback="0.6"))
            p["edges_mode"] = cfg.get(sect,"edges_mode", fallback="Fusion")
            p["out_mode"] = cfg.get(sect,"out_mode", fallback="neon")  # neon|lineart
            p["edge_boost"] = _bool(cfg.get(sect,"edge_boost", fallback="False"), False)
            p["edge_thin"]  = _bool(cfg.get(sect,"edge_thin",  fallback="False"), False)
            p["edge_seg"]   = _bool(cfg.get(sect,"edge_seg",   fallback="False"), False)
            p["mask_strength"] = float(cfg.get(sect,"mask_strength", fallback="0.7"))
            # edge_params
            ep = {}
            def gf(k, fb): 
                try: return float(cfg.get(sect,k,fallback=str(fb)))
                except Exception: return fb
            def gi(k, fb):
                try: return int(cfg.get(sect,k,fallback=str(fb)))
                except Exception: return fb
            ep["canny_sigma"] = gf("edge_canny_sigma", 0.28)
            ep["edge_thresh"] = gf("edge_edge_thresh", 0.22)
            ep["clahe_clip"]  = gf("edge_clahe_clip", 2.2)
            ep["clahe_grid"]  = gi("edge_clahe_grid", 8)
            ep["ms_levels"]   = gi("edge_ms_levels", 2)
            ep["chroma_w"]    = gf("edge_chroma_w", 0.5)
            ep["log_sigma"]   = gf("edge_log_sigma", 1.0)
            ep["median_ksize"]= gi("edge_median_ksize", 3)
            ep["morph_dilate"]= gi("edge_morph_dilate", 2)
            ep["morph_erode"] = gi("edge_morph_erode", 0)
            p["edge_params"] = ep
            presets[name] = p
    return presets

def save_user_preset(cfg, name, p):
    sect = f"preset:{name}"
    if not cfg.has_section(sect):
        cfg.add_section(sect)
    cfg.set(sect,"size", str(int(p.get("size",1024))))
    cfg.set(sect,"square", str(bool(p.get("square",True))))
    cfg.set(sect,"stretch", str(bool(p.get("stretch",False))))
    cfg.set(sect,"tint", p.get("tint","#E7C2FF"))
    cfg.set(sect,"line", str(float(p.get("line",1.8))))
    cfg.set(sect,"glow", str(float(p.get("glow",12.0))))
    cfg.set(sect,"glow_gain", str(float(p.get("glow_gain",1.35))))
    cfg.set(sect,"softness", str(float(p.get("line_soft",0.6))))
    cfg.set(sect,"edges_mode", p.get("edges_mode","Fusion"))
    cfg.set(sect,"out_mode", p.get("out_mode","neon"))
    cfg.set(sect,"edge_boost", str(bool(p.get("edge_boost", False))))
    cfg.set(sect,"edge_thin",  str(bool(p.get("edge_thin", False))))
    cfg.set(sect,"edge_seg",   str(bool(p.get("edge_seg", False))))
    cfg.set(sect,"mask_strength", str(float(p.get("mask_strength", 0.7))))
    ep = p.get("edge_params", {})
    def s(k, v): cfg.set(sect,k, str(v))
    s("edge_canny_sigma", float(ep.get("canny_sigma",0.28)))
    s("edge_edge_thresh", float(ep.get("edge_thresh",0.22)))
    s("edge_clahe_clip",  float(ep.get("clahe_clip",2.2)))
    s("edge_clahe_grid",  int(ep.get("clahe_grid",8)))
    s("edge_ms_levels",   int(ep.get("ms_levels",2)))
    s("edge_chroma_w",    float(ep.get("chroma_w",0.5)))
    s("edge_log_sigma",   float(ep.get("log_sigma",1.0)))
    s("edge_median_ksize",int(ep.get("median_ksize",3)))
    s("edge_morph_dilate",int(ep.get("morph_dilate",2)))
    s("edge_morph_erode", int(ep.get("morph_erode",0)))

def delete_user_preset(cfg, name):
    sect = f"preset:{name}"
    if cfg.has_section(sect):
        cfg.remove_section(sect)

# ======= Ядро I/O та утиліти =======
def read_image_any(path_str):
    p = Path(path_str)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass
    pil = Image.open(p).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def write_image_any(path_str, bgr):
    p = Path(path_str); p.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ext = p.suffix.lower()
    if ext in (".jpg",".jpeg"):
        ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        ok, buf = cv2.imencode(".png", rgb)
    if not ok: raise RuntimeError("Помилка кодування вихідного зображення.")
    p.write_bytes(buf.tobytes())

def to_square_pad(img, color=(0,0,0)):
    h,w = img.shape[:2]
    if h==w: return img
    size = max(h,w)
    canvas = np.zeros((size,size,3), dtype=img.dtype); canvas[:,:,:] = color
    y = (size-h)//2; x = (size-w)//2
    canvas[y:y+h, x:x+w] = img
    return canvas

def to_square_stretch(img, out_size):
    if out_size <= 0:
        h,w = img.shape[:2]; out_size = max(h,w)
    return cv2.resize(img, (int(out_size), int(out_size)), interpolation=cv2.INTER_AREA)

def add_scanlines(img, strength=0.12, period=3):
    if strength<=0: return img
    h,w = img.shape[:2]; mask = np.ones((h,1), dtype=np.float32)
    for y in range(h):
        if (y % max(1,period)) == 0: mask[y,0] = 1.0 - strength
    mask = np.repeat(mask, w, axis=1)
    out = (img.astype(np.float32)/255.0) * mask[...,None]
    return np.clip(out*255.0,0,255).astype(np.uint8)

def add_vignette(img, amount=0.35):
    if amount<=0: return img
    h,w = img.shape[:2]; Y, X = np.ogrid[:h,:w]; cy, cx = h/2, w/2
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2); maxd = np.sqrt(cx**2 + cy**2)
    mask = 1.0 - amount*(dist/maxd)**1.2; mask = np.clip(mask, 0.2, 1.0).astype(np.float32)
    out = (img.astype(np.float32)/255.0) * mask[...,None]
    return np.clip(out*255.0,0,255).astype(np.uint8)

def add_grain(img, sigma=5.0):
    if sigma<=0: return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def color_from_hex(hexstr, default=(235,200,255)):
    if not hexstr: return (235,200,255)
    s = hexstr.strip()
    if s.startswith("#") and len(s)==7:
        r = int(s[1:3],16); g=int(s[3:5],16); b=int(s[5:7],16)
        return (b,g,r)
    return default

# ======= Контури =======
def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    return cv2.Canny(gray, int(max(0,(1.0-sigma)*v)), int(min(255,(1.0+sigma)*v)))

def _post_morph(ed, params):
    morph_d = int(params.get("morph_dilate", 2))
    morph_e = int(params.get("morph_erode", 0))
    if morph_d>0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_d, morph_d))
        ed = cv2.dilate(ed, kd, 1)
    if morph_e>0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_e, morph_e))
        ed = cv2.erode(ed, ke, 1)
    return ed

def _median_cleanup(ed, params):
    med_ks = int(params.get("median_ksize", 3))
    if med_ks and med_ks>=3 and (med_ks % 2)==1:
        ed = cv2.medianBlur(ed, med_ks)
    return ed

def canny_edges(bgr, params):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    ed = auto_canny(gray, sigma=float(params.get("canny_sigma",0.28)))
    ed = _post_morph(ed, params)
    ed = _median_cleanup(ed, params)
    return ed

def fusion_edges(bgr, params):
    clahe_clip = float(params.get("clahe_clip", 2.2))
    clahe_grid = int(params.get("clahe_grid", 8))
    ms_levels  = int(params.get("ms_levels", 2))
    chroma_w   = float(params.get("chroma_w", 0.5))
    log_sigma  = float(params.get("log_sigma", 1.0))
    edge_thresh= float(params.get("edge_thresh", 0.22))
    canny_sigma= float(params.get("canny_sigma", 0.28))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,_,_ = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); _,_,V = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clahe_clip), tileGridSize=(max(2,clahe_grid), max(2,clahe_grid)))
    Lc = clahe.apply(L)

    eL = auto_canny(Lc, sigma=canny_sigma)
    eV = auto_canny(V,  sigma=canny_sigma)

    gx = cv2.Scharr(Lc, cv2.CV_32F, 1, 0); gy = cv2.Scharr(Lc, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy); mag = (mag / (mag.max()+1e-6))
    eS = (mag >= edge_thresh).astype(np.uint8)*255

    if log_sigma > 0:
        Lg = cv2.GaussianBlur(Lc, (0,0), log_sigma)
        lap = cv2.Laplacian(Lg, cv2.CV_32F, ksize=3)
        lapn = np.abs(lap); lapn = lapn / (lapn.max()+1e-6)
        eLoG = (lapn >= edge_thresh).astype(np.uint8)*255
    else:
        eLoG = np.zeros_like(eL)

    ems = np.zeros_like(eL)
    h, w = Lc.shape[:2]
    for lvl in range(1, ms_levels+1):
        s = 1.0/(2**lvl)
        small = cv2.resize(Lc, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
        es = auto_canny(small, sigma=canny_sigma)
        es_up = cv2.resize(es, (w,h), interpolation=cv2.INTER_LINEAR)
        ems = np.maximum(ems, es_up)

    fused = np.maximum.reduce([
        eL.astype(np.float32)/255.0,
        (eV.astype(np.float32)/255.0) * chroma_w,
        eS.astype(np.float32)/255.0,
        eLoG.astype(np.float32)/255.0,
        ems.astype(np.float32)/255.0
    ])
    out = (np.clip(fused, 0, 1)*255).astype(np.uint8)
    out = _post_morph(out, params)
    out = _median_cleanup(out, params)
    return out

# --- HED (DL) ---
_HED_NET = None
def _try_load_hed():
    global _HED_NET
    if _HED_NET is not None:
        return _HED_NET
    here = SCRIPT_DIR
    protos = ["deploy.prototxt", "hed.prototxt"]
    models = ["hed_pretrained_bsds.caffemodel", "hed.caffemodel"]
    proto = None; model = None
    for name in protos:
        p = here / name
        if p.exists(): proto = str(p); break
    for name in models:
        m = here / name
        if m.exists(): model = str(m); break
    if proto and model:
        try:
            _HED_NET = cv2.dnn.readNetFromCaffe(proto, model)
            return _HED_NET
        except Exception:
            _HED_NET = None
            return None
    return None

def hed_edges(bgr, params):
    net = _try_load_hed()
    if net is None:
        return fusion_edges(bgr, params)
    H, W = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, size=(W, H),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    ed = out[0,0,:,:]
    ed = (ed - ed.min())/(ed.max()-ed.min()+1e-6)
    ed = (ed*255).astype(np.uint8)
    ed = _post_morph(ed, params)
    ed = _median_cleanup(ed, params)
    return ed

# --- Segmentation (U2Net/GrabCut) ---
_U2_NET = None
def _try_load_u2net():
    global _U2_NET
    if _U2_NET is not None:
        return _U2_NET
    for name in ("u2net.onnx","u2netp.onnx"):
        p = SCRIPT_DIR / name
        if p.exists():
            try:
                _U2_NET = cv2.dnn.readNetFromONNX(str(p))
                return _U2_NET
            except Exception:
                _U2_NET = None
    return None

def u2net_mask(bgr):
    net = _try_load_u2net()
    if net is None:
        return None
    H,W = bgr.shape[:2]
    inp = cv2.dnn.blobFromImage(bgr, scalefactor=1/255.0, size=(320,320), swapRB=True, crop=False)
    net.setInput(inp)
    pr = net.forward()
    if pr.ndim == 4:
        pr = pr[0,0,:,:]
    pr = cv2.resize(pr, (W,H), interpolation=cv2.INTER_LINEAR)
    pr = (pr - pr.min())/(pr.max()-pr.min()+1e-6)
    mask = np.clip(pr, 0, 1).astype(np.float32)
    return mask

_HAS_SALIENCY = hasattr(cv2, "saliency")

def saliency_map(bgr):
    if not _HAS_SALIENCY: return None
    try:
        obj = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, sal = obj.computeSaliency(bgr)
        if not ok: return None
        sal = (sal - sal.min())/(sal.max()-sal.min()+1e-6)
        sal = cv2.GaussianBlur(sal, (0,0), 1.2)
        return np.clip(sal, 0, 1).astype(np.float32)
    except Exception:
        return None

def grabcut_refine(bgr, init_mask=None, iters=1):
    h,w = bgr.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    if init_mask is None:
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        cv2.grabCut(bgr, mask, rect, np.zeros((1,65),np.float64), np.zeros((1,65),np.float64), iters, cv2.GC_INIT_WITH_RECT)
    else:
        mask[:] = cv2.GC_PR_BGD
        mask[init_mask>0.6] = cv2.GC_PR_FGD
        cv2.grabCut(bgr, mask, None, np.zeros((1,65),np.float64), np.zeros((1,65),np.float64), iters, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    return fg

def build_foreground_mask(bgr):
    m = u2net_mask(bgr)
    if m is not None:
        try:
            m = grabcut_refine(bgr, m, iters=1)
        except Exception:
            pass
        return m
    s = saliency_map(bgr)
    if s is not None:
        try:
            s = grabcut_refine(bgr, s, iters=1)
        except Exception:
            pass
        return s
    return grabcut_refine(bgr, None, iters=1)

_HAS_XIMGPROC = hasattr(cv2, "ximgproc")
def thinning(img):
    if _HAS_XIMGPROC:
        return cv2.ximgproc.thinning(img)
    img = (img>0).astype(np.uint8)*255
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    bw = img.copy()
    while True:
        eroded = cv2.erode(bw, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(bw, temp)
        skel = cv2.bitwise_or(skel, temp)
        bw = eroded.copy()
        if cv2.countNonZero(bw) == 0:
            break
    return skel

def build_edges(bgr, mode="Fusion", params=None, *, boost=False, thin=False, seg=False, mask_strength=0.7):
    params = params or {}
    if mode == "HED":
        ed = hed_edges(bgr, params)
    elif mode == "Canny":
        ed = canny_edges(bgr, params)
    else:
        ed = fusion_edges(bgr, params)

    if boost:
        sm = saliency_map(bgr)
        if sm is not None:
            e = ed.astype(np.float32)/255.0
            e *= (0.4 + 0.6*sm)
            ed = (np.clip(e,0,1)*255).astype(np.uint8)

    if seg:
        fm = build_foreground_mask(bgr)
        if fm is not None:
            e = ed.astype(np.float32)/255.0
            s = float(mask_strength)
            e *= ( (1.0 - s) + s*fm )
            ed = (np.clip(e,0,1)*255).astype(np.uint8)

    if thin:
        ed = thinning(ed)
    return ed

# ======= Рендери =======
def render_neon(bgr, *, size=1024, square=True, stretch_square=False,
                edges_mode="Fusion", edge_params=None,
                edge_boost=False, edge_thin=False, edge_seg=False, mask_strength=0.7,
                line_width=1.8, glow_radius=12, glow_gain=1.35,
                tint_hex="#E7C2FF"):
    edge_params = edge_params or {}
    h,w = bgr.shape[:2]
    if stretch_square:
        target = size if size and size>0 else max(h,w)
        bgr = to_square_stretch(bgr, target)
    else:
        if size and size>0:
            s = size/max(h,w)
            if s!=1.0:
                bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        if square:
            bgr = to_square_pad(bgr, color=(0,0,0))

    edges = build_edges(bgr, mode=edges_mode, params=edge_params, boost=edge_boost, thin=edge_thin, seg=edge_seg, mask_strength=mask_strength)
    if line_width>1:
        k = max(1, int(round(line_width)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        edges = cv2.dilate(edges, kernel, 1)

    e = edges.astype(np.float32)/255.0
    tint = np.array(color_from_hex(tint_hex))[None,None,:]/255.0
    gl = cv2.GaussianBlur(e, (int(glow_radius*2+1)|1, int(glow_radius*2+1)|1), glow_radius) if glow_radius>0 else e
    neon = np.clip(e + gl*glow_gain, 0, 1)
    neon_rgb = (neon[...,None] * tint)
    neon_bgr = (neon_rgb[...,::-1]*255.0).astype(np.uint8)

    neon_bgr = add_scanlines(neon_bgr, strength=0.12, period=3)
    neon_bgr = add_vignette(neon_bgr, amount=0.35)
    neon_bgr = add_grain(neon_bgr, sigma=5.0)
    return neon_bgr

def render_lineart(bgr, *, size=1024, square=True, stretch_square=False,
                   edges_mode="Fusion", edge_params=None,
                   edge_boost=False, edge_thin=False, edge_seg=False, mask_strength=0.7,
                   line_width=2.0, softness=0.6):
    edge_params = edge_params or {}
    h,w = bgr.shape[:2]
    if stretch_square:
        target = size if size and size>0 else max(h,w)
        bgr = to_square_stretch(bgr, target)
    else:
        if size and size>0:
            s = size/max(h,w)
            if s!=1.0:
                bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        if square:
            bgr = to_square_pad(bgr, color=(0,0,0))

    edges = build_edges(bgr, mode=edges_mode, params=edge_params, boost=edge_boost, thin=edge_thin, seg=edge_seg, mask_strength=mask_strength)
    if line_width>1:
        k = max(1, int(round(line_width)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        edges = cv2.dilate(edges, kernel, 1)

    e = edges.astype(np.float32)/255.0
    if softness and softness>0:
        e = cv2.GaussianBlur(e, (int(softness*2+1)|1, int(softness*2+1)|1), softness)
        e = np.clip(e, 0, 1)
    line = (e*255.0).astype(np.uint8)
    return cv2.merge([line, line, line])

# ======= Вбудовані пресети =======
BUILTIN_PRESETS = {
    "darkpfp-exact": {
        "size": 1024, "square": True, "tint": "#E7C2FF",
        "line": 1.8, "glow": 12.0, "glow_gain": 1.35,
        "edges_mode": "Fusion",
        "edge_params": {"canny_sigma":0.28, "clahe_clip":2.2, "clahe_grid":8, "ms_levels":2,
                        "chroma_w":0.5, "log_sigma":1.0, "edge_thresh":0.22,
                        "median_ksize":3, "morph_dilate":2, "morph_erode":0},
        "out_mode": "neon",
        "edge_boost": False, "edge_thin": False, "edge_seg": False, "mask_strength":0.7
    },
    "lineart-clean": {
        "size": 1024, "square": True,
        "edges_mode": "Fusion",
        "edge_params": {"canny_sigma":0.30, "clahe_clip":2.0, "clahe_grid":8, "ms_levels":2,
                        "chroma_w":0.4, "log_sigma":0.8, "edge_thresh":0.22,
                        "median_ksize":3, "morph_dilate":2, "morph_erode":0},
        "line": 2.0, "softness": 0.6,
        "out_mode": "lineart",
        "edge_boost": False, "edge_thin": False, "edge_seg": False, "mask_strength":0.7
    }
}

# ======= Воркери =======
class RenderWorker(QtCore.QObject):
    preview_ready = QtCore.pyqtSignal(np.ndarray)
    saved = QtCore.pyqtSignal(np.ndarray, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, src_path, upscale, params, save_path=None, parent=None):
        super().__init__(parent)
        self.src_path = src_path
        self.upscale = upscale
        self.params = params
        self.save_path = save_path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            img = read_image_any(self.src_path)
            if img is None:
                raise RuntimeError("Не вдалося прочитати вхідне зображення.")
            if self.upscale in (2,4):
                img = cv2.resize(img, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LANCZOS4)

            p = self.params
            out_mode = p.get("out_mode","neon")

            if out_mode == "lineart":
                out = render_lineart(
                    img, size=p["size"], square=p["square"], stretch_square=p.get("stretch", False),
                    edges_mode=p["edges_mode"], edge_params=p["edge_params"],
                    edge_boost=p.get("edge_boost", False), edge_thin=p.get("edge_thin", False),
                    edge_seg=p.get("edge_seg", False), mask_strength=p.get("mask_strength", 0.7),
                    line_width=int(round(p["line"])), softness=float(p.get("line_soft",0.6))
                )
            else:
                out = render_neon(
                    img, size=p["size"], square=p["square"], stretch_square=p.get("stretch", False),
                    edges_mode=p["edges_mode"], edge_params=p["edge_params"],
                    edge_boost=p.get("edge_boost", False), edge_thin=p.get("edge_thin", False),
                    edge_seg=p.get("edge_seg", False), mask_strength=p.get("mask_strength", 0.7),
                    line_width=p["line"], glow_radius=p["glow"], glow_gain=p["glow_gain"],
                    tint_hex=p["tint"]
                )

            self.preview_ready.emit(out)

            if self.save_path:
                out_path = Path(self.save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_image_any(str(out_path), out)
                self.saved.emit(out, str(out_path.resolve()))
        except Exception as e:
            self.failed.emit(str(e))

class BatchWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)  # i, total, message
    finished = QtCore.pyqtSignal(int, int)       # saved, skipped
    failed = QtCore.pyqtSignal(str)

    def __init__(self, files, out_dir, upscale, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.out_dir = Path(out_dir)
        self.upscale = upscale
        self.params = params
        self._cancel = False

    @QtCore.pyqtSlot()
    def run(self):
        saved = 0; skipped = 0
        total = len(self.files)
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(self.files, 1):
                if self._cancel:
                    break
                try:
                    img = read_image_any(f)
                    if img is None:
                        skipped += 1
                        self.progress.emit(i, total, f"Пропущено (не прочитано): {f}")
                        continue

                    if self.upscale in (2,4):
                        img = cv2.resize(img, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LANCZOS4)

                    p = self.params
                    out_mode = p.get("out_mode","neon")
                    if out_mode == "lineart":
                        out = render_lineart(
                            img, size=p["size"], square=p["square"], stretch_square=p.get("stretch", False),
                            edges_mode=p["edges_mode"], edge_params=p["edge_params"],
                            edge_boost=p.get("edge_boost", False), edge_thin=p.get("edge_thin", False),
                            edge_seg=p.get("edge_seg", False), mask_strength=p.get("mask_strength", 0.7),
                            line_width=int(round(p["line"])), softness=float(p.get("line_soft",0.6))
                        )
                        suffix = "_lineart"
                    else:
                        out = render_neon(
                            img, size=p["size"], square=p["square"], stretch_square=p.get("stretch", False),
                            edges_mode=p["edges_mode"], edge_params=p["edge_params"],
                            edge_boost=p.get("edge_boost", False), edge_thin=p.get("edge_thin", False),
                            edge_seg=p.get("edge_seg", False), mask_strength=p.get("mask_strength", 0.7),
                            line_width=p["line"], glow_radius=p["glow"], glow_gain=p["glow_gain"],
                            tint_hex=p["tint"]
                        )
                        suffix = "_neon"

                    in_name = Path(f).stem
                    out_path = self.out_dir / f"{in_name}{suffix}.png"
                    write_image_any(str(out_path), out)
                    saved += 1
                    self.progress.emit(i, total, f"Збережено: {out_path.name}")
                except Exception as e:
                    skipped += 1
                    self.progress.emit(i, total, f"Помилка на {Path(f).name}: {e}")
            self.finished.emit(saved, skipped)
        except Exception as e:
            self.failed.emit(str(e))

    def cancel(self):
        self._cancel = True

# ======= GUI =======
def labeled_slider(parent, title, minv, maxv, step, init, decimals=1):
    box = QtWidgets.QGroupBox(title, parent)
    lay = QtWidgets.QHBoxLayout(box)
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); slider.setMinimum(0); slider.setMaximum(int((maxv-minv)/step))
    spin = QtWidgets.QDoubleSpinBox(); spin.setRange(minv, maxv); spin.setSingleStep(step); spin.setDecimals(decimals); spin.setValue(init)
    def spin_to_slider(val):
        slider.blockSignals(True); slider.setValue(int(round((val-minv)/step))); slider.blockSignals(False)
    def slider_to_spin(v):
        spin.blockSignals(True); spin.setValue(minv + v*step); spin.blockSignals(False)
    spin.valueChanged.connect(spin_to_slider); slider.valueChanged.connect(slider_to_spin)
    spin_to_slider(init)
    lay.addWidget(slider); lay.addWidget(spin, 0)
    return box, spin

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("darkpfp — покращені контури + пресети (INI) + пакетна обробка")
        self.resize(1260, 800)

        # INI
        self.cfg = load_ini()
        self.user_presets = load_user_presets(self.cfg)

        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        # Ліва панель (скрол)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        h.addWidget(scroll, 0)
        host = QtWidgets.QWidget(); scroll.setWidget(host)
        left = QtWidgets.QVBoxLayout(host)

        # --- Основне ---
        box_basic = QtWidgets.QGroupBox("Основне")
        lyb = QtWidgets.QVBoxLayout(box_basic)

        row1 = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("Відкрити…"); self.btn_open.clicked.connect(self.open_image)
        self.btn_save = QtWidgets.QPushButton("Зберегти…"); self.btn_save.clicked.connect(self.process_and_save)
        self.btn_batch = QtWidgets.QPushButton("Пакетна обробка…"); self.btn_batch.clicked.connect(self.on_batch)
        row1.addWidget(self.btn_open); row1.addWidget(self.btn_save); row1.addWidget(self.btn_batch)
        lyb.addLayout(row1)

        self.combo_upscale = QtWidgets.QComboBox(); self.combo_upscale.addItems(["1x","2x","4x"])
        self.spin_size = QtWidgets.QSpinBox(); self.spin_size.setRange(0,8192); self.spin_size.setValue(1024)
        self.chk_square = QtWidgets.QCheckBox("Пад до 1:1 (поля)"); self.chk_square.setChecked(True)
        self.chk_stretch = QtWidgets.QCheckBox("Розтягнути до 1:1 (без полів)")
        lyb.addWidget(QtWidgets.QLabel("Апскейл + Розмір:"))
        row_s = QtWidgets.QHBoxLayout(); row_s.addWidget(self.combo_upscale); row_s.addWidget(self.spin_size); lyb.addLayout(row_s)
        lyb.addWidget(self.chk_square); lyb.addWidget(self.chk_stretch)

        # Пресети
        self.combo_preset = QtWidgets.QComboBox()
        self.refresh_preset_combo()
        self.combo_preset.currentIndexChanged.connect(self.apply_current_preset)

        rowp = QtWidgets.QHBoxLayout()
        self.btn_save_preset = QtWidgets.QPushButton("Зберегти як пресет…"); self.btn_save_preset.clicked.connect(self.on_save_preset)
        self.btn_del_preset = QtWidgets.QPushButton("Видалити пресет"); self.btn_del_preset.clicked.connect(self.on_delete_preset)
        rowp.addWidget(self.btn_save_preset); rowp.addWidget(self.btn_del_preset)

        lyb.addWidget(QtWidgets.QLabel("Пресети:"))
        lyb.addWidget(self.combo_preset)
        lyb.addLayout(rowp)

        self.combo_outmode = QtWidgets.QComboBox(); self.combo_outmode.addItems(["Neon (darkpfp)","Line Art (біле на чорному)"])
        self.combo_outmode.currentIndexChanged.connect(self.on_outmode_change)
        lyb.addWidget(QtWidgets.QLabel("Стиль виводу:")); lyb.addWidget(self.combo_outmode)

        self.edit_tint = QtWidgets.QLineEdit("#E7C2FF")
        lyb.addWidget(QtWidgets.QLabel("Колір неону (hex, для Neon):")); lyb.addWidget(self.edit_tint)

        grp_line, self.sp_line = labeled_slider(host, "Товщина лінії", 0.5, 5.0, 0.1, 1.8, 1)
        grp_glow, self.sp_glow = labeled_slider(host, "Сяйво (радіус)", 0.0, 30.0, 0.5, 12.0, 1)
        grp_gg, self.sp_glow_gain = labeled_slider(host, "Сяйво (інтенсивність)", 0.5, 2.0, 0.05, 1.35, 2)
        grp_soft, self.sp_line_soft = labeled_slider(host, "М'якість ліній (Line Art)", 0.0, 5.0, 0.1, 0.6, 1)
        lyb.addWidget(grp_line); lyb.addWidget(grp_glow); lyb.addWidget(grp_gg); lyb.addWidget(grp_soft)

        left.addWidget(box_basic)

        # --- Контури ---
        box_edges = QtWidgets.QGroupBox("Контури")
        lye = QtWidgets.QVBoxLayout(box_edges)
        self.combo_edges = QtWidgets.QComboBox(); self.combo_edges.addItems(["Fusion","Canny","HED"])
        lye.addWidget(QtWidgets.QLabel("Детектор:")); lye.addWidget(self.combo_edges)

        grp_canny_sigma, self.sp_canny_sigma = labeled_slider(host, "Canny σ (чутливість)", 0.05, 1.0, 0.01, 0.28, 2)
        lye.addWidget(grp_canny_sigma)

        grp_edge_thresh, self.sp_edge_thresh = labeled_slider(host, "Fusion: поріг бінаризації", 0.0, 1.0, 0.01, 0.22, 2)
        grp_clahe_clip, self.sp_clahe_clip = labeled_slider(host, "Fusion: CLAHE clip", 0.1, 5.0, 0.1, 2.2, 1)
        grp_clahe_grid, self.sp_clahe_grid = labeled_slider(host, "Fusion: CLAHE grid", 2, 16, 1, 8, 0)
        grp_ms_levels, self.sp_ms_levels = labeled_slider(host, "Fusion: Multi-scale levels", 0, 2, 1, 2, 0)
        grp_chroma_w, self.sp_chroma_w = labeled_slider(host, "Fusion: вага хроми (V)", 0.0, 1.0, 0.05, 0.5, 2)
        grp_log_sigma, self.sp_log_sigma = labeled_slider(host, "Fusion: LoG σ", 0.0, 3.0, 0.1, 1.0, 1)
        grp_morph_d, self.sp_morph_d = labeled_slider(host, "Morph: розширення (dilate)", 0, 6, 1, 2, 0)
        grp_morph_e, self.sp_morph_e = labeled_slider(host, "Morph: звуження (erode)", 0, 6, 1, 0, 0)
        grp_median, self.sp_median_ks = labeled_slider(host, "Median cleanup (непарний px)", 0, 9, 1, 3, 0)
        for g in (grp_edge_thresh, grp_clahe_clip, grp_clahe_grid, grp_ms_levels, grp_chroma_w, grp_log_sigma, grp_morph_d, grp_morph_e, grp_median):
            lye.addWidget(g)

        # Опції якості
        self.chk_boost = QtWidgets.QCheckBox("Foreground boost (saliency)")
        self.chk_seg   = QtWidgets.QCheckBox("Segmentation mask (U2Net/GrabCut)")
        grp_mask_strength, self.sp_mask_strength = labeled_slider(host, "Mask strength (seg)", 0.0, 1.0, 0.05, 0.7, 2)
        self.chk_thin  = QtWidgets.QCheckBox("Thinning (скелетизація)")
        lye.addWidget(self.chk_boost); lye.addWidget(self.chk_seg); lye.addWidget(grp_mask_strength); lye.addWidget(self.chk_thin)

        left.addWidget(box_edges)

        # --- Preview control ---
        rowc = QtWidgets.QHBoxLayout()
        self.chk_auto = QtWidgets.QCheckBox("Авто-предпросмотр"); self.chk_auto.setChecked(True)
        self.btn_preview = QtWidgets.QPushButton("Оновити предпросмотр"); self.btn_preview.clicked.connect(self.update_preview)
        rowc.addWidget(self.chk_auto); rowc.addWidget(self.btn_preview)
        left.addLayout(rowc)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(130)
        left.addWidget(QtWidgets.QLabel("Логи:")); left.addWidget(self.log)
        left.addStretch(1)

        # Права панель (preview)
        right = QtWidgets.QVBoxLayout(); h.addLayout(right, 1)
        self.lbl_in = QtWidgets.QLabel("Вхідне зображення"); self.lbl_in.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_in.setMinimumSize(520,400); self.lbl_in.setFrameShape(QtWidgets.QFrame.Box); right.addWidget(self.lbl_in, 1)
        self.lbl_out = QtWidgets.QLabel("Предпросмотр"); self.lbl_out.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_out.setMinimumSize(520,400); self.lbl_out.setFrameShape(QtWidgets.QFrame.Box); right.addWidget(self.lbl_out, 1)

        self.src_path = None

        # Авто-предпросмотр підписки
        for w in [self.combo_outmode, self.combo_upscale, self.combo_preset, self.combo_edges,
                  self.edit_tint, self.spin_size, self.chk_square, self.chk_stretch,
                  self.sp_canny_sigma, self.sp_edge_thresh, self.sp_clahe_clip, self.sp_clahe_grid,
                  self.sp_ms_levels, self.sp_chroma_w, self.sp_log_sigma, self.sp_morph_d, self.sp_morph_e, self.sp_median_ks,
                  self.sp_line, self.sp_glow, self.sp_glow_gain, self.sp_line_soft,
                  self.chk_boost, self.chk_thin, self.chk_seg, self.sp_mask_strength]:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.on_any_change)
            elif hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self.on_any_change)
            elif hasattr(w, "textChanged"):
                w.textChanged.connect(self.on_any_change)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self.on_any_change)

        # Відновити останній пресет
        last_preset = self.cfg.get("ui","last_preset", fallback=None)
        if last_preset:
            idx = self.combo_preset.findText(last_preset)
            if idx >= 0:
                self.combo_preset.setCurrentIndex(idx)

        self.apply_current_preset()
        self.on_outmode_change()

    # ---- Presets helpers ----
    def refresh_preset_combo(self):
        self.combo_preset.clear()
        for k in BUILTIN_PRESETS.keys():
            self.combo_preset.addItem(k)
        for k in sorted(self.user_presets.keys()):
            self.combo_preset.addItem(k)

    def is_builtin_preset(self, name):
        return name in BUILTIN_PRESETS

    def current_preset_name(self):
        return self.combo_preset.currentText()

    def save_ui_state(self):
        if not self.cfg.has_section("ui"):
            self.cfg.add_section("ui")
        self.cfg.set("ui","last_preset", self.current_preset_name())
        save_ini(self.cfg)

    # ---- logic ----
    def on_outmode_change(self):
        is_lineart = (self.combo_outmode.currentIndex() == 1)
        for w in [self.edit_tint, self.sp_glow, self.sp_glow_gain]:
            w.setEnabled(not is_lineart)
        self.sp_line_soft.setEnabled(is_lineart)
        if self.chk_auto.isChecked() and self.src_path:
            self.update_preview()

    def log_msg(self, s): self.log.appendPlainText(s)

    def build_param_dict_from_ui(self):
        edge_params = {
            "canny_sigma": float(self.sp_canny_sigma.value()),
            "edge_thresh": float(self.sp_edge_thresh.value()),
            "clahe_clip": float(self.sp_clahe_clip.value()),
            "clahe_grid": int(self.sp_clahe_grid.value()),
            "ms_levels": int(self.sp_ms_levels.value()),
            "chroma_w": float(self.sp_chroma_w.value()),
            "log_sigma": float(self.sp_log_sigma.value()),
            "median_ksize": int(self.sp_median_ks.value()),
            "morph_dilate": int(self.sp_morph_d.value()),
            "morph_erode": int(self.sp_morph_e.value()),
        }
        out_mode = "lineart" if self.combo_outmode.currentIndex()==1 else "neon"
        return {
            "out_mode": out_mode,
            "tint": self.edit_tint.text().strip(),
            "size": int(self.spin_size.value()),
            "square": bool(self.chk_square.isChecked()),
            "stretch": bool(self.chk_stretch.isChecked()),
            "line": float(self.sp_line.value()),
            "glow": float(self.sp_glow.value()),
            "glow_gain": float(self.sp_glow_gain.value()),
            "edges_mode": self.combo_edges.currentText(),
            "edge_params": edge_params,
            "line_soft": float(self.sp_line_soft.value()),
            "edge_boost": bool(self.chk_boost.isChecked()),
            "edge_thin": bool(self.chk_thin.isChecked()),
            "edge_seg": bool(self.chk_seg.isChecked()),
            "mask_strength": float(self.sp_mask_strength.value()),
        }

    def apply_current_preset(self):
        name = self.combo_preset.currentText()
        p = None
        if name in BUILTIN_PRESETS:
            p = BUILTIN_PRESETS[name]
        elif name in self.user_presets:
            p = self.user_presets[name]
        if not p:
            return

        if p.get("out_mode","neon") == "lineart":
            self.combo_outmode.setCurrentIndex(1)
        else:
            self.combo_outmode.setCurrentIndex(0)

        self.spin_size.setValue(p.get("size", 1024))
        self.chk_square.setChecked(bool(p.get("square", True)))
        self.chk_stretch.setChecked(bool(p.get("stretch", False)))

        self.sp_line.setValue(p.get("line", 1.8))
        self.sp_glow.setValue(p.get("glow", 12.0))
        self.sp_glow_gain.setValue(p.get("glow_gain", 1.35))
        self.edit_tint.setText(p.get("tint", "#E7C2FF"))
        self.sp_line_soft.setValue(p.get("softness", 0.6))

        self.combo_edges.setCurrentText(p.get("edges_mode","Fusion"))
        ep = p.get("edge_params", {})
        self.sp_canny_sigma.setValue(ep.get("canny_sigma",0.28))
        self.sp_edge_thresh.setValue(ep.get("edge_thresh",0.22))
        self.sp_clahe_clip.setValue(ep.get("clahe_clip",2.2))
        self.sp_clahe_grid.setValue(ep.get("clahe_grid",8))
        self.sp_ms_levels.setValue(ep.get("ms_levels",2))
        self.sp_chroma_w.setValue(ep.get("chroma_w",0.5))
        self.sp_log_sigma.setValue(ep.get("log_sigma",1.0))
        self.sp_morph_d.setValue(ep.get("morph_dilate",2))
        self.sp_morph_e.setValue(ep.get("morph_erode",0))
        self.sp_median_ks.setValue(ep.get("median_ksize",3))

        self.chk_boost.setChecked(bool(p.get("edge_boost", False)))
        self.chk_thin.setChecked(bool(p.get("edge_thin", False)))
        self.chk_seg.setChecked(bool(p.get("edge_seg", False)))
        self.sp_mask_strength.setValue(float(p.get("mask_strength", 0.7)))

        self.save_ui_state()
        if self.chk_auto.isChecked():
            self.update_preview()

    def open_image(self):
        start_dir = self.cfg.get("ui","last_open_dir", fallback=str(SCRIPT_DIR))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Вибрати зображення", start_dir, "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path: return
        self.src_path = path
        self.show_image(path, self.lbl_in)
        if not self.cfg.has_section("ui"): self.cfg.add_section("ui")
        self.cfg.set("ui","last_open_dir", str(Path(path).parent))
        save_ini(self.cfg)
        if self.chk_auto.isChecked(): self.update_preview()

    def process_and_save(self):
        if not self.src_path:
            QtWidgets.QMessageBox.warning(self, "Немає вхідного", "Спочатку відкрий зображення.")
            return
        start_dir = self.cfg.get("ui","last_save_dir", fallback=str(SCRIPT_DIR))
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Зберегти як…", start_dir, "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not save_path: return
        if not self.cfg.has_section("ui"): self.cfg.add_section("ui")
        self.cfg.set("ui","last_save_dir", str(Path(save_path).parent))
        save_ini(self.cfg)

        upsel = self.combo_upscale.currentText()
        upscale = 1
        if "2x" in upsel: upscale=2
        if "4x" in upsel: upscale=4
        params = self.build_param_dict_from_ui()
        self.run_worker(upscale, params, save_path=save_path)

    def on_save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Новий пресет", "Назва пресету:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if self.is_builtin_preset(name):
            QtWidgets.QMessageBox.warning(self, "Неможливо", "Це ім'я зайняте вбудованим пресетом. Оберіть іншу назву.")
            return
        p = self.build_param_dict_from_ui()
        self.user_presets[name] = p
        save_user_preset(self.cfg, name, p)
        self.save_ui_state()
        save_ini(self.cfg)
        self.refresh_preset_combo()
        idx = self.combo_preset.findText(name)
        if idx >= 0: self.combo_preset.setCurrentIndex(idx)
        self.log_msg(f"Збережено пресет: {name}")

    def on_delete_preset(self):
        name = self.current_preset_name()
        if self.is_builtin_preset(name):
            QtWidgets.QMessageBox.information(self, "Не видаляється", "Вбудовані пресети не можна видалити.")
            return
        if name not in self.user_presets:
            QtWidgets.QMessageBox.information(self, "Немає", "Цей пресет не є користувацьким.")
            return
        yn = QtWidgets.QMessageBox.question(self, "Підтвердження", f"Видалити пресет «{name}»?")
        if yn != QtWidgets.QMessageBox.Yes:
            return
        delete_user_preset(self.cfg, name)
        self.user_presets.pop(name, None)
        save_ini(self.cfg)
        self.refresh_preset_combo()
        self.combo_preset.setCurrentIndex(0)
        self.apply_current_preset()
        self.log_msg(f"Видалено пресет: {name}")

    # --- Пакетна обробка ---
    def on_batch(self):
        start_in = self.cfg.get("ui","last_batch_in_dir", fallback=self.cfg.get("ui","last_open_dir", fallback=str(SCRIPT_DIR)))
        in_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Вибрати вхідну папку", start_in)
        if not in_dir: return
        in_dir = Path(in_dir)
        if not self.cfg.has_section("ui"): self.cfg.add_section("ui")
        self.cfg.set("ui","last_batch_in_dir", str(in_dir)); save_ini(self.cfg)

        exts = {".png",".jpg",".jpeg",".webp",".bmp"}
        files = [str(p) for p in sorted(in_dir.glob("*")) if p.suffix.lower() in exts]
        if not files:
            QtWidgets.QMessageBox.information(self, "Немає файлів", "У вибраній папці немає підтримуваних зображень.")
            return

        start_out = self.cfg.get("ui","last_batch_out_dir", fallback=self.cfg.get("ui","last_save_dir", fallback=str(SCRIPT_DIR)))
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Куди зберегти результати", start_out)
        if not out_dir: return
        out_dir = Path(out_dir)
        self.cfg.set("ui","last_batch_out_dir", str(out_dir)); save_ini(self.cfg)

        upsel = self.combo_upscale.currentText()
        upscale = 1
        if "2x" in upsel: upscale=2
        if "4x" in upsel: upscale=4
        params = self.build_param_dict_from_ui()

        progress = QtWidgets.QProgressDialog("Обробка зображень…", "Скасувати", 0, len(files), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.batch_thread = QtCore.QThread(self)
        self.batch_worker = BatchWorker(files, out_dir, upscale, params)
        self.batch_worker.moveToThread(self.batch_thread)

        def on_prog(i, total, msg):
            progress.setMaximum(total)
            progress.setValue(i)
            progress.setLabelText(msg)
            if progress.wasCanceled():
                self.batch_worker.cancel()

        def on_done(saved, skipped):
            progress.reset()
            QtWidgets.QMessageBox.information(self, "Готово", f"Збережено: {saved}\nПропущено: {skipped}")
            self.log_msg(f"Пакет: готово. saved={saved}, skipped={skipped}")
            self.batch_thread.quit()

        def on_fail(err):
            progress.reset()
            QtWidgets.QMessageBox.critical(self, "Помилка", err)
            self.log_msg("Batch помилка: " + err)
            self.batch_thread.quit()

        self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(on_prog)
        self.batch_worker.finished.connect(on_done)
        self.batch_worker.failed.connect(on_fail)
        self.batch_thread.finished.connect(self.batch_worker.deleteLater)
        self.batch_thread.start()

    def show_image(self, path_or_array, label):
        if isinstance(path_or_array, str):
            img = read_image_any(path_or_array)
        else:
            img = path_or_array
        if img is None: return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.src_path:
            self.show_image(self.src_path, self.lbl_in)

    def on_any_change(self, *args):
        self.save_ui_state()
        if self.chk_auto.isChecked() and self.src_path:
            self.update_preview()

    def update_preview(self):
        if not self.src_path: return
        upsel = self.combo_upscale.currentText()
        upscale = 1
        if "2x" in upsel: upscale=2
        if "4x" in upsel: upscale=4
        params = self.build_param_dict_from_ui()
        self.run_worker(upscale, params, save_path=None)

    def run_worker(self, upscale, params, save_path=None):
        self.thread = QtCore.QThread(self)
        self.worker = RenderWorker(self.src_path, upscale, params, save_path=save_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.preview_ready.connect(self.on_preview_ready)
        self.worker.saved.connect(self.on_saved)
        self.worker.failed.connect(self.on_failed)
        self.worker.preview_ready.connect(self.thread.quit)
        self.worker.saved.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def on_preview_ready(self, img):
        self.show_image(img, self.lbl_out)
        self.log_msg("Предпросмотр оновлено.")

    @QtCore.pyqtSlot(np.ndarray, str)
    def on_saved(self, img, path):
        self.show_image(img, self.lbl_out)
        self.log_msg(f"Збережено: {path}")

    @QtCore.pyqtSlot(str)
    def on_failed(self, err):
        QtWidgets.QMessageBox.critical(self, "Помилка", err)
        self.log_msg("Помилка: " + err)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
