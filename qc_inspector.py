#!/usr/bin/env python3
"""
Steel surface QC inspector - Arm Bharat AI-SoC Challenge 2026, PS5.

Pipeline (all on-device, offline):
  image -> pre-process (Pillow/numpy; OpenCV used if installed) -> YOLO11n (ONNX Runtime, Arm CPU)
        -> post-process (NMS) -> severity rules -> JSON findings
        -> Llama 3.2 1B Instruct (ExecuTorch llama_main, Arm CPU) on FAIL
        -> operator output + CSV log

Two runtimes, kept separate on purpose (slides: "Keep the two runtimes
clearly separated in the report"):
  * Vision  : ONNX Runtime  (yolo11n_neu.onnx)
  * Advisory: ExecuTorch    (llama_main + .pte)

Usage examples (run inside ~/qc on the phone):
  python qc_inspector.py inspect --image test/img.jpg
  python qc_inspector.py run --images test_images/ --save-annotated out/
  python qc_inspector.py bench --images test_images/ --labels test_labels/
  python qc_inspector.py run --images test_images/ --no-llama     # vision only
"""

import argparse
import csv
import glob
import json
import os
import re
import resource
import subprocess
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:  # Termux has no OpenCV package: use Pillow + numpy instead
    cv2 = None
from PIL import Image, ImageDraw

import onnxruntime as ort

# ---------------------------------------------------------------------------
# Configuration (edit here, document these values in the report)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model": os.path.join(HERE, "yolo11n_neu.onnx"),
    "labels": os.path.join(HERE, "labels.txt"),
    "conf_threshold": 0.25,      # min detection confidence to count as a defect
    "iou_threshold": 0.45,       # NMS overlap
    "contain_threshold": 0.7,    # same-class box mostly inside another -> drop the weaker one
    "max_defects_to_llama": 3,   # keep the prompt short
    "vision_threads": 4,

    # Llama / ExecuTorch
    "llama_dir": os.path.expanduser("~/llama"),
    "llama_bin": "llama_main",
    "llama_model": "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte",
    "llama_tokenizer": "tokenizer.model",
    "llama_threads": 4,
    "llama_seq_len": 1024,       # TOTAL tokens = prompt (~300) + answer. 400 cut answers off
    "llama_raw_file": os.path.join(HERE, "llama_last_raw.txt"),  # raw stdout, for debugging
    "llama_temperature": 0,      # deterministic, grounded answers
    "llama_timeout_s": 120,

    "log_csv": os.path.join(HERE, "inspection_log.csv"),
}

# Severity rules. Confidence is NOT severity (slides FAQ 3 Q3).
# Severity comes from defect type, size (box area as % of image) and location.
# These thresholds are a documented starting point; tune them on validation data.
SEVERITY_RULES = {
    # defect classes that can propagate / weaken the part
    "critical_classes": {"crazing", "pitted_surface", "inclusion"},
    "high_area_pct": 5.0,      # critical class bigger than this -> High
    "medium_area_pct": 2.0,    # any class bigger than this -> Medium
    "edge_margin_pct": 10.0,   # box within 10% of the image border = near edge
}
SEV_ORDER = {"None": 0, "Low": 1, "Medium": 2, "High": 3}


def severity_for(defect, area_pct, near_edge):
    r = SEVERITY_RULES
    critical = defect in r["critical_classes"]
    if critical and area_pct > r["high_area_pct"]:
        return "High"
    if area_pct > r["medium_area_pct"] or (critical and near_edge):
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Vision: pre-process, inference, post-process
# ---------------------------------------------------------------------------
def load_labels(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_image(path):
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, 3 channels
        if img is None:
            raise ValueError(f"cannot read {path}")
        return img
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def letterbox(img, size):
    """Resize keeping aspect ratio and pad to size x size (Ultralytics style)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if cv2 is not None:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.array(Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
                           .resize((nw, nh), Image.BILINEAR))[:, :, ::-1]
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def nms(boxes, scores, iou_thr):
    """Plain numpy NMS. boxes: (N,4) x1,y1,x2,y2."""
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def suppress_contained(boxes, confs, cls_ids, keep, thr):
    """Drop a same-class box when most of it (intersection / smaller area > thr)
    lies inside a higher-confidence box. Catches the nested duplicates that
    plain IoU-NMS misses for long thin defects such as scratches."""
    keep = sorted(keep, key=lambda i: -confs[i])
    out = []
    for i in keep:
        bi = boxes[i]
        ai = max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1e-9)
        dup = False
        for j in out:
            if cls_ids[j] != cls_ids[i]:
                continue
            bj = boxes[j]
            aj = max((bj[2] - bj[0]) * (bj[3] - bj[1]), 1e-9)
            iw = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
            ih = max(0.0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            if iw * ih / min(ai, aj) > thr:
                dup = True
                break
        if not dup:
            out.append(i)
    return out


class Detector:
    def __init__(self, model_path, labels, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])  # model input is square, e.g. 320 or 640
        self.labels = labels

    def __call__(self, img, conf_thr, iou_thr):
        t0 = time.perf_counter()
        lb, r, left, top = letterbox(img, self.size)
        x = lb[:, :, ::-1].transpose(2, 0, 1)            # BGR->RGB, HWC->CHW
        x = np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: x})[0]  # (1, 4+nc, N)
        t2 = time.perf_counter()

        pred = out[0].T                                   # (N, 4+nc)
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(1)
        confs = cls_scores[np.arange(len(cls_ids)), cls_ids]
        m = confs >= conf_thr
        pred, cls_ids, confs = pred[m], cls_ids[m], confs[m]
        dets = []
        if len(pred):
            cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            # undo letterbox -> original image pixels
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
            H, W = img.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
            # class-aware NMS (offset boxes per class)
            offs = cls_ids[:, None] * 4096.0
            keep = nms(boxes + offs, confs, iou_thr)
            keep = suppress_contained(boxes, confs, cls_ids, keep,
                                      CONFIG["contain_threshold"])
            for i in keep:
                dets.append((self.labels[int(cls_ids[i])], float(confs[i]),
                             [int(round(v)) for v in boxes[i]]))
        t3 = time.perf_counter()
        timing = {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3,
                  "post_ms": (t3 - t2) * 1e3}
        return dets, timing


# ---------------------------------------------------------------------------
# Rules -> JSON findings (the agreed hand-off format)
# ---------------------------------------------------------------------------
def build_findings(image_name, img_shape, dets):
    H, W = img_shape[:2]
    margin = SEVERITY_RULES["edge_margin_pct"] / 100.0
    defects = []
    for name, conf, (x1, y1, x2, y2) in dets:
        area_pct = 100.0 * (x2 - x1) * (y2 - y1) / float(W * H)
        near_edge = (x1 < W * margin or y1 < H * margin or
                     x2 > W * (1 - margin) or y2 > H * (1 - margin))
        defects.append({
            "defect": name,
            "severity": severity_for(name, area_pct, near_edge),
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2],
            "area_pct": round(area_pct, 2),
            "location": "edge" if near_edge else "centre",
        })
    defects.sort(key=lambda d: (SEV_ORDER[d["severity"]], d["area_pct"]), reverse=True)
    worst = defects[0]["severity"] if defects else "None"
    return {
        "image": image_name,
        "verdict": "FAIL" if defects else "PASS",   # rule-based pass/fail (FAQ 3 Q4)
        "max_severity": worst,
        "defect_count": len(defects),
        "defects": defects,
    }


# ---------------------------------------------------------------------------
# Advisory: Llama 3.2 1B via ExecuTorch llama_main
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a quality-control assistant for a hot-rolled steel strip line. "
    "You receive defect findings as JSON from a vision model. Do not invent defects. "
    "Answer in exactly 3 lines and nothing else:\n"
    "CAUSE: <likely process cause, one sentence, max 15 words>\n"
    "ACTION: <one full sentence: what the operator should do now and what to check, max 20 words>\n"
    "LOG: <one line: verdict - defect, severity, area, location; decision>"
)

EXAMPLE_FINDINGS = {
    "verdict": "FAIL", "max_severity": "Medium",
    "defects": [{"defect": "rolled-in_scale",
                 "meaning": "oxide scale pressed into the surface during rolling",
                 "count": 2, "severity": "Medium", "max_confidence": 0.81,
                 "total_area_pct": 3.4, "locations": "centre"}],
}
EXAMPLE_ANSWER = (
    "CAUSE: Mill scale pressed into the surface, likely poor descaling before rolling.\n"
    "ACTION: Divert strip for grinding check; inspect descaler nozzles and water pressure.\n"
    "LOG: FAIL - rolled-in_scale x2, Medium, 3.4% area, centre; held for rework."
)

DEFECT_HINTS = {  # short domain context passed with each defect (FAQ 3 Q5)
    "crazing": "network of fine surface cracks",
    "inclusion": "non-metallic particles embedded in the steel",
    "patches": "irregular discoloured patches",
    "pitted_surface": "small pits or cavities in the surface",
    "rolled-in_scale": "oxide scale pressed into the surface during rolling",
    "scratches": "linear scratches from mechanical contact",
}


def slim_findings(findings):
    """Group same-class defects into one entry (count, total area, worst severity)
    so the prompt stays short and the model is not repeating itself."""
    groups = {}
    for d in findings["defects"]:
        g = groups.setdefault(d["defect"], {
            "defect": d["defect"], "meaning": DEFECT_HINTS.get(d["defect"], ""),
            "count": 0, "severity": "Low", "max_confidence": 0.0,
            "total_area_pct": 0.0, "locations": set()})
        g["count"] += 1
        if SEV_ORDER[d["severity"]] > SEV_ORDER[g["severity"]]:
            g["severity"] = d["severity"]
        g["max_confidence"] = max(g["max_confidence"], d["confidence"])
        g["total_area_pct"] += d["area_pct"]
        g["locations"].add(d["location"])
    items = sorted(groups.values(), key=lambda g: (SEV_ORDER[g["severity"]],
                                                   g["total_area_pct"]), reverse=True)
    out = {"verdict": findings["verdict"], "max_severity": findings["max_severity"],
           "defects": []}
    for g in items[:CONFIG["max_defects_to_llama"]]:
        g["total_area_pct"] = round(g["total_area_pct"], 1)
        g["max_confidence"] = round(g["max_confidence"], 2)
        g["locations"] = "/".join(sorted(g["locations"]))
        out["defects"].append(g)
    return out


def build_prompt(findings):
    def turn(role, text):
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
    return ("<|begin_of_text|>"
            + turn("system", SYSTEM_PROMPT)
            + turn("user", json.dumps(EXAMPLE_FINDINGS))
            + turn("assistant", EXAMPLE_ANSWER)
            + turn("user", json.dumps(slim_findings(findings)))
            + "<|start_header_id|>assistant<|end_header_id|>\n\n")


def parse_answer(raw):
    """Pull CAUSE/ACTION/LOG out of llama_main stdout.
    Tolerates echoed prompt, markdown (**CAUSE:**), lower case, '-' or ':'
    separators, values on the next line, and all three keys on one line."""
    text = raw
    marker = "assistant<|end_header_id|>"
    if marker in text:                       # runner may echo the prompt
        text = text.split(marker)[-1]
    for stop in ("<|eot_id|>", "<|end_of_text|>", "PyTorchObserver"):
        text = text.split(stop)[0]
    text = text.replace("**", "").replace("__", "")
    keys = r"(CAUSE|ACTION|LOG)"
    parts = re.split(rf"(?i)(?:^|\s|[>#*\-])\b{keys}\b\s*[:\-]\s*", text)
    out = {"cause": "", "action": "", "log": ""}
    # parts = [before, KEY1, value1, KEY2, value2, ...]
    for k, v in zip(parts[1::2], parts[2::2]):
        v = " ".join(v.split()).strip(" -*")
        if v and not out[k.lower()]:
            out[k.lower()] = v
    return out


def rule_log_line(findings):
    """Deterministic log line built from the JSON findings."""
    g = slim_findings(findings)["defects"]
    parts = [f"{x['defect']} x{x['count']} {x['severity']} {x['total_area_pct']}% {x['locations']}"
             for x in g]
    decision = "held for inspection" if findings["max_severity"] in ("High", "Medium") \
        else "passed with note"
    return f"{findings['verdict']} - " + "; ".join(parts) + f"; {decision}"


def rule_action(findings):
    sev = findings["max_severity"]
    if sev == "High":
        return "Stop and quarantine this part, then check the upstream process for the defect cause."
    if sev == "Medium":
        return "Hold the part for manual inspection and check the line for the likely cause."
    return "Pass with a note and watch the next parts for the same defect."


def rule_fallback(findings):
    return {"cause": "(advisor unavailable)", "action": rule_action(findings),
            "log": rule_log_line(findings)}


def fill_gaps(ans, findings):
    """Keep Llama's text, but never leave a field empty or a cut-off stub."""
    src = {}
    if len(ans["action"].split()) < 4:
        ans["action"] = rule_action(findings); src["action"] = "rule"
    if len(ans["log"].split()) < 4:
        ans["log"] = rule_log_line(findings); src["log"] = "rule"
    if not ans["cause"]:
        ans["cause"] = "(not given)"; src["cause"] = "rule"
    ans["filled_by_rules"] = ",".join(src) if src else ""
    return ans


def run_llama(findings):
    c = CONFIG
    prompt = build_prompt(findings)
    cmd = [os.path.join(c["llama_dir"], c["llama_bin"]),
           "--model_path", c["llama_model"],
           "--tokenizer_path", c["llama_tokenizer"],
           "--prompt", prompt,
           "--cpu_threads", str(c["llama_threads"]),
           "--seq_len", str(c["llama_seq_len"]),
           "--temperature", str(c["llama_temperature"])]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "." + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=c["llama_dir"], env=env, capture_output=True,
                           text=True, timeout=c["llama_timeout_s"])
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, (time.perf_counter() - t0) * 1e3, str(e)
    ms = (time.perf_counter() - t0) * 1e3
    if p.returncode != 0:
        return None, ms, (p.stderr or p.stdout)[-800:]
    try:
        with open(c["llama_raw_file"], "w") as f:
            f.write("=== PROMPT ===\n" + prompt + "\n=== STDOUT ===\n" + p.stdout
                    + "\n=== STDERR ===\n" + p.stderr)
    except OSError:
        pass
    ans = parse_answer(p.stdout)
    if not any(ans.values()):
        return None, ms, "unparseable output: " + p.stdout[-400:]
    return fill_gaps(ans, findings), ms, p.stdout + "\n" + p.stderr


# ---------------------------------------------------------------------------
# Output: annotation, CSV log, operator display
# ---------------------------------------------------------------------------
SEV_COLOURS = {"High": (0, 0, 255), "Medium": (0, 165, 255), "Low": (0, 200, 0)}


def save_annotated(img, findings, path):
    if cv2 is None:
        im = Image.fromarray(img[:, :, ::-1].copy())
        dr = ImageDraw.Draw(im)
        for d in findings["defects"]:
            dr.rectangle(d["box"], outline=SEV_COLOURS[d["severity"]][::-1], width=2)
            dr.text((d["box"][0], max(0, d["box"][1] - 10)),
                    f"{d['defect']} {d['severity']}", fill=(255, 255, 0))
        im.save(path)
        return
    out = img.copy()
    for d in findings["defects"]:
        x1, y1, x2, y2 = d["box"]
        col = SEV_COLOURS[d["severity"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d['defect']} {d['severity']} {d['confidence']:.2f}",
                    (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.putText(out, findings["verdict"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255) if findings["verdict"] == "FAIL" else (0, 200, 0), 1)
    cv2.imwrite(path, out)


LOG_FIELDS = ["timestamp", "image", "verdict", "max_severity", "defect_count",
              "defects_json", "pre_ms", "infer_ms", "post_ms", "llama_ms",
              "cause", "action", "log", "filled_by_rules"]


def append_log(row):
    path = CONFIG["log_csv"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def print_operator(findings, advice, timing, llama_ms):
    bar = "=" * 60
    print(bar)
    print(f"{findings['image']}:  {findings['verdict']}   (max severity: {findings['max_severity']})")
    for d in findings["defects"]:
        print(f"  - {d['defect']:<16} {d['severity']:<6} conf {d['confidence']:.2f}  "
              f"area {d['area_pct']:>5.1f}%  {d['location']}")
    if advice:
        print(f"  CAUSE : {advice['cause']}")
        print(f"  ACTION: {advice['action']}")
        print(f"  LOG   : {advice['log']}")
        if advice.get("filled_by_rules"):
            print(f"  (filled by rules: {advice['filled_by_rules']})")
    t = timing
    extra = f"  llama {llama_ms:.0f} ms" if llama_ms else ""
    print(f"  [vision pre {t['pre_ms']:.1f} / infer {t['infer_ms']:.1f} / post {t['post_ms']:.1f} ms{extra}]")


# ---------------------------------------------------------------------------
# Inspection loop
# ---------------------------------------------------------------------------
def inspect_one(det, path, use_llama, annotate_dir=None, verbose=True):
    img = read_image(path)
    dets, timing = det(img, CONFIG["conf_threshold"], CONFIG["iou_threshold"])
    findings = build_findings(os.path.basename(path), img.shape, dets)
    advice, llama_ms = None, 0.0
    if findings["verdict"] == "FAIL":
        if use_llama:
            advice, llama_ms, info = run_llama(findings)
            if advice is None:
                print(f"  [llama failed: {info.strip()[:300]}]", file=sys.stderr)
                advice = rule_fallback(findings)
        else:
            advice = rule_fallback(findings)
    findings["advice"] = advice
    if annotate_dir:
        os.makedirs(annotate_dir, exist_ok=True)
        save_annotated(img, findings, os.path.join(annotate_dir, os.path.basename(path)))
    append_log({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image": findings["image"], "verdict": findings["verdict"],
        "max_severity": findings["max_severity"], "defect_count": findings["defect_count"],
        "defects_json": json.dumps(findings["defects"]),
        "pre_ms": f"{timing['pre_ms']:.2f}", "infer_ms": f"{timing['infer_ms']:.2f}",
        "post_ms": f"{timing['post_ms']:.2f}", "llama_ms": f"{llama_ms:.0f}",
        "cause": (advice or {}).get("cause", ""), "action": (advice or {}).get("action", ""),
        "log": (advice or {}).get("log", ""),
        "filled_by_rules": (advice or {}).get("filled_by_rules", ""),
    })
    if verbose:
        print_operator(findings, advice, timing, llama_ms)
    return findings, timing, llama_ms, dets, img.shape


def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


# ---------------------------------------------------------------------------
# Benchmark: precision / recall / F1 (IoU 0.5) + latency + memory
# ---------------------------------------------------------------------------
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path, shape, labels):
    """YOLO txt: class cx cy w h (normalised)."""
    H, W = shape[:2]
    gts = []
    if os.path.exists(label_path):
        for ln in open(label_path):
            p = ln.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            gts.append((labels[c], [(cx - w / 2) * W, (cy - h / 2) * H,
                                    (cx + w / 2) * W, (cy + h / 2) * H]))
    return gts


def match(dets, gts, thr=0.5):
    """Greedy per-class matching. Returns tp, fp, fn and image-level flags."""
    used = set()
    tp = 0
    for name, conf, box in sorted(dets, key=lambda d: -d[1]):
        best, bj = 0.0, -1
        for j, (gname, gbox) in enumerate(gts):
            if j in used or gname != name:
                continue
            v = iou(box, gbox)
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj)
            tp += 1
    return tp, len(dets) - tp, len(gts) - tp


def peak_rss_mb():
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    return me, kids  # Linux/Android report KB


def cmd_bench(det, args):
    files = list_images(args.images)
    if args.limit and args.limit < len(files):
        # evenly spaced sample so every defect class is represented
        # (files are sorted by name, i.e. grouped by class)
        idx = np.linspace(0, len(files) - 1, args.limit).round().astype(int)
        files = [files[i] for i in sorted(set(idx))]
    if not files:
        sys.exit(f"no images in {args.images}")
    # warm-up so the first-run allocation does not skew timings
    for _ in range(3):
        det(read_image(files[0]), CONFIG["conf_threshold"], CONFIG["iou_threshold"])

    TP = FP = FN = 0
    img_tp = img_fp = img_fn = img_tn = 0
    pre, inf, post, llama, e2e = [], [], [], [], []
    for f in files:
        t0 = time.perf_counter()
        findings, timing, llama_ms, dets, shape = inspect_one(
            det, f, use_llama=args.llama, annotate_dir=args.save_annotated,
            verbose=args.verbose)
        e2e.append((time.perf_counter() - t0) * 1e3)
        pre.append(timing["pre_ms"]); inf.append(timing["infer_ms"]); post.append(timing["post_ms"])
        if llama_ms:
            llama.append(llama_ms)
        if args.labels:
            stem = os.path.splitext(os.path.basename(f))[0]
            gts = load_gt(os.path.join(args.labels, stem + ".txt"), shape, det.labels)
            tp, fp, fn = match(dets, gts)
            TP += tp; FP += fp; FN += fn
            has_gt, has_det = bool(gts), bool(dets)
            img_tp += has_gt and has_det
            img_fn += has_gt and not has_det
            img_fp += (not has_gt) and has_det
            img_tn += (not has_gt) and (not has_det)

    def stats(xs):
        return f"mean {np.mean(xs):7.1f}  p50 {np.median(xs):7.1f}  p95 {np.percentile(xs, 95):7.1f}" if xs else "n/a"

    vision = [a + b + c for a, b, c in zip(pre, inf, post)]
    me, kids = peak_rss_mb()
    print("\n" + "=" * 60)
    print(f"BENCHMARK  images: {len(files)}   model: {os.path.basename(CONFIG['model'])} "
          f"({det.size}x{det.size})   threads: {CONFIG['vision_threads']}")
    print(f"pre-process ms   {stats(pre)}")
    print(f"inference ms     {stats(inf)}")
    print(f"post-process ms  {stats(post)}")
    print(f"vision total ms  {stats(vision)}   -> vision FPS {1000.0 / np.mean(vision):.1f}")
    print(f"llama ms (FAIL)  {stats(llama)}   calls: {len(llama)}")
    print(f"end-to-end ms    {stats(e2e)}")
    print(f"peak RSS MB      python+vision {me:.0f}   llama (child) {kids:.0f}")
    if args.labels:
        p = TP / (TP + FP) if TP + FP else 0.0
        r = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"box level  (IoU>=0.5, class match): P {p:.3f}  R {r:.3f}  F1 {f1:.3f}  "
              f"(TP {TP} FP {FP} FN {FN})")
        print(f"image level (defect present?): TP {img_tp} FN {img_fn} FP {img_fp} TN {img_tn}")
    print(f"log: {CONFIG['log_csv']}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Steel surface QC inspector (PS5)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("inspect", help="inspect one image")
    a.add_argument("--image", required=True)

    b = sub.add_parser("run", help="inspect every image in a folder")
    b.add_argument("--images", required=True)

    c = sub.add_parser("bench", help="benchmark: latency, FPS, memory, P/R/F1")
    c.add_argument("--images", required=True)
    c.add_argument("--labels", help="folder of YOLO .txt ground truth (same names)")
    c.add_argument("--llama", action="store_true", help="include Llama calls (slower)")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--verbose", action="store_true")

    for p in (a, b, c):
        p.add_argument("--model", default=CONFIG["model"])
        p.add_argument("--labels-file", default=CONFIG["labels"])
        p.add_argument("--conf", type=float, default=CONFIG["conf_threshold"])
        p.add_argument("--save-annotated", default=None, help="folder for boxed images")
    for p in (a, b):
        p.add_argument("--no-llama", action="store_true", help="vision + rules only")

    args = ap.parse_args()
    CONFIG["conf_threshold"] = args.conf
    CONFIG["model"] = args.model
    labels = load_labels(args.labels_file)
    det = Detector(args.model, labels, CONFIG["vision_threads"])

    if args.cmd == "inspect":
        f, _, _, _, _ = inspect_one(det, args.image, not args.no_llama, args.save_annotated)
        print(json.dumps(f, indent=2))
    elif args.cmd == "run":
        files = list_images(args.images)
        for f in files:
            inspect_one(det, f, not args.no_llama, args.save_annotated)
        print(f"\n{len(files)} images inspected. Log: {CONFIG['log_csv']}")
    else:
        cmd_bench(det, args)


if __name__ == "__main__":
    main()
