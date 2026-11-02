#!/usr/bin/env python3
"""
Noise event recorder + classifier + review web UI (Raspberry Pi 4, USB mic).

1. Listens continuously (~1 s blocks, A-weighted level). A block that is louder than
   the adaptive background by --delta dB starts an event. The event ends after --hold s
   (10) of quiet, or is split at --max-len s (60). Only events are saved:
   recorded_sound/YYYY-MM-DD_HH-MM-SS.wav (1 s of pre-roll is NOT kept, 1 block of tail is).
2. YAMNet classifies each loud block; results go to SQLite table `events`:
   timestamp, length, category, likelihood %, + top raw labels, dB, wav file.
3. Web UI on :8080 (open http://<pi-ip>:8080 from any machine on the network):
   listen, change category (saved instantly to the table), delete false alarms, export CSV.

Setup:  sudo apt install -y libportaudio2 python3-scipy python3-numpy python3-flask
        pip install sounddevice tflite-runtime --break-system-packages
Run:    python3 noise_events.py --device 4 --offset 94
        python3 noise_events.py --muted           (no microphone, no model: only the web UI; same as --web-only)
No login on the web UI: keep it on your home network.
"""
import argparse, csv, datetime, io, json, os, queue, re, shutil, socket, sqlite3, struct, threading, time, urllib.request, wave, zlib
from collections import Counter, deque
import numpy as np
from scipy.signal import bilinear, find_peaks, lfilter, medfilt, spectrogram
from flask import Flask, Response, abort, jsonify, request, send_from_directory


def get_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    return Interpreter

SR, BLOCK = 16000, 15600           # YAMNet frame = 0.975 s
BLK_S = BLOCK / SR
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, CMAP, DBPATH = (os.path.join(HERE, f) for f in ("yamnet.tflite", "yamnet_class_map.csv", "events.db"))
MODEL_URLS = [
    "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite",
    "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/classification/tflite/1.tflite",
]
MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
REC_DIR = os.path.join(HERE, "recorded_sound")
SETTINGS = dict(device=4, offset=94.0, delta=8.0, hold=10.0, max_len=60.0, split=3.0, recording=False)   # shown in the "How it works" window

# YAMNet (AudioSet) class names -> your categories. Edit freely; unmapped classes become "other".
CATEGORIES = {
    "voices": ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Shout", "Yell",
               "Screaming", "Crowd", "Children shouting", "Laughter", "Whistling", "Singing", "Babbling"],
    "car_horn": ["Vehicle horn, car horn, honking", "Toot", "Car alarm", "Beep, bleep", "Air horn, truck horn"],
    "siren": ["Siren", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Civil defense siren"],
    "car": ["Car", "Car passing by", "Vehicle", "Motor vehicle (road)", "Traffic noise, roadway noise", "Engine",
            "Accelerating, revving, vroom", "Tire squeal", "Skidding", "Race car, auto racing",
            "Engine starting", "Medium engine (mid frequency)"],
    "scooter_moto": ["Motorcycle", "Light engine (high frequency)"],
    "truck_bus": ["Truck", "Bus", "Heavy engine (low frequency)", "Air brake", "Reversing beeps",
                  "Ice cream truck, ice cream van"],
    "idling_hum": ["Idling", "Hum", "Mechanical fan", "Air conditioning", "Static"],
    "machinery": ["Jackhammer", "Drill", "Power tool", "Hammer", "Chainsaw", "Lawn mower", "Sawing", "Tools",
                  "Dental drill, dentist's drill", "Engine knocking"],
    "clatter_impact": ["Clang", "Clatter", "Bang", "Thump, thud", "Smash, crash", "Crushing", "Rattle",
                       "Crumpling, crinkling", "Squeak", "Slam"],
    "animals": ["Animal", "Dog", "Bark", "Bird", "Bird vocalization, bird call, bird song", "Crow", "Cat", "Meow",
                "Pigeon, dove", "Chirp, tweet", "Domestic animals, pets"],
    "aircraft": ["Aircraft", "Helicopter", "Fixed-wing aircraft, airplane", "Aircraft engine", "Jet engine"],
    "wind_weather": ["Wind", "Wind noise (microphone)", "Rain", "Raindrop", "Thunder", "Thunderstorm", "Rustle"],
}
NAME2CAT = {n.lower(): c for c, ns in CATEGORIES.items() for n in ns}
BASE_CATS = list(CATEGORIES) + ["other", "false_alarm"]
COLORS_PATH = os.path.join(HERE, "category_colors.json")     # your pill colors are saved here
DEFAULT_COLORS = {"voices": "#43a047", "car_horn": "#fb8c00", "siren": "#e53935", "car": "#1e88e5",
                  "truck_bus": "#5e35b1", "idling_hum": "#8d6e63", "machinery": "#fdd835",
                  "clatter_impact": "#d81b60", "animals": "#00acc1", "aircraft": "#78909c",
                  "wind_weather": "#7cb342", "scooter_moto": "#f4511e", "other": "#9e9e9e", "false_alarm": "#616161"}
_colors_lock = threading.Lock()
ENGINE_CATS = ("scooter_moto", "car", "truck_bus", "machinery", "idling_hum", "clatter_impact")   # engine-like events
HORN_CONF = 0.15      # ... need at least this YAMNet horn score before detector beeps are believed
PAIR = ("car", "scooter_moto")                                  # the two classes the learner tells apart
VEH_MODEL = os.path.join(HERE, "vehicle_model.json")
VEH_CLASSES = ["Car", "Car passing by", "Vehicle", "Motor vehicle (road)", "Traffic noise, roadway noise", "Engine",
               "Accelerating, revving, vroom", "Tire squeal", "Skidding", "Race car, auto racing", "Motorcycle",
               "Engine starting", "Light engine (high frequency)", "Medium engine (mid frequency)",
               "Heavy engine (low frequency)", "Idling", "Engine knocking", "Truck", "Bus", "Lawn mower", "Chainsaw",
               "Power tool", "Mechanical fan", "Hum", "Air brake"]


ALIASES_PATH = os.path.join(HERE, "category_aliases.json")   # renamed built-in categories also apply to new events


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_json(path, d):
    with open(path + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(path + ".tmp", path)


def load_colors():
    return load_json(COLORS_PATH)


def category_list():
    al = load_json(ALIASES_PATH)
    out = []
    for c in BASE_CATS + sorted(r[0] for r in sql("select distinct category from events")):
        c = al.get(c, c)
        if c not in out:
            out.append(c)
    return out


# ---------------------------------------------------------------- helpers
def a_weighting(fs):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    num = [(2 * np.pi * f4) ** 2 * 10 ** (1.9997 / 20), 0, 0, 0, 0]
    den = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    den = np.polymul(np.polymul(den, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(num, den, fs)


def db(ms):
    return float(10 * np.log10(ms + 1e-12))


def fetch(urls, path):
    """Download to path trying several URLs; a model file must look like a .tflite (TFL3 header)."""
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return
    for u in ([urls] if isinstance(urls, str) else urls):
        try:
            print("downloading", u)
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if path.endswith(".tflite") and data[4:8] != b"TFL3":
                raise ValueError("not a tflite file")
            with open(path, "wb") as f:
                f.write(data)
            return
        except Exception as e:
            print("  failed:", e)
    print(f"\nCould not download {os.path.basename(path)}. Download it in a browser (see MODEL_URLS in this "
          f"script) and save it as:\n  {path}")
    os._exit(1)


def load_names():
    """Class names: from the label list embedded in the .tflite (it is also a zip), else from the CSV."""
    import zipfile
    try:
        with zipfile.ZipFile(MODEL) as z:
            n = [l.strip() for l in z.read("yamnet_label_list.txt").decode().splitlines() if l.strip()]
        if len(n) == 521:
            return n
    except Exception:
        pass
    fetch(MAP_URL, CMAP)
    with open(CMAP, newline="") as f:
        return [r[2] for r in list(csv.reader(f))[1:]]


def sql(query, params=(), commit=False):
    con = sqlite3.connect(DBPATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(query, params).fetchall()
        if commit:
            con.commit()
        return rows
    finally:
        con.close()


def init_db():
    sql("""create table if not exists events(id integer primary key autoincrement, ts real, length real,
           category text, pct real, raw text, file text, laeq real, peak_db real,
           orig_category text, corrected int default 0)""", commit=True)
    sql("create table if not exists horn(event_id integer primary key, ver int, data text)", commit=True)
    sql("create table if not exists horn_manual(event_id integer primary key, data text, updated real)", commit=True)
    have = {r["name"] for r in sql("pragma table_info(events)")}
    for col in ("labels", "timeline", "feat"):                  # added later: extra labels per event, dominant-sound strip
        if col not in have:
            sql("alter table events add column " + col + " text", commit=True)


TAG_MIN = 0.25      # a category counts as "present" in a block when its score reaches this
OTHER_MIN = 0.15    # below this the dominant sound of a block is "other"


def build_cat_idx(names):
    idx = {}
    for i, n in enumerate(names):
        c = NAME2CAT.get(n.lower())
        if c:
            idx.setdefault(c, []).append(i)
    return {c: np.array(v) for c, v in idx.items()}


def block_info(sc, cat_idx):
    """Score per category (best class of the group) and the dominant sound of one block."""
    cs = {c: float(sc[i].max()) for c, i in cat_idx.items()}
    top = float(sc.max())
    c, v = max(cs.items(), key=lambda kv: kv[1], default=("other", 0.0))
    return cs, ("other", top) if (v < OTHER_MIN or v < 0.5 * top) else (c, v)


def summarize(ev, names):
    """One event -> primary category, extra labels (max-pooled over blocks) and dominant-sound timeline."""
    recs = ev["recs"]
    mx = np.max([r["sc"] for r in recs], axis=0)
    top = np.argsort(mx)[::-1][:3]
    raw = " | ".join("%s %.0f%%" % (names[i], mx[i] * 100) for i in top)
    stats = {}
    for c in {c for r in recs for c in r["cs"]}:
        v = np.array([r["cs"].get(c, 0.0) for r in recs])
        stats[c] = (float(v.max()), float((v >= TAG_MIN).sum() * BLK_S))
    dom = Counter(r["dom"] for r in recs)
    best = max(dom, key=lambda c: (dom[c], stats.get(c, (0.0, 0))[0]))
    pct = round((stats[best][0] if best in stats else float(mx[top[0]])) * 100)
    labels = [[c, round(m * 100), round(sec, 1), "y"] for c, (m, sec) in
              sorted(stats.items(), key=lambda kv: (-kv[1][1], -kv[1][0])) if m >= TAG_MIN]
    runs = []
    for r in recs:
        if runs and runs[-1][2] == r["dom"] and runs[-1][1] == r["idx"] - 1:
            runs[-1][1] = r["idx"]
        else:
            runs.append([r["idx"], r["idx"], r["dom"]])
    return dict(category=best, pct=pct, raw=raw, labels=labels, cmax={c: v[0] for c, v in stats.items()},
                timeline=dict(n=min(ev["last"] + 2, len(ev["blocks"])), r=runs))


def split_point(recs, K, min_score=0.25):
    """Index of the block where a different dominant sound (>= K blocks) took over, else None."""
    if len(recs) < 2 * K:
        return None
    tail = recs[-K:]
    c = tail[-1]["dom"]
    if any(r["dom"] != c for r in tail) or np.mean([r["ds"] for r in tail]) < min_score:
        return None
    prev, n = Counter(r["dom"] for r in recs[:-K]).most_common(1)[0]
    return None if (prev == c or n < K) else tail[0]["idx"]


def split_event(ev, cut):
    ra = [r for r in ev["recs"] if r["idx"] < cut]
    rb = [dict(r, idx=r["idx"] - cut) for r in ev["recs"] if r["idx"] >= cut]
    a = dict(blocks=ev["blocks"][:cut], n_pre=ev["n_pre"], last=ra[-1]["idx"], t0=ev["t0"], recs=ra)
    b = dict(blocks=ev["blocks"][cut:], n_pre=0, last=rb[-1]["idx"],
             t0=ev["t0"] + (cut - ev["n_pre"]) * BLK_S, recs=rb)
    return a, b


def _png(rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_wav(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768


def spectro_rgb(x, W, H, nperseg=512, noverlap=256):
    """Log-frequency spectrogram (60 Hz - 8 kHz, low frequencies at the bottom) as an RGB image."""
    f, t, S = spectrogram(x, SR, nperseg=nperseg, noverlap=noverlap)
    idx = np.searchsorted(f, np.geomspace(60, 8000, H + 1))
    bands = np.array([S[idx[i]:max(idx[i + 1], idx[i] + 1)].mean(0) for i in range(H)])
    D = 10 * np.log10(np.clip(np.nan_to_num(bands), 1e-12, None))
    T = D.shape[1]
    cols = [D[:, int(c * T / W):max(int((c + 1) * T / W), int(c * T / W) + 1)].mean(1) for c in range(W)]
    D = np.stack(cols, 1)
    n = np.clip((D - (D.max() - 60)) / 60, 0, 1)
    pos = [0, .25, .5, .75, 1]
    anchors = np.array([(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)], dtype=float)
    return np.stack([np.interp(n, pos, anchors[:, k]) for k in range(3)], -1).astype(np.uint8)[::-1]


def make_spectrogram(wav_path, png_path, H=48, W=260):
    with open(png_path, "wb") as fh:
        fh.write(_png(np.ascontiguousarray(spectro_rgb(read_wav(wav_path), W, H))))


HORN_MODEL = os.path.join(HERE, "horn_model.json")      # what the beep fixes taught the analyser
TOL_GRID = [round(1.02 + 0.005 * i, 3) for i in range(27)]   # candidate pitch tolerances, 2% to 15%
HORN_VER = 5          # bump to force re-analysis of all events after changing the algorithm
PITCH_TOL = 1.045     # beeps whose pitch differs by less than 4.5% are treated as the same horn


def _swings(ps, thr=0.12):
    """Number of direction reversals of at least thr (12%) in a pitch track; a siren goes up and down, a rev only up."""
    d, ext, n = 0, ps[0], 0
    for v in ps[1:]:
        if d == 0:
            if v > ext * (1 + thr):
                d, ext = 1, v
            elif v < ext * (1 - thr):
                d, ext = -1, v
        elif d == 1:
            if v > ext:
                ext = v
            elif v < ext * (1 - thr):
                d, ext, n = -1, v, n + 1
        elif v < ext:
            ext = v
        elif v > ext * (1 + thr):
            d, ext, n = 1, v, n + 1
    return n


def _runs(mask):
    d = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(d[::2], d[1::2] - 1))


def load_horn_model():
    m = load_json(HORN_MODEL)
    return m if m.get("tol") else None


def beep_feat(dur, snr, f, drift):
    return [dur, snr, f / 1000, drift * 10]


KEEP_MIN_CV = 70      # the learned beep filter is only applied when its cross-validated accuracy reaches this


def horn_keep_ok(m, feat):
    if not m or not m.get("w") or m.get("keep_cv", 0) < KEEP_MIN_CV:
        return True
    z = float(np.dot((np.array(feat) - np.array(m["mean"])) / np.array(m["std"]), m["w"]) + m["b"])
    return z >= 0


def clusters_count(freqs, tol):
    n, last = 0, None
    for f in sorted(freqs):
        if last is None or f > last * tol:
            n, last = n + 1, f
    return n


def pattern_of(nb, longest):
    return ("none" if nb == 0 else ("single_long" if longest >= 1.2 else "single") if nb == 1 else
            "double" if nb == 2 else "series" if nb <= 5 else "long_series")


def horn_effective(det, man):
    """The horn result shown to you: the detector's, unless you fixed the beeps by hand (man = your beep list)."""
    if man is None:
        return det
    b = sorted(man, key=lambda x: x[0])
    gs = sorted({x[2] for x in b})
    longest = max((x[1] - x[0] for x in b), default=0)
    return dict(det, n=len(b), horn_s=round(sum(x[1] - x[0] for x in b), 2), n_pitch=len(gs),
                pattern=pattern_of(len(b), longest), periodic=False, manual=True,
                beeps=[[x[0], x[1], x[3], x[2]] for x in b],
                clusters=[[round(float(np.median([x[3] for x in b if x[2] == g]))), sum(1 for x in b if x[2] == g)]
                          for g in gs])


def editor_beeps(det, man):
    """Rows for the fix window: every detected beep (kept or not), plus beeps you added by hand. [s,e,horn,Hz,keep,auto]"""
    rows, used = [], set()
    for o in det.get("beeps", []):
        s, e, f, g = o[0], o[1], o[2], o[3]
        if man is None:
            rows.append([s, e, g, f, 1, 1])
            continue
        k = max(range(len(man)), key=lambda k: min(e, man[k][1]) - max(s, man[k][0]), default=None)
        if k is not None and min(e, man[k][1]) - max(s, man[k][0]) >= 0.5 * (e - s):
            used.add(k)
            rows.append([s, e, man[k][2], man[k][3] or f, 1, 1])
        else:
            rows.append([s, e, g, f, 0, 1])
    for k, m in enumerate(man or []):
        if k not in used:
            rows.append([m[0], m[1], m[2], m[3], 1, 0])
    return sorted(rows, key=lambda r: r[0])


def sync_horn_labels(i):
    """After a manual fix (or reset): recompute the detector labels of the event from the effective beeps."""
    r = sql("select e.labels, e.length, h.data hd, mn.data md from events e "
            "left join horn h on h.event_id=e.id and h.ver=? left join horn_manual mn on mn.event_id=e.id "
            "where e.id=?", (HORN_VER, i))
    if not r or not r[0]["hd"]:
        return None
    det = json.loads(r[0]["hd"])
    man = json.loads(r[0]["md"])["beeps"] if r[0]["md"] else None
    labs = []
    for l in json.loads(r[0]["labels"] or "[]"):
        if l[3] == "d":                              # added by the detector before: recomputed below
            continue
        labs.append([l[0], l[1], l[2], "y" if l[3] == "yd" else l[3]])
    labs = detector_tags(labs, horn_effective(det, man), r[0]["length"], load_json(ALIASES_PATH),
                         trust=True if man is not None else det.get("trusted", True))
    sql("update events set labels=? where id=?", (json.dumps(labs), i), commit=True)
    return labs


def horn_summary(i):
    r = sql("select h.data hd, mn.data md from horn h left join horn_manual mn on mn.event_id=h.event_id "
            "where h.event_id=? and h.ver=?", (i, HORN_VER))
    if not r or not r[0]["hd"]:
        return None
    man = json.loads(r[0]["md"])["beeps"] if r[0]["md"] else None
    h = horn_effective(json.loads(r[0]["hd"]), man)
    return dict(n=h["n"], s=h["horn_s"], p=h.get("periodic", False), ok=h.get("trusted", True), m=man is not None)


def learn_horn():
    """Learn from your beep fixes: which detected beeps are real (beep filter), and how far apart pitches
    must be to count as one horn (pitch rule). Afterwards the other events are analysed again."""
    rows = sql("select mn.data md, e.file from horn_manual mn join events e on e.id=mn.event_id")
    feats, labels, evs = [], [], []
    for r in rows:
        man = json.loads(r["md"])["beeps"]
        try:     # re-detect without the learned filter, so beeps it rejected still appear as negatives
            det = analyze_horn(os.path.join(REC_DIR, os.path.basename(r["file"])), filtered=False)["beeps"]
        except Exception as e:
            print("learn: skipped", r["file"], e)
            continue
        kept, kmax = [], len({m[2] for m in man})
        for d in det:
            if len(d) < 7:
                continue
            ov = max([min(d[1], m[1]) - max(d[0], m[0]) for m in man] or [0.0])
            k = int(ov >= 0.5 * (d[1] - d[0]))
            feats.append(beep_feat(d[6], d[4], d[2], d[5]))
            labels.append(k)
            if k:
                kept.append(d[2])
        evs.append((kept, kmax))
    if len(evs) < 3:
        return dict(ok=False, msg="Fix at least 3 events first (now %d)." % len(evs))
    hm = load_horn_model() or {}
    tol0 = hm.get("tol", PITCH_TOL)
    err = lambda tol: sum(abs(clusters_count(f, tol) - k) for f, k in evs)
    best = min(TOL_GRID, key=lambda t: (err(t), abs(t - PITCH_TOL)))
    err0, err1 = err(tol0), err(best)
    m = dict(hm)
    X, y = np.array(feats, dtype=float), np.array(labels, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if len(y) >= 20 and min(n0, n1) >= 3:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        order = np.random.RandomState(0).permutation(len(y))
        k, hit = min(5, n0, n1), 0
        for fold in range(k):
            te = order[fold::k]
            tr = np.setdiff1d(order, te)
            w, b = _fit(Z[tr], y[tr])
            hit += int((((Z[te] @ w + b) >= 0) == (y[te] == 1)).sum())
        w, b = _fit(Z, y)
        m.update(w=w.tolist(), b=float(b), mean=mu.tolist(), std=sd.tolist(), keep_cv=round(100 * hit / len(y)))
        keep_msg = ("beep filter: %d detections rejected, %d kept, cross-validated accuracy %d%%, %s" %
                    (n0, n1, m["keep_cv"], "applied" if m["keep_cv"] >= KEEP_MIN_CV else
                     "NOT applied yet (needs %d%%): fix more events" % KEEP_MIN_CV))
    else:
        keep_msg = "beep filter not learned yet (%d detected beeps, need 20 with at least 3 of each kind)" % len(y)
    m.update(tol=best, err_before=err0, err_after=err1, tol_events=len(evs))
    save_json(HORN_MODEL, m)
    sql("delete from horn where event_id not in (select event_id from horn_manual)", commit=True)
    threading.Thread(target=horn_backfill, daemon=True).start()
    return dict(ok=True, msg="Learned from %d fixed events. Pitch rule %s (was %s), wrong horn counts %d → %d. %s. "
                "Re-analysing the other events in the background." % (len(evs), best, tol0, err0, err1, keep_msg))


def analyze_horn(wav_path, fmin=180, fmax=2500, snr_db=15.0, filtered=True):
    """Count horn beeps (separate tonal bursts) and group them by pitch (~ different horns / cars)."""
    with wave.open(wav_path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768
    dur = len(x) / SR
    f, t, S = spectrogram(x, SR, nperseg=1024, noverlap=768)          # 64 ms window, 16 ms hop
    P = 10 * np.log10(S + 1e-14)
    band = (f >= fmin) & (f <= fmax)
    fb, Pb, df, hop = f[band], P[band], f[1] - f[0], t[1] - t[0]
    n = Pb.shape[1]
    active, pitch = np.zeros(n, bool), np.zeros(n)
    exc = np.zeros(n)                       # peak height above the noise floor, per frame
    strong, ridge = np.zeros(n, bool), np.zeros(n)
    for j in range(n):                      # tonal frame = narrow spectral peak well above the noise floor
        col = Pb[:, j]
        med = np.median(col)
        ridge[j], strong[j] = fb[int(col.argmax())], col.max() - med >= 12.0     # for sweeping tones (sirens)
        k = [i for i in find_peaks(col, height=med + snr_db)[0]
             if col[i] - max(col[max(i - 3, 0)], col[min(i + 3, len(col) - 1)]) >= 6]
        if not k:
            continue
        snr = col[k] - med
        i = min(i for i, sn in zip(k, snr) if sn >= snr.max() - 10)   # lowest strong peak = fundamental
        a, b, c = col[max(i - 1, 0)], col[i], col[min(i + 1, len(col) - 1)]
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        pitch[j], active[j] = fb[i] + float(np.clip(d, -1, 1)) * df, True
        exc[j] = float(col[i] - med)
    gap, minf = int(0.09 / hop), int(0.08 / hop)
    merged = []
    for a, b in _runs(active):
        if merged and a - merged[-1][1] - 1 <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    segs = []
    for a, b in merged:
        if b - a + 1 < minf or active[a:b + 1].mean() < 0.75:      # a horn is solid; engine tones flicker
            continue
        ps = pitch[a:b + 1][active[a:b + 1]]
        n3 = max(len(ps) // 3, 1)
        ang = 2 * np.pi * (np.log2(ps) % 1)          # pitch class, so hopping between harmonics is not a "glide"
        mid = lambda v: np.angle(np.mean(np.exp(1j * v)))
        drift = abs(np.angle(np.exp(1j * (mid(ang[-n3:]) - mid(ang[:n3]))))) / (2 * np.pi)      # octaves, start vs end
        segs.append((a, b, float(np.median(ps)), float(drift), (b - a + 1) * hop))
    segs = [g for g in segs if g[3] < 0.12 or g[1] - g[0] + 1 < 12]    # a horn holds its pitch; a revving engine glides
    sirens = []                             # siren = strong tone that sweeps smoothly over >= 25% for >= 1 s
    sm = []
    for a, b in _runs(strong):
        if sm and a - sm[-1][1] - 1 <= 3:
            sm[-1][1] = b
        else:
            sm.append([a, b])
    for a, b in sm:
        ps = ridge[a:b + 1]
        ps = medfilt(ps, 5) if len(ps) >= 5 else ps
        if (b - a + 1) * hop >= 1.0 and np.median(ps) >= 450 and _swings(ps) >= 1 \
                and (np.percentile(ps, 95) - np.percentile(ps, 5)) / np.median(ps) >= 0.25 \
                and (np.abs(np.diff(ps)) / ps[:-1] < 0.15).mean() >= 0.8:
            sirens.append((a, b, 0, 0, (b - a + 1) * hop))
    segs = [g for g in segs if not any(g[0] <= sb and sa <= g[1] for sa, sb, *_ in sirens)]   # sirens are not beeps
    hm = load_horn_model() if filtered else None
    snr_of = lambda g: float(np.median(exc[g[0]:g[1] + 1][active[g[0]:g[1] + 1]]))
    segs = [g for g in segs if horn_keep_ok(hm, beep_feat(g[4], snr_of(g), g[2], g[3]))]    # learned beep filter
    tol = hm["tol"] if hm else PITCH_TOL                                                  # learned pitch rule
    beeps = [(g[0], g[1]) for g in segs]
    freqs = [g[2] for g in segs]
    groups, cid = [], [0] * len(beeps)
    for i in sorted(range(len(beeps)), key=lambda i: freqs[i]):
        if groups and freqs[i] <= groups[-1][0] * tol:
            groups[-1][1].append(i)
        else:
            groups.append([freqs[i], [i]])
    for c, (_, idxs) in enumerate(groups):
        for i in idxs:
            cid[i] = c
    out = [[float(round(max(t[a] - 0.032, 0), 2)), float(round(min(t[b] + 0.032, dur), 2)), int(round(fr)), cid[i],
            round(snr_of(g), 1), round(g[3], 3), round(g[4], 3)]
           for i, ((a, b), fr, g) in enumerate(zip(beeps, freqs, segs))]
    longest = max((o[1] - o[0] for o in out), default=0)
    nb = len(out)
    iv, du = np.diff([o[0] for o in out]), [o[1] - o[0] for o in out]
    periodic = bool(nb >= 4 and len(groups) == 1 and freqs and np.median(freqs) >= 700
                    and iv.std() / iv.mean() < 0.15 and np.std(du) / np.mean(du) < 0.25)   # evenly spaced, one pitch, high
    pattern = ("periodic" if periodic else "none" if nb == 0 else ("single_long" if longest >= 1.2 else "single") if nb == 1 else
               "double" if nb == 2 else "series" if nb <= 5 else "long_series")
    return dict(n=nb, horn_s=round(sum(o[1] - o[0] for o in out), 2), n_pitch=len(groups), pattern=pattern,
                dur=round(dur, 2), beeps=out, periodic=periodic, lf=round(float(S[f < 200].sum() / (S.sum() + 1e-20)), 2),
                siren_s=round(sum(g[4] for g in sirens), 2),
                sirens=[[round(float(max(t[g[0]] - 0.032, 0)), 2), round(float(min(t[g[1]] + 0.032, dur)), 2)] for g in sirens],
                clusters=[[round(float(np.median([freqs[i] for i in ix]))), len(ix)] for _, ix in groups])


def veh_index(names):
    pos = {n: i for i, n in enumerate(names)}
    return [pos[n] for n in VEH_CLASSES if n in pos]


def engine_features(x):
    """Spectral character of the loud part: share below 200 Hz, share 500-4000 Hz, centroid (fraction of 4 kHz)."""
    f, t, S = spectrogram(x, SR, nperseg=1024, noverlap=512)
    E = S.sum(0)
    P = S[:, E >= np.percentile(E, 50)].mean(1)
    tot = P.sum() + 1e-20
    return [float(P[f < 200].sum() / tot), float(P[(f >= 500) & (f <= 4000)].sum() / tot),
            float((f * P).sum() / tot / 4000)]


def feat_vector(recs, audio, length, veh_idx):
    """Vehicle-related YAMNet scores (max over blocks) + spectral character + length: input of the learner."""
    mx = np.max([r["sc"] for r in recs], axis=0)
    return [round(float(v), 4) for v in list(mx[veh_idx]) + engine_features(audio) + [min(length, 30) / 30]]


def load_vehicle_model():
    m = load_json(VEH_MODEL)
    return m if m.get("w") else None


def apply_vehicle_model(m, feat):
    if len(feat) != len(m["w"]):
        return None
    z = float(np.dot((np.array(feat) - np.array(m["mean"])) / np.array(m["std"]), m["w"]) + m["b"])
    p = 1 / (1 + np.exp(-z))
    return ("scooter_moto", p) if p >= 0.5 else ("car", 1 - p)


def _fit(Z, y, steps=1500, lr=0.3):
    sw = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max(len(y) - y.sum(), 1))      # balanced classes
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        g = sw * (1 / (1 + np.exp(-(Z @ w + b))) - y)
        w -= lr * (Z.T @ g + 0.01 * w)
        b -= lr * g.sum()
    return w, b


def train_vehicle_model():
    """Learn car vs scooter_moto from events you labelled by hand, then relabel the ones you did not touch."""
    rows = sql("select category, feat from events where corrected=1 and feat is not null "
               "and category in ('car','scooter_moto')")
    nofeat = sql("select count(*) c from events where feat is null")[0]["c"]
    feats = [json.loads(r["feat"]) for r in rows]
    ln = Counter(len(f) for f in feats).most_common(1)[0][0] if feats else 0
    rows, feats = zip(*[(r, f) for r, f in zip(rows, feats) if len(f) == ln]) if feats else ((), ())
    y = np.array([1.0 if r["category"] == "scooter_moto" else 0.0 for r in rows])
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if min(n0, n1) < 8:
        return dict(ok=False, msg="Need at least 8 hand-labelled events of each type with analysis data (now: car %d, "
                    "scooter_moto %d)%s." % (n0, n1, "; older events need the one-time command "
                    "python3 noise_events.py --reprocess" if nofeat else ""))
    X = np.array(feats, dtype=float)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    order = np.random.RandomState(0).permutation(len(y))
    k, hit = min(5, n0, n1), 0
    for fold in range(k):                                   # cross-validation: how well does it do on unseen events?
        te = order[fold::k]
        tr = np.setdiff1d(order, te)
        w, b = _fit(Z[tr], y[tr])
        hit += int((((Z[te] @ w + b) > 0) == (y[te] == 1)).sum())
    w, b = _fit(Z, y)
    acc = round(100 * hit / len(y))
    m = dict(w=w.tolist(), b=float(b), mean=mu.tolist(), std=sd.tolist(), n_car=n0, n_scooter=n1, cv_acc=acc)
    save_json(VEH_MODEL, m)
    changed = 0
    for r in sql("select id, category, feat from events where corrected=0 and feat is not null "
                 "and category in ('car','scooter_moto')"):
        res = apply_vehicle_model(m, json.loads(r["feat"]))
        if res:
            changed += res[0] != r["category"]
            sql("update events set category=?, pct=?, orig_category=? where id=?",
                (res[0], round(res[1] * 100), res[0], r["id"]), commit=True)
    return dict(ok=True, msg="Learned from %d car + %d scooter_moto events (cross-validated accuracy %d%%). "
                "Re-labelled %d events you had not corrected." % (n0, n1, acc, changed))


def horn_backfill():
    """Background task: (re)run the signal detectors for events whose horn data is missing or from an older version."""
    while True:
        rows = sql("select id,file,length,labels,category from events where id not in (select event_id from horn where ver=?) "
                   "order by id desc limit 20", (HORN_VER,))
        if not rows:
            return
        al = load_json(ALIASES_PATH)
        for r in rows:
            det = dict(n=0, horn_s=0, n_pitch=0, pattern="error", dur=0, beeps=[], clusters=[], periodic=False, lf=0,
                       siren_s=0, sirens=[], trusted=True)
            try:
                det = analyze_horn(os.path.join(REC_DIR, os.path.basename(r["file"])))
                labs = json.loads(r["labels"] or "[]")
                hn = al.get("car_horn", "car_horn")
                conf = max([(l[1] or 0) / 100 for l in labs if l[0] == hn and "y" in l[3]], default=0.0)
                prim = next((k for k, v in al.items() if v == r["category"]), r["category"])     # name before renaming
                det["trusted"] = horn_trusted(prim, conf)
                labels = [[l[0], l[1], l[2], "y"] for l in labs if l[3] != "d"]
                mrow = sql("select data from horn_manual where event_id=?", (r["id"],))
                man = json.loads(mrow[0]["data"])["beeps"] if mrow else None      # your fixes win over the detector
                trust = True if man is not None else det["trusted"]
                sql("update events set labels=? where id=?",
                    (json.dumps(detector_tags(labels, horn_effective(det, man), r["length"], al, trust=trust)), r["id"]),
                    commit=True)
            except Exception as e:
                print("horn backfill failed:", r["file"], e)
            sql("insert or replace into horn(event_id,ver,data) values(?,?,?)", (r["id"], HORN_VER, json.dumps(det)),
                commit=True)
            time.sleep(0.2)                                  # stay gentle with the recorder on a Pi


def reprocess(args, mode):
    """Run YAMNet + detectors again on saved recordings (fills labels, strip, features; relabels uncorrected events)."""
    names, cat_idx, score = load_model()
    veh, (b, a) = veh_index(names), a_weighting(SR)
    model, al = load_vehicle_model(), load_json(ALIASES_PATH)
    rows = sql("select id,file,length,corrected from events" + (" where feat is null" if mode == "missing" else "")
               + " order by id")
    print("reprocessing %d recordings..." % len(rows))
    for k, r in enumerate(rows):
        path = os.path.join(REC_DIR, os.path.basename(r["file"]))
        try:
            with wave.open(path) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768
            nb = len(x) // BLOCK
            blocks = [x[i * BLOCK:(i + 1) * BLOCK] for i in range(nb)]
            y = lfilter(b, a, x[:nb * BLOCK])
            lv = [db(np.mean(y[i * BLOCK:(i + 1) * BLOCK] ** 2)) + args.offset for i in range(nb)]
            loud = [i for i, l in enumerate(lv) if l > min(lv) + 0.6 * args.delta] or [int(np.argmax(lv))]
            recs = []
            for i in loud:
                sc = score(blocks[i])
                cs, (dom, ds) = block_info(sc, cat_idx)
                recs.append(dict(idx=i, lvl=lv[i], sc=sc, cs=cs, dom=dom, ds=ds))
            sm = summarize(dict(blocks=blocks, n_pre=loud[0], last=loud[-1], recs=recs), names)
            det = analyze_horn(path)
            feat = feat_vector(recs, np.concatenate(blocks[:loud[-1] + 2]), r["length"], veh)
            prim, pct = sm["category"], sm["pct"]
            if model and prim in PAIR:
                res = apply_vehicle_model(model, feat)
                if res:
                    prim, pct = res[0], round(res[1] * 100)
            det["trusted"] = horn_trusted(prim, sm["cmax"].get("car_horn", 0.0))
            labels = detector_tags(sm["labels"], det, r["length"], trust=det["trusted"])
            for l in labels:
                l[0] = al.get(l[0], l[0])
            sm["timeline"]["r"] = [[p, q, al.get(c, c)] for p, q, c in sm["timeline"]["r"]]
            cat = al.get(prim, prim)
            sql("update events set raw=?, labels=?, timeline=?, feat=? where id=?",
                (sm["raw"], json.dumps(labels), json.dumps(sm["timeline"]), json.dumps(feat), r["id"]), commit=True)
            if not r["corrected"]:
                sql("update events set category=?, pct=?, orig_category=? where id=?", (cat, pct, cat, r["id"]),
                    commit=True)
            sql("insert or replace into horn(event_id,ver,data) values(?,?,?)", (r["id"], HORN_VER, json.dumps(det)),
                commit=True)
        except Exception as e:
            print("  failed:", r["file"], e)
        if k % 20 == 0:
            print("  %d / %d" % (k + 1, len(rows)))
    print("done")


def horn_trusted(primary, conf):
    """Beeps in engine-dominated events (scooters, cars, trucks) only count when YAMNet also hears a horn."""
    return primary not in ENGINE_CATS or conf >= HORN_CONF


def detector_tags(labels, det, length, al=None, trust=True):
    """Add what the signal detectors found (horn beeps, siren, regular beeper, low rumble) as extra labels."""
    add = []
    if det:
        if det["periodic"]:
            add.append(("reversing_beeps", det["horn_s"]))
        elif det["n"] and trust:
            add.append(("car_horn", det["horn_s"]))
        if det["siren_s"]:
            add.append(("siren", det["siren_s"]))
        if det["lf"] >= 0.6 and length >= 1.5:
            add.append(("rumble", length))
    for name, sec in add:
        name = (al or {}).get(name, name)
        for l in labels:
            if l[0] == name:
                l[2], l[3] = max(l[2], round(sec, 1)), "yd"
                break
        else:
            labels.append([name, None, round(sec, 1), "d"])
    return labels


def save_event(ev, names):
    first, last = ev["n_pre"], ev["last"]
    length = round((last - first + 1) * BLK_S, 1)
    audio = np.concatenate(ev["blocks"][:last + 2])
    fn = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(ev["t0"])) + ".wav"
    path = os.path.join(REC_DIR, fn)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    try:
        make_spectrogram(path, os.path.join(REC_DIR, fn[:-4] + ".png"))
    except Exception as e:
        print("spectrogram failed:", e)
    try:
        det = analyze_horn(path)
    except Exception as e:
        print("detectors failed:", e)
        det = None
    sm = summarize(ev, names)
    feat = feat_vector(ev["recs"], audio, length, veh_index(names))
    model = load_vehicle_model()
    if model and sm["category"] in PAIR:                 # car vs scooter_moto decided by what you taught it
        res = apply_vehicle_model(model, feat)
        if res:
            sm["category"], sm["pct"] = res[0], round(res[1] * 100)
    trust = horn_trusted(sm["category"], sm["cmax"].get("car_horn", 0.0))
    if det:
        det["trusted"] = trust
    al = load_json(ALIASES_PATH)
    labels = detector_tags(sm["labels"], det, length, trust=trust)
    for l in labels:
        l[0] = al.get(l[0], l[0])
    cat = al.get(sm["category"], sm["category"])
    sm["timeline"]["r"] = [[a, b, al.get(c, c)] for a, b, c in sm["timeline"]["r"]]
    lv = np.array([r["lvl"] for r in ev["recs"]])
    con = sqlite3.connect(DBPATH, timeout=10)
    cur = con.execute("insert into events(ts,length,category,pct,raw,file,laeq,peak_db,orig_category,labels,timeline,feat) "
                      "values(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (ev["t0"], length, cat, sm["pct"], sm["raw"], fn, round(db(np.mean(10 ** (lv / 10))), 1),
                       round(float(lv.max()), 1), cat, json.dumps(labels), json.dumps(sm["timeline"]), json.dumps(feat)))
    if det:
        con.execute("insert or replace into horn(event_id,ver,data) values(?,?,?)", (cur.lastrowid, HORN_VER, json.dumps(det)))
    con.commit()
    con.close()
    print("%s  %ss  %s %s%%  peak %.0f dB  labels: %s" % (fn, length, cat, sm["pct"], lv.max(),
                                                         ", ".join(l[0] for l in labels)))


# ---------------------------------------------------------------- recorder
def open_stream(device, q):
    import sounddevice as sd
    return sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK, dtype="float32", device=device,
                          callback=lambda d, f, t, s: q.put(d[:, 0].copy()))


def load_model():
    fetch(MODEL_URLS, MODEL)
    names = load_names()
    it = get_interpreter()(model_path=MODEL)
    it.allocate_tensors()
    i_in, i_out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def score(x):
        it.set_tensor(i_in, x.astype(np.float32))
        it.invoke()
        return it.get_tensor(i_out).reshape(-1)[:len(names)].copy()
    return names, build_cat_idx(names), score


def recorder(args):
    names, cat_idx, score = load_model()

    def new_rec(idx, lvl, x):
        sc = score(x)
        cs, (dom, ds) = block_info(sc, cat_idx)
        return dict(idx=idx, lvl=lvl, sc=sc, cs=cs, dom=dom, ds=ds)

    b, a = a_weighting(SR)
    zi = np.zeros(max(len(a), len(b)) - 1)
    hold = int(args.hold / BLK_S + 0.5)
    max_blocks = int(args.max_len / BLK_S)
    K = max(2, int(round(args.split / BLK_S))) if args.split > 0 else 0
    q, pre, warm, floor, ev = queue.Queue(), deque(maxlen=1), [], None, None
    print("listening... (first seconds calibrate the background level)")
    with open_stream(args.device, q):
        while True:
            x = q.get()
            t_end = time.time()
            y, zi = lfilter(b, a, x, zi=zi)
            lvl = db(np.mean(y ** 2)) + args.offset
            if floor is None:
                warm.append(lvl)
                pre.append(x)
                if len(warm) >= 3:
                    floor = float(np.median(warm))
                continue
            loud = lvl > floor + args.delta * (0.6 if ev else 1)
            if ev is None:
                if loud:
                    ev = dict(blocks=list(pre) + [x], n_pre=len(pre), last=len(pre), t0=t_end - BLK_S,
                              recs=[new_rec(len(pre), lvl, x)])
                else:
                    floor += 0.05 * (lvl - floor)      # follow slow background changes
                    pre.append(x)
                continue
            ev["blocks"].append(x)
            floor += 0.003 * (lvl - floor)
            if loud:
                ev["last"] = len(ev["blocks"]) - 1
                ev["recs"].append(new_rec(ev["last"], lvl, x))
                cut = split_point(ev["recs"], K) if K else None
                if cut is not None:                     # a different dominant sound took over: close the first part
                    first, ev = split_event(ev, cut)
                    save_event(first, names)
            quiet = len(ev["blocks"]) - 1 - ev["last"]
            forced = len(ev["blocks"]) - ev["n_pre"] >= max_blocks
            if quiet >= hold or forced:
                if forced and loud:                     # endless loud noise becomes the new background
                    floor += 0.3 * (np.mean([r["lvl"] for r in ev["recs"]]) - floor)
                save_event(ev, names)
                ev = None
                pre.clear()


# ---------------------------------------------------------------- web
PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<script>try{var t=localStorage.getItem('ne_theme');if(t==='dark'||t==='light')document.documentElement.dataset.theme=t}catch(e){}</script>
<title>Noise events</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}
html[data-theme=dark]{color-scheme:dark}
html[data-theme=light]{color-scheme:light}
body{margin:0;padding:8px 14px}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;position:sticky;top:0;background:Canvas;padding:8px 0;z-index:2}
h2{margin:0 8px 0 0}.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:3px 8px;border-bottom:1px solid #8884;text-align:left;vertical-align:middle;white-space:nowrap}
tr.fixed td{background:#2a92}small{opacity:.7}audio{height:32px;width:260px}
button,select{font:inherit}#st span{margin-right:6px;cursor:pointer;white-space:nowrap;padding:2px 10px;border-radius:999px;font-size:13px;font-weight:600}
.pill{border:0;border-radius:999px;padding:3px 8px;font-weight:600}option{background:Canvas;color:CanvasText}
.sw{width:24px;height:22px;padding:0;border:0;background:none;vertical-align:middle;cursor:pointer}
.lis{display:flex;align-items:center;gap:8px}
.sp{position:relative;width:182px;height:34px;flex:none;cursor:pointer}
.sp img{width:100%;height:100%;display:block;border-radius:4px;background:#000}
dialog{width:min(1100px,96vw);border:1px solid #8886;border-radius:12px;padding:12px 16px;color:CanvasText;background:Canvas}
dialog{top:3vh;bottom:auto;margin:0 auto}   /* windows keep their top edge when content changes */
dialog::backdrop{background:#0009}.anh{display:flex;flex-wrap:wrap;gap:14px 14px;align-items:center;margin-bottom:6px}
.cl{opacity:.75;font-size:13px;margin-right:6px}.cl2{display:block;margin-top:12px}
.seg button,.chips button{border:1px solid #8886;background:none;color:inherit;padding:3px 10px;cursor:pointer;font:inherit}
.seg button:first-child{border-radius:999px 0 0 999px}.seg button:last-child{border-radius:0 999px 999px 0}
.chips{margin:4px 0 6px}.chips button{border-radius:999px;margin:0 6px 10px 0}.seg .on,.chips .on{background:Highlight;color:HighlightText}
#anlg{margin-top:8px}#anlg span{display:inline-block;margin:0 6px 6px 0;padding:2px 10px;border-radius:999px;font-size:13px;font-weight:600;cursor:pointer}
#anlg .off{opacity:.3}#antt{position:fixed;display:none;pointer-events:none;background:Canvas;border:1px solid #8886;border-radius:8px;padding:6px 10px;font-size:13px;z-index:10;box-shadow:0 2px 10px #0006}
#antt i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}rect[data-i]:hover{fill:#8883}
.cards{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.cards div{border:1px solid #8886;border-radius:10px;padding:6px 12px;min-width:110px}
.cards b{display:block;font-size:20px}.cards span{font-size:12px;opacity:.7}.hg{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:700px){.hg{grid-template-columns:1fr}}.capt{opacity:.75;font-size:13px;margin-top:4px}dialog{max-height:94vh;overflow:auto}
.ren,.ok,.rs{border:0;background:none;cursor:pointer;opacity:.55;color:inherit;padding:0 2px}.ren:hover,.ok:hover,.rs:hover{opacity:1}
.info p{margin:8px 0;line-height:1.45}#inf{width:min(760px,96vw)}
.tls{position:absolute;left:0;right:0;bottom:0;height:5px;pointer-events:none}.tls span{position:absolute;top:0;bottom:0}
.tg{display:inline-block;margin:0 3px 0 0;padding:0 7px;border-radius:999px;font-size:11px;font-weight:600}
.dnav{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.dnav input{font:inherit;padding:2px 4px}#dcnt{font-size:13px;opacity:.8;margin-left:4px}
#hfx{width:min(760px,96vw)}
#hfx{top:3vh;bottom:auto;margin:0 auto;max-height:92vh}   /* pinned top: adding or deleting rows does not move the window */#hfxb tr.off td{opacity:.45}.hfx{border:0;background:none;cursor:pointer;opacity:.6;color:inherit}.hfx:hover{opacity:1}
#hcl{position:fixed;overflow:hidden;z-index:1;pointer-events:none;background:Canvas;box-shadow:0 2px 6px #0004}
#hcl th{background:Canvas}
.sk{opacity:.55;font-size:11px;white-space:nowrap}
#toast{position:fixed;left:14px;bottom:14px;background:#000d;color:#fff;padding:8px 14px;border-radius:8px;font-size:14px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:9}
th[draggable=true]{cursor:grab}
.cur{position:absolute;top:0;bottom:0;width:2px;left:0;background:#fff;mix-blend-mode:difference;pointer-events:none}
</style>
<header><h2>Noise events</h2>
<span class=dnav><button id=dprev title="previous day with events">◀</button><input type=date id=ddate title="jump to a date">
<button id=dnext title="next day with events">▶</button><button id=dtod>today</button><button id=dall title="show events of all days, newest first">all days</button><span id=dcnt></span><span id=dsum style="opacity:.75;font-size:13px"></span><button id=dmore style="display:none" title="load older events of this day">load older</button></span>
<select id=f><option value="">all categories</option></select><button id=fr style="display:none" title="show everything again">✕ reset filter</button>
<button id=anb>📊 Analysis</button> <button id=hnb>🔊 Horns</button> <button id=inb>ℹ How it works</button>
<label><input type=checkbox id=u> not reviewed only</label>
<span id=st></span><span id=th class=seg><button data-th=auto title="theme: follow the device">🖥</button><button data-th=dark title="dark theme">🌙</button><button data-th=light title="light theme">☀</button></span> <button id=colr title="reset the column order">⇆ columns</button> <a href="/export.csv">export CSV</a></header>
<div class=tw><table><thead><tr><th>time<th>length<th>avg / peak<th>vol<th>category<th>conf.<th>beeps · sound<th>other sounds<th>listen<th></thead>
<tbody id=tb></tbody></table></div>
<dialog id=inf><div class=anh><b style="font-size:18px">How sound is recorded</b><button id=infx style="margin-left:auto">✕</button></div>
<div id=infb class=info></div></dialog>
<dialog id=hfx><div class=anh><b style="font-size:17px">Fix beeps</b><span id=hfxs style="opacity:.75"></span><button id=hfxx style="margin-left:auto">✕</button></div>
<div class=anh style="margin:6px 0 2px"><button id=hfxzo title="zoom out">−</button><button id=hfxzi title="zoom in">+</button><button id=hfxzf>full</button>
 <input type=range id=hfxpan min=0 max=1000 value=0 style="flex:1;display:none;min-width:120px"><span id=hfxz style="opacity:.75;font-size:12px"></span></div>
<div id=hfxw style="position:relative;margin:4px 0;user-select:none">
 <img id=hfxi alt="spectrogram" draggable=false style="width:100%;height:120px;display:block;border-radius:4px;background:#000;cursor:crosshair">
 <div id=hfxo style="position:absolute;inset:0;cursor:crosshair"></div>
 <i id=hfxp style="position:absolute;top:0;bottom:0;width:2px;background:#fff;left:0;pointer-events:none;box-shadow:0 0 3px #000"></i></div>
<audio id=hfxa controls preload=metadata style="width:100%;margin:4px 0"></audio>
<p style="font-size:12px;opacity:.7;margin:0 0 6px">Spectrogram: time left to right, low pitch at the bottom. Coloured bars are the beeps (one colour per horn): left click a bar to switch its horn colour, right click to remove it. Drag on empty space to add a beep with exactly that start and end; a plain click adds one of typical length. Zoom in for precision, or type the start and length in the table.</p>
<div style="max-height:45vh;overflow:auto"><table><thead><tr><th>keep<th>play<th>start<th>length<th>pitch<th>which horn<th>source<th></thead><tbody id=hfxb></tbody></table></div>
<p style="font-size:12px;opacity:.7;margin:8px 0">Same horn = same pitch: give beeps from one horn the same letter (A, B, …). Use the table or the spectrogram to remove or add beeps; "+ add beep" lets you type an exact start time.</p>
<div class=anh><button id=hfxadd>+ add beep</button><button id=hfxsave>save</button><button id=hfxrst title="drop your fixes for this event">reset to auto</button></div></dialog>
<dialog id=hn style="width:min(1265px,96vw)"><div class=anh><b style="font-size:18px">Horn analysis</b>
 <select id=hnc></select>
 <label style="display:inline-flex;align-items:center;gap:6px" title="show only events whose average level reaches this value">min level <input type=range id=hnml min=30 max=90 step=1 value=30 style="width:110px"> <span id=hnmlv style="min-width:76px">all levels</span></label>
 <span class=seg id=hnv><button data-v=list class=on>events</button><button data-v=spec>spectrograms</button></span>
 <select id=hnr><option value=24>last 24 hours</option><option value=168>last 7 days</option><option value=720>last 30 days</option><option value=0 selected>all time</option></select>
 <span id=hnst style="opacity:.7"></span><button id=hnl title="learn from your beep fixes to improve the next predictions">🧠 Learn from fixes</button><button id=hnx style="margin-left:auto">✕</button></div>
<div id=hns class=cards></div>
<div class=hg><div><div class=capt>beeps per event</div><div id=hnh1></div></div>
 <div><div class=capt>distinct horn pitches per event (≈ cars)</div><div id=hnh2></div></div></div>
<div id=hntw style="max-height:42vh;overflow:auto"><table><thead><tr><th>time<th>beeps<th>pattern<th>pitches (Hz × beeps)<th>timeline (click to play from there)<th></thead><tbody id=hnt></tbody></table></div>
<div id=hns2 style="display:none;max-height:58vh;overflow:auto;padding-right:6px"></div>
<p style="opacity:.65;font-size:12px;margin:8px 0 0">Beeps = separate tonal bursts found in the saved recording. Cars are estimated from distinct horn pitches (beeps within 4.5% count as one horn): two cars with the same horn model look like one, and a pitch that drifts a lot (fast pass-by, Doppler) may look like two. Regular, evenly spaced beepers (reversing, alarms) are flagged as such, and long sweeping tones (sirens) are not counted as beeps. Events where a horn was heard under another sound are included too.</p></dialog>
<dialog id=an><div class=anh><b style="font-size:18px">Analysis</b>
 <span class=seg id=anmode><button data-m=recent>Recent hours</button><button data-m=hod>By hour of day</button><button data-m=src>Sources</button><button data-m=heat>Heat map</button></span>
 <select id=anr></select>
 <select id=ans><option value=900>15 min</option><option value=3600>1 hour</option><option value=86400>per day</option><option value=0>free: each event</option></select>
 <select id=anm><option value=n>events</option><option value=s>sound minutes</option></select>
 <label><input type=checkbox id=ann> 100% mix</label>
 <label id=anavl><input type=checkbox id=anav checked> avg per day</label>
 <span id=anheat style="display:none"><select id=anhm><option value=avg>average level</option><option value=pk>peak level</option><option value=s>sound minutes / day</option><option value=n>events / day</option></select></span>
 <span id=ansrc style="display:none"><select id=anpe><option value=all>all day</option><option value=day>day 07–19</option><option value=evening>evening 19–23</option><option value=night>night 23–07</option></select>
 <select id=anlv><option value=avg>loudness: average</option><option value=pk>loudness: peak</option></select>
 <select id=anvw><option value=scatter>view: scatter</option><option value=bars>view: share bars</option><option value=thr>view: time above threshold</option><option value=stats>view: level statistics</option></select>
 <select id=anth><option>40</option><option>45</option><option>50</option><option>55</option><option>60</option><option>65</option><option>70</option><option>75</option><option>80</option><option>85</option><option>90</option></select>
 <label><input type=checkbox id=anex> count extra sounds inside events</label>
 <label style="display:inline-flex;align-items:center;gap:6px">hide below <input type=range id=ansmin min=0 max=30 step=0.5 value=0 style="width:120px"> <span id=ansminv style="min-width:52px">0 %</span></label></span>
 <label style="display:inline-flex;align-items:center;gap:6px" title="show only events whose average level reaches this value">min level <input type=range id=anml min=30 max=90 step=1 value=30 style="width:110px"> <span id=anmlv style="min-width:76px">all levels</span></label>
 <button id=anx style="margin-left:auto">✕</button></div>
<select id=agrp style="font:inherit"><option value=sun>week: Sun</option><option value=mon>week: Mon</option></select>
<div id=anwd class=chips></div>
<div id=ancap style="opacity:.75;margin:6px 0"></div>
<div id=anch></div><div id=anvol style="font-size:12px;opacity:.8;margin:2px 0 6px"></div><div id=anlg></div><div id=antt></div></dialog>
<script>
let cats=[],rows=[],maxId=0,colors={};
const $=s=>document.querySelector(s);
const hsl=(h,s,l)=>{l/=100;const a=s*Math.min(l,1-l)/100,f=n=>{const k=(n+h/30)%12;return Math.round(255*(l-a*Math.max(Math.min(k-3,9-k,1),-1))).toString(16).padStart(2,'0')};return '#'+f(0)+f(8)+f(4)};
const VOL=[45,85],volCol=v=>hsl(Math.max(0,Math.min(120,120-(v-VOL[0])*120/(VOL[1]-VOL[0]))),70,45);
const volSq=r=>`<svg width=22 height=22 style="display:block" shape-rendering=crispEdges><title>avg ${r.laeq.toFixed(0)} dB | peak ${r.peak_db.toFixed(0)} dB</title><rect width=11 height=22 fill="${volCol(r.laeq)}"/><rect x=11 width=11 height=22 fill="${volCol(r.peak_db)}"/><rect x=.5 y=.5 width=21 height=21 fill=none stroke="#8888"/></svg>`;
const colorOf=c=>colors[c]||hsl([...c].reduce((h,ch)=>(h*31+ch.charCodeAt(0))%360,7),55,45);
const txt=h=>{const v=[1,3,5].map(i=>parseInt(h.substr(i,2),16));return (0.299*v[0]+0.587*v[1]+0.114*v[2])>150?'#000':'#fff'};
function paint(){document.querySelectorAll('#tb .tg').forEach(x=>{const c=colorOf(x.dataset.c);x.style.background=c;x.style.color=txt(c)});
 document.querySelectorAll('#tb .tls span').forEach(x=>x.style.background=colorOf(x.dataset.c));
 document.querySelectorAll('#tb tr').forEach(tr=>{const s=tr.querySelector('select.pill');if(!s)return;
 const c=colorOf(s.value);s.style.background=c;s.style.color=txt(c);tr.querySelector('.sw').value=c})}
async function setColor(sw,save){const cat=sw.closest('tr').querySelector('select.pill').value;colors[cat]=sw.value;paint();stats();
 if(save)await fetch('/api/colors',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({category:cat,color:sw.value})})}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const opts=sel=>cats.map(c=>`<option${c===sel?' selected':''}>${esc(c)}</option>`).join('')+'<option value="__new__">+ new…</option>';
const show=r=>(!$('#f').value||r.category===$('#f').value)&&(!$('#u').checked||!r.corrected);
function addCat(c){if(!cats.includes(c)){cats.push(c);$('#f').insertAdjacentHTML('beforeend',`<option>${esc(c)}</option>`)}}
const hornCell=r=>(r.horn?(r.horn.n>0&&r.horn.ok!==false?`<b>${r.horn.n}</b> · ${r.horn.s.toFixed(1)} s${r.horn.p?' ⟳':''}${r.horn.m?' · fixed':''}`:(r.category==='car_horn'?'–':'')):(r.category==='car_horn'?'…':''))+
  ((r.horn||r.category==='car_horn')?` <button class=hfx data-id="${r.id}" title="fix beeps: keep / remove / group by horn, with spectrogram">✎</button>`:'');
const tagsHtml=r=>{const t=(r.labels||[]).filter(l=>l[0]!==r.category).map(l=>`<span class=tg data-c="${esc(l[0])}" title="double-click: make this the category" style="background:${colorOf(l[0])};color:${txt(colorOf(l[0]))}" title="${l[1]!=null?l[1]+'% · ':''}${l[2]} s${l[3].includes('d')?' · confirmed by signal detector':''}">${esc(l[0])} ${l[2]}s${l[3].includes('d')?' 🎯':''}</span>`).join('');
 return t||`<small>${esc(r.raw)}</small>`};
const tlHtml=r=>r.timeline&&r.timeline.r.length?'<div class=tls>'+r.timeline.r.map(([a,b,c])=>`<span data-c="${esc(c)}" title="${esc(c)}" style="left:${a/r.timeline.n*100}%;width:${(b-a+1)/r.timeline.n*100}%;background:${colorOf(c)}"></span>`).join('')+'</div>':'';
async function hornFill(){if(!rows.some(r=>r.category==='car_horn'&&!r.horn))return;
 for(;;){const d=await (await fetch('/api/horns?cat=car_horn&hours=0')).json(),m={};d.rows.forEach(h=>m[h.id]=h);
  rows.forEach(r=>{const h=m[r.id];if(h)r.horn={n:h.n,s:h.horn_s,p:h.periodic,ok:h.trusted!==false}});
  document.querySelectorAll('#tb tr').forEach(tr=>{const r=rows.find(x=>x.id===+tr.dataset.id);if(r)tr.querySelector('.hc').innerHTML=hornCell(r)});
  if(!d.pending)return}}
function rowHtml(r){const d=new Date(r.ts*1000);
 return `<tr data-id="${r.id}" class="${r.corrected?'fixed':''}"><td>${d.toLocaleDateString()} <b>${d.toLocaleTimeString()}</b>
 <td>${r.length.toFixed(1)} s<td>${r.laeq.toFixed(0)} / ${r.peak_db.toFixed(0)} dB<td>${volSq(r)}
 <td><select class=pill data-cur="${esc(r.category)}">${opts(r.category)}</select> <input type=color class=sw title="pill color for this category"><button class=ren title="rename this category everywhere">✎</button><button class=ok title="the label is correct: confirm it (counts as hand-labelled)">✓</button><button class=rs title="reset to the automatic label">↺</button>
 <td class=cf>${r.corrected?'✓ manual':r.pct.toFixed(0)+'%'}<td class=hc>${hornCell(r)}<td title="${esc(r.raw)}">${tagsHtml(r)}
 <td><div class=lis><div class=sp title="spectrogram 60 Hz-8 kHz (log, low at bottom); click to seek"><img loading=lazy src="/spec/${encodeURIComponent(r.file.replace('.wav','.png'))}">${tlHtml(r)}<i class=cur></i></div><audio controls preload=none src="/audio/${encodeURIComponent(r.file)}"></audio></div>
 <td><button class=del title=delete>🗑</button>`}
function stats(){const n={};rows.forEach(r=>n[r.category]=(n[r.category]||0)+1);
 $('#st').innerHTML=Object.entries(n).sort((a,b)=>b[1]-a[1]).map(([c,k])=>`<span data-c="${esc(c)}" style="background:${colorOf(c)};color:${txt(colorOf(c))}${$('#f').value===c?';outline:2px solid currentColor;outline-offset:1px':''}">${esc(c)} ${k}</span>`).join('')}
function render(){$('#fr').style.display=($('#f').value||$('#u').checked)?'':'none';$('#tb').innerHTML=sortRows(rows).filter(show).map(rowHtml).join('');paint();stats()}
let DAYS=[],TODAY='',curDay='',ALL=false,TOT={n:0,s:0,first:0};
async function load(day){                       // (re)load the chosen day; default = newest day with events
 if(day)curDay=day;
 const D=await (await fetch('/api/days')).json();DAYS=D.days;TODAY=D.today;maxId=Math.max(maxId,D.maxId);TOT={n:D.total_n,s:D.total_s,first:D.first_ts};
 if(!curDay)curDay=DAYS.length?DAYS[0].day:TODAY;
 await fetchDay()}
async function fetchDay(){                       // one day, newest first, 500 at a time
 rows=await (await fetch(ALL?'/api/events?limit=500':`/api/events?day=${curDay}&limit=500`)).json();
 rows.forEach(r=>addCat(r.category));maxId=Math.max(maxId,0,...rows.map(r=>r.id));
 render();hornFill();navUpdate()}
async function goDay(d){ALL=false;curDay=d;DAYS=(await (await fetch('/api/days')).json()).days;await fetchDay()}
async function olderDay(){if(!rows.length)return;const oldest=Math.min(...rows.map(r=>r.id));
 const j=await (await fetch((ALL?'/api/events?':`/api/events?day=${curDay}&`)+`before=${oldest}&limit=500`)).json();
 j.forEach(r=>addCat(r.category));rows=rows.concat(j);render();hornFill();navUpdate()}
const dayCount=d=>(DAYS.find(x=>x.day===d)||{n:0}).n;
function fmtDay(s){const[y,m,d]=s.split('-').map(Number);return new Date(y,m-1,d).toLocaleDateString(undefined,{weekday:'short',day:'numeric',month:'short',year:'numeric'})}
function navUpdate(){const cnt=ALL?TOT.n:dayCount(curDay),older=DAYS.find(x=>x.day<curDay),newer=DAYS.filter(x=>x.day>curDay).pop();
 $('#dprev').disabled=ALL||!older;$('#dnext').disabled=ALL||!newer;$('#ddate').value=ALL?'':curDay;
 $('#dtod').style.display=(!ALL&&curDay===TODAY)?'none':'';$('#dall').classList.toggle('on',ALL);
 $('#dcnt').textContent=ALL?`all days · ${rows.length} of ${TOT.n} events`:`${fmtDay(curDay)} · ${rows.length}${cnt>rows.length?' of '+cnt:''} events`;
 const sec=ALL?TOT.s:((DAYS.find(x=>x.day===curDay)||{s:0}).s);
 $('#dsum').textContent=`· sound ${dur(sec)}`+(ALL&&TOT.first?` · recorded since ${new Date(TOT.first*1000).toLocaleDateString()}`:'');
 $('#dmore').style.display=cnt>rows.length?'':'none'}
$('#dall').onclick=()=>{ALL=!ALL;fetchDay()};
$('#dprev').onclick=()=>{const o=DAYS.find(x=>x.day<curDay);if(o)goDay(o.day)};
$('#dnext').onclick=()=>{const n=DAYS.filter(x=>x.day>curDay).pop();if(n)goDay(n.day)};
$('#dtod').onclick=()=>goDay(TODAY);
$('#ddate').onchange=e=>e.target.value&&goDay(e.target.value);
$('#dmore').onclick=olderDay;
async function poll(){                           // live updates only make sense for today (or all days)
 if(!ALL&&(!curDay||curDay!==TODAY))return;
 const n=await (await fetch('/api/events?after='+maxId)).json();if(!n.length)return;
 maxId=Math.max(...n.map(r=>r.id));n.forEach(r=>addCat(r.category));rows=[...n,...rows];TOT.n+=n.length;TOT.s+=n.reduce((t,r)=>t+(r.length||0),0);
 const t=DAYS.find(x=>x.day===TODAY);if(t)t.n+=n.length;else DAYS.unshift({day:TODAY,n:n.length});
 render();navUpdate()}
const MTH=$('.tw thead');
// ordering: click a column name (or its arrow) to sort by it; click again to reverse
const SORT={key:'ts',dir:-1};
const SKEYS=['ts','length','laeq','peak_db','category','pct','beeps','other',null,null];
const SKEY_OF={ts:r=>r.ts,length:r=>r.length,laeq:r=>r.laeq,peak_db:r=>r.peak_db,category:r=>r.category,
 pct:r=>r.pct||0,beeps:r=>r.horn?r.horn.n:-1,other:r=>(r.labels||[]).filter(l=>l[0]!==r.category).length};
function sortRows(arr){const f=SKEY_OF[SORT.key],sec=r=>r.horn?r.horn.s:-1;
 return [...arr].sort((a,b)=>{const x=f(a),y=f(b);let c=x<y?-1:x>y?1:0;
  if(!c&&SORT.key==='beeps')c=sec(a)-sec(b);if(!c)c=a.ts-b.ts;return c*SORT.dir})}
function sortIcons(){const ths=[...MTH.querySelectorAll('th')];
 ths.forEach((th,i)=>{const k=SKEYS[+th.dataset.ci];if(!k)return;let ic=th.querySelector('.sk');
  if(!ic){ic=document.createElement('span');ic.className='sk';th.appendChild(ic)}
  ic.textContent=SORT.key===k?(SORT.dir>0?' ▲':' ▼'):' ⇅';th.style.cursor='pointer';th.style.userSelect='none'})}
MTH.addEventListener('click',e=>{const th=e.target.closest('th');if(!th)return;
 const k=SKEYS[+th.dataset.ci];if(!k)return;
 if(SORT.key===k)SORT.dir*=-1;else{SORT.key=k;SORT.dir=k==='category'?1:-1}
 sortIcons();render()});
// key S: the event you listened to last takes the category you set last (for chains of similar sounds)
let LASTLIS=null,LAST_SET=null;
document.addEventListener('play',e=>{if(e.target.tagName!=='AUDIO')return;const tr=e.target.closest('#tb tr');if(tr)LASTLIS=+tr.dataset.id},true);
function toast(t){let el=$('#toast');if(!el){el=document.createElement('div');el.id='toast';document.body.appendChild(el)}
 el.textContent=t;el.style.opacity=1;clearTimeout(el._t);el._t=setTimeout(()=>el.style.opacity=0,1600)}
document.addEventListener('keydown',async e=>{if(String(e.key).toLowerCase()!=='s'||e.ctrlKey||e.metaKey||e.altKey)return;
 if(e.target.closest&&e.target.closest('input,select,textarea,[contenteditable]'))return;
 if(document.querySelector('dialog[open]'))return;
 if(!LAST_SET){toast('set a category on an event first');return}
 if(!LASTLIS){toast('listen to an event first');return}
 e.preventDefault();const id=LASTLIS,cat=LAST_SET;
 const r=await fetch('/api/events/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({category:cat})});
 if(!r.ok){toast('could not change the category');return}
 addCat(cat);const row=rows.find(x=>x.id===id);if(row){row.category=cat;row.corrected=1;render()}
 toast('event → '+cat)});
// column order: drag a column name to move the column; saved in this browser
[...MTH.querySelectorAll('th')].forEach((th,i)=>{th.dataset.ci=i;th.draggable=true});
const MTH_CANON=MTH.querySelector('tr').cloneNode(true);     // the header in its original order
let COLS=(()=>{try{const a=JSON.parse(localStorage.getItem('ne_colorder')||'null');if(Array.isArray(a)&&a.length===10)return a}catch(e){}return [0,1,2,3,4,5,6,7,8,9]})();
function applyCols(){MTH.querySelector('tr').replaceWith(MTH_CANON.cloneNode(true));sortIcons();
 const trs=[MTH.querySelector('tr'),...document.querySelectorAll('#tb tr')];
 trs.forEach(tr=>{if(!tr)return;const cells=[...tr.children];if(cells.length!==COLS.length)return;COLS.forEach(ci=>tr.appendChild(cells[ci]))})}
let DRAG_FROM=null;
MTH.addEventListener('dragstart',e=>{const th=e.target.closest('th');if(!th)return;DRAG_FROM=COLS.indexOf(+th.dataset.ci);e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('text/plain','col')});
MTH.addEventListener('dragover',e=>{if(e.target.closest('th'))e.preventDefault()});
MTH.addEventListener('drop',e=>{const th=e.target.closest('th');if(!th||DRAG_FROM===null)return;e.preventDefault();
 const to=COLS.indexOf(+th.dataset.ci);if(to!==DRAG_FROM){const [x]=COLS.splice(DRAG_FROM,1);COLS.splice(to,0,x)}
 DRAG_FROM=null;localStorage.setItem('ne_colorder',JSON.stringify(COLS));applyCols();syncHead()});
$('#colr').onclick=()=>{COLS=[0,1,2,3,4,5,6,7,8,9];localStorage.removeItem('ne_colorder');render()};
sortIcons();
$('#tb').addEventListener('input',e=>{if(e.target.classList.contains('sw'))setColor(e.target,false)});
$('#tb').addEventListener('change',async e=>{if(e.target.classList.contains('sw')){setColor(e.target,true);return}
 const s=e.target,tr=s.closest('tr'),id=+tr.dataset.id;let v=s.value;
 if(v==='__new__'){v=(prompt('New category name')||'').trim().toLowerCase().split(' ').join('_');
  if(!v){s.value=s.dataset.cur;return}
  if(!cats.includes(v)){addCat(v);document.querySelectorAll('#tb select').forEach(x=>x.insertAdjacentHTML('afterbegin',`<option>${esc(v)}</option>`))}
  s.value=v}
 const r=await fetch('/api/events/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({category:v})});
 if(!r.ok){alert('save failed');s.value=s.dataset.cur;return}
 LAST_SET=v;s.dataset.cur=v;tr.classList.add('fixed');tr.querySelector('.cf').textContent='✓ manual';
 const o=rows.find(x=>x.id===id);o.category=v;o.corrected=1;paint();stats();tr.querySelector('.hc').innerHTML=hornCell(o);if(v==='car_horn')hornFill()});
$('#tb').addEventListener('click',async e=>{if(!e.target.classList.contains('del')||!confirm('Delete event and its recording?'))return;
 const tr=e.target.closest('tr'),id=+tr.dataset.id;await fetch('/api/events/'+id,{method:'DELETE'});
 tr.remove();rows=rows.filter(x=>x.id!==id);stats()});
$('#st').addEventListener('click',e=>{const c=e.target.dataset.c;if(c){$('#f').value=$('#f').value===c?'':c;render()}});
$('#fr').onclick=()=>{$('#f').value='';$('#u').checked=false;render()};
$('#f').onchange=$('#u').onchange=render;
document.addEventListener('play',e=>document.querySelectorAll('audio').forEach(a=>a!==e.target&&a.pause()),true);
cats=[...%CATS%];cats.forEach(c=>$('#f').insertAdjacentHTML('beforeend',`<option>${esc(c)}</option>`));
let MINL=30;   // minimum average level of events, shared by the analysis and horn windows (30 = all)
const A={mode:'recent',range:48,step:3600,wd:new Set([0,1,2,3,4,5,6]),metric:'n',norm:false,avg:true,off:new Set(),d:null,t:null,period:'all',lv:'avg',vw:'scatter',ex:false,th:65,hm:'avg',ws:'sun',minsh:0};
const dlg=$('#an'),pad=n=>String(n).padStart(2,'0'),free=()=>A.step===0;
const RANGES={recent:[[24,'last 24 hours'],[48,'last 48 hours'],[72,'last 3 days'],[168,'last 7 days'],[336,'last 14 days'],[720,'last 30 days'],[0,'all days']],
 hod:[[0,'all time'],[7,'last 7 days'],[30,'last 30 days'],[90,'last 90 days']],
 src:[[1,'last 24 hours'],[3,'last 3 days'],[7,'last 7 days'],[30,'last 30 days'],[90,'last 90 days'],[0,'all time']]};
RANGES.heat=RANGES.hod;
const dur=s=>s<60?Math.round(s)+' s':s<3600?(s/60).toFixed(s<600?1:0)+' min':(s/3600).toFixed(s<36000?1:0)+' h';
const WDN=[[1,'Mon'],[2,'Tue'],[3,'Wed'],[4,'Thu'],[5,'Fri'],[6,'Sat'],[0,'Sun']];
const nice=v=>{if(v<=0)return 1;const e=Math.pow(10,Math.floor(Math.log10(v))),f=v/e;return([1,1.2,1.5,2,2.5,3,4,5,6,8,10].find(x=>x>=f-1e-9))*e};
const fmt=v=>Number.isInteger(v)?String(v):v>=100?String(Math.round(v)):v>=10?v.toFixed(0):v>=1?v.toFixed(1):v.toFixed(2);
const hm=k=>k.slice(-5),endHM=k=>{const t=(+k.slice(-5,-3)*60+ +k.slice(-2)+(A.step||3600)/60)%1440;return pad(Math.floor(t/60))+':'+pad(t%60)};
// weekdays: single days ordered from the chosen week start, plus the Israeli groups and presets
const DNAME={0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'};
// the groups that belong to the chosen week: Israeli (Sun–Thu, Fri–Sat) or Monday-based (Mon–Fri, Sat–Sun)
const GROUPS={sun:[['Sun–Thu',[0,1,2,3,4]],['Fri–Sat',[5,6]]],mon:[['Mon–Fri',[1,2,3,4,5]],['Sat–Sun',[6,0]]]};
const grps=()=>GROUPS[A.ws];
const dayOrder=()=>A.ws==='mon'?[1,2,3,4,5,6,0]:[0,1,2,3,4,5,6];
function anChips(){
 $('#anwd').innerHTML='<span class=cl>Days:</span> '+dayOrder().map(d=>`<button data-d=${d}>${DNAME[d]}</button>`).join('')+
  '<span class=cl2><span class=cl>Groups:</span> '+grps().map(([t],i)=>`<button data-g=${i}>${t}</button>`).join('')+
  ' <button data-p=all>All days</button></span>';
 document.querySelectorAll('#anwd button[data-d]').forEach(b=>b.classList.toggle('on',A.wd.has(+b.dataset.d)));
 document.querySelectorAll('#anwd button[data-g]').forEach(b=>b.classList.toggle('on',grps()[+b.dataset.g][1].every(d=>A.wd.has(d))))}
function anControls(){const h=A.mode==='hod',sr=A.mode==='src',ht=A.mode==='heat',f=free()&&!sr&&!ht;
 if(A.step===86400&&A.mode!=='recent')A.step=3600;
 $('#ans option[value="86400"]').disabled=A.mode!=='recent';
 anChips();$('#agrp').style.display=A.mode==='recent'?'none':'';
 $('#anr').innerHTML=RANGES[A.mode].map(([v,t])=>`<option value=${v}${v==A.range?' selected':''}>${t}</option>`).join('');
 $('#ans').value=A.step;$('#ans').style.display=sr?'none':'';
 $('#anwd').style.display=(h||sr)?'':'none';$('#anavl').style.display=(h&&!f)?'':'none';
 $('#ann').parentNode.style.display=$('#anm').style.display=(f||sr||ht)?'none':'';$('#anheat').style.display=ht?'':'none';$('#anhm').value=A.hm;$('#ans option[value="0"]').disabled=ht;
 $('#ansrc').style.display=sr?'':'none';$('#anvol').style.display=sr?'none':'';
 $('#anth').style.display=(sr&&A.vw==='thr')?'':'none';$('#anth').value=A.th;$('#anpe').value=A.period;$('#ansmin').value=A.minsh;$('#ansminv').textContent=A.minsh+' %';$('#anlv').value=A.lv;$('#anvw').value=A.vw;$('#anex').checked=A.ex;
 document.querySelectorAll('#anmode button').forEach(b=>b.classList.toggle('on',b.dataset.m===A.mode));
 }
async function anLoad(){if(A.mode==='recent'&&A.step===900&&A.range>168)A.range=168;if(A.mode==='recent'&&A.step!==86400&&A.range===0)A.range=48;if(A.mode==='heat'&&A.step===0)A.step=3600;anControls();const wd=[...A.wd].join(',');
 const q=A.mode==='heat'?`/api/heat?days=${A.range}&step=${A.step}`:A.mode==='src'?`/api/sources?days=${A.range}&wd=${wd}&period=${A.period}&ex=${A.ex?1:0}`:
  A.mode==='recent'?`/api/stats?mode=recent&hours=${A.range}&step=${A.step}`:`/api/stats?mode=hod&days=${A.range}&wd=${wd}&step=${A.step}`;
 A.d=await (await fetch(q+(q.includes('?')?'&':'?')+'minl='+MINL)).json();anDraw()}
const wdText=()=>A.wd.size===7?'all weekdays':WDN.filter(x=>A.wd.has(x[0])).map(x=>x[1]).join(', ')||'no days selected';
function anLegend(all,tot){$('#anlg').innerHTML=all.map(c=>`<span data-c="${esc(c)}" class="${A.off.has(c)?'off':''}" style="background:${colorOf(c)};color:${txt(colorOf(c))}">${esc(c)} ${fmt(+tot[c].toFixed(2))}</span>`).join('')||'<i>no events in this range</i>'}
function drawHeat(d){const keys=d.keys,nc=keys.length,tot={};
 // rows: all days, then each group of the chosen week (Sun–Thu / Fri–Sat, or single weekdays Mon..Sun)
 const ROWS=[{name:'all days',days:[0,1,2,3,4,5,6]}].concat(grps().map(([t,ds])=>({name:t,days:ds})),dayOrder().map(d=>({name:DNAME[d],days:[d]})));
 d.cells.forEach(c=>tot[c[2]]=(tot[c[2]]||0)+c[4]/60);
 const all=Object.keys(tot).sort((a,b)=>tot[b]-tot[a]),G=ROWS.map(()=>keys.map(()=>({n:0,s:0,pw:0,pk:-Infinity})));
 d.cells.forEach(([w,i,c,n,s,pw,pk])=>{if(A.off.has(c))return;
  ROWS.forEach((r,ri)=>{if(!r.days.includes(w))return;const x=G[ri][i];x.n+=n;x.s+=s;x.pw+=pw;x.pk=Math.max(x.pk,pk)})});
 const nd=ri=>Math.max(ROWS[ri].days.reduce((t,dd)=>t+(d.wdays[dd]||0),0),1),
  val=(ri,i)=>{const x=G[ri][i];if(!x.n)return null;if(A.hm==='avg')return x.s>0?10*Math.log10(x.pw/x.s):null;if(A.hm==='pk')return x.pk;return A.hm==='s'?x.s/60/nd(ri):x.n/nd(ri)};
 let vmax=0;ROWS.forEach((_,ri)=>keys.forEach((_,i)=>{const v=val(ri,i);if(v!==null&&v>vmax)vmax=v}));
 const lvl=A.hm==='avg'||A.hm==='pk',seq=f=>hsl(215,75,Math.round(93-55*f)),color=v=>lvl?volCol(v):seq(Math.min(v/(vmax||1),1));
 const GN=grps().length,W=Math.max($('#anch').clientWidth,360),L=84,R=60,T=4,ch=30,gap=8,cw=(W-L-R)/nc;
 const ys=[];let yy=T;ROWS.forEach((r,ri)=>{if(ri===1||ri===1+GN)yy+=gap;else if(ri>1&&ri<1+GN)yy+=gap/2;ys.push(yy);yy+=ch});
 const H=yy+26,rowN=ROWS.map((_,ri)=>G[ri].reduce((t,x)=>t+x.n,0));
 let s=`<svg width=${W} height=${H} style="display:block">`;
 ROWS.forEach((r,ri)=>{const y=ys[ri];
  s+=`<text x=${L-8} y=${y+ch/2+3} text-anchor=end font-size=12 font-weight=${ri?400:700} fill=currentColor>${r.name}</text>`+
     `<text x=${W-4} y=${y+ch/2+3} text-anchor=end font-size=12 fill=currentColor opacity=.8>${rowN[ri]} ev</text>`;
  keys.forEach((_,i)=>{const v=val(ri,i);
   s+=`<rect data-h="${ri}|${i}" x=${(L+i*cw).toFixed(1)} y=${y} width=${Math.max(cw-(cw>6?1:0),1).toFixed(1)} height=${ch-2} fill="${v===null?'#8881':color(v)}"/>`})});
 const hpb=nc/24,step=[1,2,3,6].find(x=>x*hpb*cw>=26)||6;
 keys.forEach((k,i)=>{if(i%hpb||(i/hpb)%step)return;s+=`<text x=${(L+i*cw+hpb*cw/2).toFixed(1)} y=${H-8} text-anchor=middle font-size=11 fill=currentColor opacity=.75>${k.slice(0,2)}</text>`});
 $('#anch').innerHTML=s+'</svg>';A.t={heat:true,G,keys,ROWS};anLegend(all,tot);
 const rl=(RANGES.hod.find(r=>r[0]==A.range)||[0,''])[1],
  mt={avg:'average level of the events (time-weighted LAeq)',pk:'loudest peak',s:'minutes of sound per day (average over the days that have events)',n:'events per day (average over the days that have events)'}[A.hm];
 $('#ancap').textContent=`${mt}, by weekday and time of day (${A.step===900?'15-minute':'hourly'} cells) · ${rl}. Grey cell = no events.`;
 $('#anvol').style.paddingLeft='84px';$('#anvol').innerHTML=lvl?`colours: <i style="display:inline-block;vertical-align:middle;width:120px;height:8px;background:linear-gradient(90deg,${[0,.25,.5,.75,1].map(f=>volCol(VOL[0]+f*(VOL[1]-VOL[0]))).join(',')})"></i> ${VOL[0]} → ${VOL[1]} dB`:
  `colours: <i style="display:inline-block;vertical-align:middle;width:120px;height:8px;background:linear-gradient(90deg,${[0,.5,1].map(seq).join(',')})"></i> 0 → ${fmt(+vmax.toFixed(1))} ${A.hm==='s'?'min':'events'} per day`}
function drawSrc(d){const all=(d.cats||[]).map(c=>c.c),tot={};(d.cats||[]).forEach(c=>tot[c.c]=c.share*100);
 const C=(d.cats||[]).filter(c=>!A.off.has(c.c)&&c.share*100>=A.minsh),hid=(d.cats||[]).length-C.length,W=Math.max($('#anch').clientWidth,360),lv=c=>A.lv==='avg'?c.avg:c.pk;
 const rl=(RANGES.src.find(r=>r[0]==A.range)||[0,''])[1],per={all:'all day',day:'day 07–19',evening:'evening 19–23',night:'night 23–07'}[A.period];
 if(!C.length){$('#anch').innerHTML='<p style="opacity:.7">no events in this selection</p>';A.t=null;anLegend(all,tot);$('#ancap').textContent='';return}
 let s;
 if(A.vw==='scatter'){const H=380,L=58,R=18,T=12,B=44,pw=W-L-R,ph=H-T-B;
  const lx=Math.log10(Math.max(Math.min(...C.map(c=>c.s))/2.5,0.5)),hx=Math.log10(Math.max(...C.map(c=>c.s))*2.5),ys=C.map(lv),
   y0=Math.floor((Math.min(...ys)-4)/5)*5,y1=Math.ceil((Math.max(...ys)+4)/5)*5,xp=v=>L+(Math.log10(v)-lx)/(hx-lx)*pw,yp=v=>T+ph-(v-y0)/(y1-y0)*ph;
  s=`<svg width=${W} height=${H} style="display:block"><clipPath id=cpl><rect x=${L} y=${T} width=${pw} height=${ph}/></clipPath>`;
  for(let v=y0;v<=y1;v+=5)s+=`<line x1=${L} x2=${W-R} y1=${yp(v)} y2=${yp(v)} stroke="#8884"/><text x=${L-6} y=${yp(v)+4} text-anchor=end font-size=11 fill=currentColor opacity=.7>${v}</text>`;
  [1,5,10,30,60,300,600,1800,3600,7200,18000,36000,86400,259200,604800].forEach(v=>{if(v<Math.pow(10,lx)||v>Math.pow(10,hx))return;
   s+=`<line x1=${xp(v)} x2=${xp(v)} y1=${T} y2=${T+ph} stroke="#8883"/><text x=${xp(v)} y=${H-26} text-anchor=middle font-size=11 fill=currentColor opacity=.75>${dur(v)}</text>`});
  [50,60,70,80].forEach(L0=>{const f=x=>L0+10*Math.log10(3600/x),xa=Math.pow(10,lx),xb=Math.pow(10,hx),xl=Math.pow(10,hx-(hx-lx)*0.03),yl=yp(f(xl));
   s+=`<g clip-path="url(#cpl)"><line x1=${xp(xa)} y1=${yp(f(xa))} x2=${xp(xb)} y2=${yp(f(xb))} stroke=currentColor stroke-opacity=.4 stroke-dasharray="5 4"/></g>`;
   if(yl>T+10&&yl<T+ph-4)s+=`<text x=${xp(xl)} y=${yl-4} text-anchor=end font-size=10 fill=currentColor opacity=.55>= ${L0} dB for 1 h</text>`});
  [...C].sort((a,b)=>b.share-a.share).forEach(c=>{const r=6+34*Math.sqrt(c.share);
   s+=`<circle data-s=${d.cats.indexOf(c)} cx=${xp(c.s).toFixed(1)} cy=${yp(lv(c)).toFixed(1)} r=${r.toFixed(1)} fill="${colorOf(c.c)}" fill-opacity=.6 stroke="${colorOf(c.c)}" stroke-width=1.5 />`});
  C.forEach(c=>{const r=6+34*Math.sqrt(c.share),x=xp(c.s),right=x+r+90<W-R;
   s+=`<text x=${(right?x+r+4:x-r-4).toFixed(1)} y=${(yp(lv(c))+4).toFixed(1)} text-anchor=${right?'start':'end'} font-size=12 fill=currentColor pointer-events=none>${esc(c.c)} ${(c.share*100).toFixed(0)}%</text>`});
  s+=`<text x=${L+pw/2} y=${H-6} text-anchor=middle font-size=12 fill=currentColor opacity=.8>total time of the sound (log scale)</text><text transform="translate(14 ${T+ph/2}) rotate(-90)" text-anchor=middle font-size=12 fill=currentColor opacity=.8>${A.lv==='avg'?'average':'peak'} level, dB(A)</text></svg>`;
  $('#ancap').textContent=`each bubble is a sound category: further right = more total time, higher = louder; bubble size = share of the noise energy (loud × long). Dashed lines = equal energy. ${rl} · ${per} · ${wdText()}`}
 else if(A.vw==='bars'){const rowH=38,L=140,Rt=250,H=C.length*rowH+8,bw=Math.max(W-L-Rt-130,80),mxs=Math.max(...C.map(c=>Math.max(c.share,c.tshare)));
  s=`<svg width=${W} height=${H} style="display:block">`;
  C.forEach((c,i)=>{const y=4+i*rowH,e=Math.max(c.share/mxs*bw,1),t=Math.max(c.tshare/mxs*bw,1);
   s+=`<g data-s=${d.cats.indexOf(c)}><rect x=0 y=${y} width=${W} height=${rowH-2} fill=transparent /><text x=${L-8} y=${y+16} text-anchor=end font-size=13 font-weight=600 fill="${colorOf(c.c)}">${esc(c.c)}</text>
   <rect x=${L} y=${y+3} width=${e} height=13 fill="${colorOf(c.c)}"/><text x=${L+e+6} y=${y+14} font-size=11 fill=currentColor>noise energy ${(c.share*100).toFixed(1)}%</text>
   <rect x=${L} y=${y+19} width=${t} height=8 fill="#8888"/><text x=${L+t+6} y=${y+27} font-size=10 fill=currentColor opacity=.7>time ${(c.tshare*100).toFixed(1)}%</text>
   <text x=${Math.max(W-Rt,L+bw+135)} y=${y+22} font-size=12 fill=currentColor opacity=.85>${dur(c.s)} · ${c.n} ev · avg ${c.avg.toFixed(0)} / peak ${c.pk.toFixed(0)} dB</text></g>`});
  s+='</svg>';$('#ancap').textContent=`coloured bar = share of the noise energy, grey bar = share of the time; a long colour bar over a short grey bar = loud but brief. ${rl} · ${per} · ${wdText()}`}
 else if(A.vw==='thr'){const i=Math.max(d.thr.indexOf(A.th),0),key=A.lv==='avg'?'avg':'pk',rowH=34,L=140,Rt=250,bw=Math.max(W-L-Rt-90,80);
  const R=C.map(c=>({c,sec:c.th[key+'_s'][i],n:c.th[key+'_n'][i]})).sort((a,b)=>b.sec-a.sec),mx=Math.max(...R.map(r=>r.sec),1);
  s=`<svg width=${W} height=${R.length*rowH+8} style="display:block">`;
  R.forEach((r,k)=>{const y=4+k*rowH,len=r.sec>0?Math.max(r.sec/mx*bw,2):0;
   s+=`<g data-s=${d.cats.indexOf(r.c)}><rect x=0 y=${y} width=${W} height=${rowH-2} fill=transparent /><text x=${L-8} y=${y+19} text-anchor=end font-size=13 font-weight=600 fill="${colorOf(r.c.c)}">${esc(r.c.c)}</text>
   <rect x=${L} y=${y+6} width=${len} height=16 fill="${colorOf(r.c.c)}"/><text x=${L+len+6} y=${y+19} font-size=12 fill=currentColor>${r.sec?dur(r.sec):'none'}</text>
   <text x=${Math.max(W-Rt,L+bw+90)} y=${y+19} font-size=12 fill=currentColor opacity=.8>${r.n} events · ${(r.sec/r.c.s*100).toFixed(0)}% of its time</text></g>`});
  s+='</svg>';$('#ancap').textContent=`sound time in events whose ${A.lv==='avg'?'average':'peak'} level reached at least ${d.thr[i]} dB(A), per category. ${rl} · ${per} · ${wdText()}`}
 else{const rowH=40,L=140,Rt=280,lo=Math.floor((Math.min(...C.map(c=>c.l90))-3)/5)*5,hi=Math.ceil((Math.max(...C.map(c=>c.pk))+2)/5)*5,bw=Math.max(W-L-Rt-20,120),xp=v=>L+(v-lo)/(hi-lo)*bw,hh=C.length*rowH;
  s=`<svg width=${W} height=${hh+30} style="display:block">`;
  for(let v=lo;v<=hi;v+=5)s+=`<line x1=${xp(v)} x2=${xp(v)} y1=0 y2=${hh+4} stroke="#8883"/><text x=${xp(v)} y=${hh+20} text-anchor=middle font-size=11 fill=currentColor opacity=.7>${v}</text>`;
  C.forEach((c,k)=>{const y=4+k*rowH,m=y+16,col=colorOf(c.c);
   s+=`<g data-s=${d.cats.indexOf(c)}><rect x=0 y=${y} width=${W} height=${rowH-2} fill=transparent /><text x=${L-8} y=${m+4} text-anchor=end font-size=13 font-weight=600 fill="${col}">${esc(c.c)}</text>
   <line x1=${xp(c.l90)} x2=${xp(c.pk)} y1=${m} y2=${m} stroke="${col}" stroke-width=2 opacity=.6 /><line x1=${xp(c.pk)} x2=${xp(c.pk)} y1=${m-6} y2=${m+6} stroke="${col}" stroke-width=2 />
   <rect x=${xp(c.l90)} y=${m-9} width=${Math.max(xp(c.l10)-xp(c.l90),2)} height=18 fill="${col}" fill-opacity=.55 stroke="${col}" />
   <line x1=${xp(c.l50)} x2=${xp(c.l50)} y1=${m-9} y2=${m+9} stroke=currentColor stroke-width=2 />
   <path d="M${xp(c.avg)} ${m-6} l5 6 l-5 6 l-5 -6 z" fill=currentColor />
   <text x=${L+bw+14} y=${m+4} font-size=11 fill=currentColor opacity=.85>L10 ${c.l10.toFixed(0)} · L50 ${c.l50.toFixed(0)} · L90 ${c.l90.toFixed(0)} · Leq ${c.avg.toFixed(0)} · max ${c.pk.toFixed(0)}</text></g>`});
  s+='</svg>';$('#ancap').textContent=`box = L90…L10 of the event levels (weighted by sound time), black tick = median L50, diamond = LAeq, end of the line = loudest peak. These are event averages, so the quiet gaps between events are not included. ${rl} · ${per} · ${wdText()}`}
 $('#anch').innerHTML=s;A.t={src:true,cats:d.cats};anLegend(all,tot);
 if(hid)$('#ancap').textContent+=` · ${hid} small contributor${hid>1?'s':''} hidden (below ${A.minsh} % of noise energy)`}
function drawFree(d){const P=d.points||[],hod=A.mode==='hod',tot={};P.forEach(p=>tot[p.c]=(tot[p.c]||0)+1);
 const all=Object.keys(tot).sort((a,b)=>tot[b]-tot[a]),act=all.filter(c=>!A.off.has(c)),pts=P.filter(p=>!A.off.has(p.c));
 const W=Math.max($('#anch').clientWidth,360),L=120,R=10,T=8,B=30,nl=Math.max(act.length,1),lh=Math.max(28,Math.min(64,320/nl)),H=T+B+lh*nl,pw=W-L-R;
 const tz=hod?0:d.tz,x0=hod?0:d.t0,x1=hod?86400:d.t1,xp=p=>L+((hod?p.tod:p.ts)-x0)/(x1-x0)*pw,lane={};act.forEach((c,i)=>lane[c]=i);
 let s=`<svg width=${W} height=${H+24} style="display:block">`;
 act.forEach((c,i)=>{const y=T+i*lh;s+=`<rect x=${L} y=${y} width=${pw} height=${lh} fill="${i%2?'#8881':'none'}"/><text x=${L-8} y=${y+lh/2+4} text-anchor=end font-size=12 font-weight=600 fill="${colorOf(c)}">${esc(c)}</text>`});
 const cand=hod?[3600,7200,14400]:[3600,7200,10800,21600,43200,86400,172800],st=cand.find(v=>pw*v/(x1-x0)>=60)||cand[cand.length-1];
 for(let t=Math.ceil((x0+tz)/st)*st-tz;t<=x1;t+=st){const x=L+(t-x0)/(x1-x0)*pw,dt=new Date((t+tz)*1000),h=dt.getUTCHours(),
   lab=hod||(st<86400&&h!==0)?pad(h)+':00':pad(dt.getUTCMonth()+1)+'-'+pad(dt.getUTCDate());
  s+=`<line x1=${x} x2=${x} y1=${T} y2=${H-B} stroke="#8883"/><text x=${x} y=${H-12} text-anchor=middle font-size=11 fill=currentColor opacity=.75>${lab}</text>`}
 pts.forEach((p,i)=>{const jy=((((p.id*2654435761)>>>0)%1000)/1000-.5)*(lh-14),r=3+Math.min(p.l,20)/20*4;
  s+=`<circle data-p=${i} cx=${xp(p).toFixed(1)} cy=${(T+lane[p.c]*lh+lh/2+jy).toFixed(1)} r=${r.toFixed(1)} fill="${colorOf(p.c)}" fill-opacity=.7 stroke="#0007" stroke-width=.6 />`});
 s+=`<text x=${L-5} y=${H+11} text-anchor=end font-size=10 fill=currentColor opacity=.7>peak</text><text x=${L-5} y=${H+21} text-anchor=end font-size=10 fill=currentColor opacity=.7>avg</text>`;
 pts.forEach(p=>{const x=xp(p)-1.5;s+=`<rect x=${x.toFixed(1)} y=${H+3} width=3 height=9 fill="${volCol(p.pk)}" fill-opacity=".9"/><rect x=${x.toFixed(1)} y=${H+13} width=3 height=9 fill="${volCol(p.a)}" fill-opacity=".9"/>`});
 $('#anch').innerHTML=s+'</svg>';A.t={free:true,pts};anLegend(all,tot);
 const rl=(RANGES[A.mode].find(r=>r[0]==A.range)||[0,''])[1];
 $('#ancap').textContent=`each dot = one event, ${rl}${hod?', all days overlaid by time of day · '+wdText():''} · jittered vertically inside its category row · dot size = event length`+(d.truncated?' · showing newest 5000':'')}
function anDraw(){const d=A.d;if(!d)return;$('#anvol').style.paddingLeft='';$('#anvol').innerHTML=A.mode==='heat'?'':VOLHTML;if(A.mode==='heat'){drawHeat(d);return}if(A.mode==='src'){drawSrc(d);return}if(free()){drawFree(d);return}
 const keys=d.keys,n=keys.length,ix={};keys.forEach((k,i)=>ix[k]=i);
 const per=(A.mode==='hod'&&A.avg&&!A.norm)?Math.max(d.ndays,1):1,vals={},tot={};
 const PK={},PW={},NN={};
 d.rows.forEach(r=>{const i=ix[r.b];if(i===undefined)return;const v=(A.metric==='n'?r.n:r.s/60)/per;
  (vals[r.c]=vals[r.c]||new Array(n).fill(0))[i]+=v;tot[r.c]=(tot[r.c]||0)+v;
  (PK[r.c]=PK[r.c]||new Array(n).fill(-Infinity))[i]=Math.max(PK[r.c][i],r.pk);(PW[r.c]=PW[r.c]||new Array(n).fill(0))[i]+=r.pw;(NN[r.c]=NN[r.c]||new Array(n).fill(0))[i]+=r.n});
 const all=Object.keys(vals).sort((a,b)=>tot[b]-tot[a]),act=all.filter(c=>!A.off.has(c));
 const col=keys.map((_,i)=>act.reduce((s,c)=>s+vals[c][i],0));
 const mx=A.norm?100:nice(Math.max(...col,0)),yv=(c,i)=>A.norm?(col[i]?vals[c][i]/col[i]*100:0):vals[c][i];
 const W=Math.max($('#anch').clientWidth,320),H=300,L=46,R=8,T=8,B=30,pw=W-L-R,ph=H-T-B,bw=pw/n,g=bw>6?bw*0.15:0;
 let s=`<svg width=${W} height=${H+24} style="display:block">`;
 for(let t=0;t<=4;t++){const y=T+ph-ph*t/4;
  s+=`<line x1=${L} x2=${W-R} y1=${y} y2=${y} stroke="#8884"/><text x=${L-5} y=${y+4} text-anchor=end font-size=11 fill=currentColor opacity=.7>${fmt(mx*t/4)}${A.norm?'%':''}</text>`}
 keys.forEach((k,i)=>{let y=T+ph;const x=L+i*bw+g/2,w=Math.max(bw-g,1);
  act.forEach(c=>{const v=yv(c,i);if(v<=0)return;const h=ph*v/mx;y-=h;s+=`<rect x=${x} y=${y} width=${w} height=${h} fill="${colorOf(c)}"/>`});
  s+=`<rect data-i=${i} x=${L+i*bw} y=${T} width=${bw} height=${H+24-T} fill=transparent />`});
 const DAYS1=A.step===86400&&A.mode==='recent',hpb=3600/A.step,need=Math.ceil(36/bw),hs=[1,2,3,6,12,24].find(x=>x*hpb>=need)||24,dn=Math.ceil(44/bw);
 keys.forEach((k,i)=>{if(DAYS1){if(i%dn)return;s+=`<text x=${L+i*bw+bw/2} y=${H-12} text-anchor=middle font-size=11 fill=currentColor opacity=.75>${k.slice(5)}</text>`;return}
  const h=+k.slice(-5,-3),m=+k.slice(-2);if(m||h%(A.mode==='hod'?Math.min(hs,6):hs))return;
  const lab=A.mode==='hod'?pad(h):(h===0?k.slice(5,10):h);
  s+=`<text x=${L+i*bw+bw/2} y=${H-12} text-anchor=middle font-size=11 fill=currentColor opacity=.75>${lab}</text>`});
 const vol=keys.map((_,i)=>{let c0=0,pw=0,pk=-Infinity;act.forEach(c=>{if(NN[c]&&NN[c][i]){c0+=NN[c][i];pw+=PW[c][i];pk=Math.max(pk,PK[c][i])}});return c0?{avg:10*Math.log10(pw/c0),pk}:null});
 s+=`<text x=${L-5} y=${H+11} text-anchor=end font-size=10 fill=currentColor opacity=.7>peak</text><text x=${L-5} y=${H+21} text-anchor=end font-size=10 fill=currentColor opacity=.7>avg</text>`;
 vol.forEach((v,i)=>{if(!v)return;const x=L+i*bw+g/2,w=Math.max(bw-g,1);
  s+=`<rect data-vi=${i} x=${x} y=${H+3} width=${w} height=9 fill="${volCol(v.pk)}"/><rect data-vi=${i} x=${x} y=${H+13} width=${w} height=9 fill="${volCol(v.avg)}"/>`});
 $('#anch').innerHTML=s+'</svg>';A.t={keys,vals,act,col,vol};anLegend(all,tot);
 const mn=A.metric==='n'?'events':'minutes of sound',rl=(RANGES[A.mode].find(r=>r[0]==A.range)||[0,''])[1],bk=A.step===900?'15 minutes':A.step===86400?'day':'hour';
 $('#ancap').textContent=A.mode==='recent'?`${mn} per ${bk}, ${rl}`:
  `${A.norm?'category mix (%) ':(per>1||A.avg?'average '+mn+' per day ':'total '+mn+' ')}by time of day (${A.step===900?'15-minute':'hourly'} buckets) · ${rl} · ${wdText()}`+(d.ndays?` (${d.ndays} days with events)`:'')}
const VOLHTML=`volume strips under the axis: top = peak, bottom = average &nbsp;<i style="display:inline-block;vertical-align:middle;width:120px;height:8px;background:linear-gradient(90deg,${[0,.25,.5,.75,1].map(f=>volCol(VOL[0]+f*(VOL[1]-VOL[0]))).join(',')})"></i> ${VOL[0]} → ${VOL[1]} dB`;
$('#anvol').innerHTML=VOLHTML;
$('#anb').onclick=()=>{dlg.showModal();anControls();anLoad()};
function setMinl(v){MINL=+v;['anml','hnml'].forEach(i=>{const el=$('#'+i);if(el)el.value=MINL});
 ['anmlv','hnmlv'].forEach(i=>{const el=$('#'+i);if(el)el.textContent=MINL>30?'≥ '+MINL+' dB':'all levels'})}
['anml','hnml'].forEach(i=>$('#'+i).addEventListener('input',e=>{setMinl(e.target.value);if(dlg.open)anLoad();if(hdlg.open)hnLoad()}));
$('#anx').onclick=()=>dlg.close();
dlg.addEventListener('click',e=>{if(e.target===dlg)dlg.close()});
$('#anmode').addEventListener('click',e=>{const m=e.target.dataset.m;if(!m)return;A.mode=m;A.range=m==='recent'?48:m==='src'?7:0;anLoad()});
$('#anr').onchange=e=>{A.range=+e.target.value;if(A.mode==='recent'&&A.range===0)A.step=86400;anLoad()};
$('#ans').onchange=e=>{A.step=+e.target.value;anLoad()};
$('#anm').onchange=e=>{A.metric=e.target.value;anDraw()};
$('#anhm').onchange=e=>{A.hm=e.target.value;anDraw()};
$('#ansmin').oninput=e=>{A.minsh=+e.target.value;$('#ansminv').textContent=A.minsh+' %';anDraw()};
$('#anpe').onchange=e=>{A.period=e.target.value;anLoad()};
$('#anex').onchange=e=>{A.ex=e.target.checked;anLoad()};
$('#anlv').onchange=e=>{A.lv=e.target.value;anDraw()};
$('#anvw').onchange=e=>{A.vw=e.target.value;anControls();anDraw()};
$('#anth').onchange=e=>{A.th=+e.target.value;anDraw()};
$('#ann').onchange=e=>{A.norm=e.target.checked;anDraw()};
$('#anav').onchange=e=>{A.avg=e.target.checked;anDraw()};
$('#anwd').addEventListener('click',e=>{const b=e.target;
 if(b.dataset.d!==undefined){const d=+b.dataset.d;A.wd.has(d)?A.wd.delete(d):A.wd.add(d)}
 else if(b.dataset.g!==undefined){const ds=grps()[+b.dataset.g][1],on=ds.every(d=>A.wd.has(d));ds.forEach(d=>on?A.wd.delete(d):A.wd.add(d))}
 else if(b.dataset.p)A.wd=new Set([0,1,2,3,4,5,6]);else return;
 anLoad()});
$('#agrp').onchange=e=>{A.ws=e.target.value;anChips();anDraw()};
$('#anlg').addEventListener('click',e=>{const c=e.target.dataset.c;if(!c)return;A.off.has(c)?A.off.delete(c):A.off.add(c);anDraw()});
$('#anch').addEventListener('mousemove',e=>{const r=e.target.closest('[data-i],[data-p],[data-s],[data-h],[data-vi]'),tt=$('#antt');if(!r||!A.t){tt.style.display='none';return}
 if(r.dataset.vi!==undefined){const i=+r.dataset.vi,v=A.t.vol&&A.t.vol[i];if(!v)return;const lab=(A.mode==='hod'?'':A.t.keys[i].slice(0,10)+' ')+hm(A.t.keys[i])+'–'+endHM(A.t.keys[i]);
  tt.innerHTML=`<b>${lab}</b><div>average <b>${v.avg.toFixed(1)} dB</b></div><div>peak <b>${v.pk.toFixed(0)} dB</b></div>`}
 else if(r.dataset.h!==undefined){const [k,i]=r.dataset.h.split('|'),g=A.t.G[+k][+i],nm=A.t.ROWS[+k].name,ke=A.t.keys[+i];
  tt.innerHTML=`<b>${nm} ${ke}–${endHM(ke)}</b>`+(g.n?`<div>${g.n} events · ${dur(g.s)} of sound</div><div>average <b>${(g.s>0?10*Math.log10(g.pw/g.s):0).toFixed(0)}</b> · peak <b>${g.pk.toFixed(0)}</b> dB</div>`:'<div>no events</div>')}
 else if(r.dataset.s!==undefined){const c=A.t.cats[+r.dataset.s];
  tt.innerHTML=`<b style="color:${colorOf(c.c)}">${esc(c.c)}</b><div>${dur(c.s)} of sound · ${c.n} events</div><div>average ${c.avg.toFixed(1)} dB(A) · peak ${c.pk.toFixed(0)} dB</div><div>noise energy <b>${(c.share*100).toFixed(1)}%</b> · time ${(c.tshare*100).toFixed(1)}%</div><div>L10 ${c.l10.toFixed(0)} · L50 ${c.l50.toFixed(0)} · L90 ${c.l90.toFixed(0)} dB</div>`}
 else if(r.dataset.p!==undefined){const p=A.t.pts[+r.dataset.p],dt=new Date((p.ts+A.d.tz)*1000).toISOString().replace('T',' ').slice(0,19);
  tt.innerHTML=`<b>${dt}</b><div><i style="background:${colorOf(p.c)}"></i>${esc(p.c)}</div><div>${p.l.toFixed(1)} s · avg ${p.a.toFixed(0)} / peak ${p.pk.toFixed(0)} dB</div>`}
 else{const i=+r.dataset.i,{keys,vals,act,col,vol}=A.t,lab=A.step===86400&&A.mode==='recent'?keys[i]:(A.mode==='hod'?'':keys[i].slice(0,10)+' ')+hm(keys[i])+'–'+endHM(keys[i]);
  tt.innerHTML=`<b>${lab}</b>`+act.filter(c=>vals[c][i]>0).sort((a,b)=>vals[b][i]-vals[a][i]).map(c=>`<div><i style="background:${colorOf(c)}"></i>${esc(c)} <b>${fmt(+vals[c][i].toFixed(2))}</b>${A.norm&&col[i]?' ('+Math.round(vals[c][i]/col[i]*100)+'%)':''}</div>`).join('')+`<div>total <b>${fmt(+col[i].toFixed(2))}</b></div>`+(vol[i]?`<div>avg <b>${vol[i].avg.toFixed(0)}</b> · peak <b>${vol[i].pk.toFixed(0)}</b> dB</div>`:'')}
 tt.style.display='block';tt.style.left=Math.min(e.clientX+14,innerWidth-230)+'px';tt.style.top=(e.clientY+14)+'px'});
$('#anch').addEventListener('mouseleave',()=>$('#antt').style.display='none');
addEventListener('resize',()=>{if(dlg.open)anDraw()});
setInterval(()=>{if(dlg.open)anLoad()},60000);
$('#inb').onclick=async()=>{const i=await (await fetch('/api/info')).json(),l=i.learn;
 $('#infb').innerHTML=(i.recording?'':'<p><i>Web-only mode: the numbers below are defaults and may differ from the running recorder.</i></p>')+`
 <p><b>Listening.</b> The microphone (device ${i.device}) is listened to continuously in blocks of ${i.block_s} s. While it is quiet nothing is stored.</p>
 <p><b>Level.</b> Each block gets an A-weighted level: dB(A) = dBFS(A) + ${i.offset} dB (the calibration offset). The background level is a running average of quiet blocks (it follows changes in ~20 s, and only slowly, ~5 min, while an event is running).</p>
 <p><b>Event start / end.</b> An event starts at a block louder than background + ${i.delta} dB and goes on while blocks stay louder than background + ${(i.delta*0.6).toFixed(1)} dB. It ends after ${i.hold} s without such blocks, so a sound that repeats within ${i.hold} s (honks, a truck that pauses) is one event.</p>
 <p><b>Time limits.</b> An event is cut after ${i.max_len} s; a longer sound becomes several consecutive events. It is also split earlier when a different dominant sound (say voices after a truck) lasts ${i.split} s or more. The shortest event is one block (≈1 s). The saved file also holds ~1 s before and ~1 s after, so up to about ${Math.round(i.max_len)+3} s. The <i>length</i> column counts from the first to the last loud block (the ${i.hold} s wait is not included).</p>
 <p><b>Classification.</b> YAMNet runs only on the loud blocks. Each block gets a score per category (the best class of the group) and a dominant sound. The <i>category</i> is the dominant sound of the event (most blocks); the strip at the bottom of the spectrogram shows the dominant sound second by second. Any other sound that reaches 25% in at least one block is listed under <i>other sounds</i> with the seconds it was present, so a 1 s horn inside a 20 s truck pass is still found. The % is YAMNet's confidence, not a probability.</p>
 <p><b>Signal detectors (🎯).</b> Each saved recording is also checked directly: horn beeps (narrow tones), sirens (long sweeping tones), regular beepers (reversing, alarms) and low rumble (most energy below 200 Hz). What they find is added to <i>other sounds</i>, and the beeps · sound column shows the beep count and total horn sound in seconds (⟳ = regular beeper).</p>
 <p><b>Cars vs scooters.</b> YAMNet has a separate <i>scooter_moto</i> group (classes Motorcycle, Light engine), but it is not very reliable for small scooters. You can teach it: relabel at least 8 events of each type in the table, then press the button. It learns from YAMNet's vehicle scores plus the spectral character of the sound (low rumble versus buzzy mid frequencies) and relabels the events you have not corrected yourself. Now: ${l.car} car / ${l.scooter} scooter_moto hand-labelled events with data; ${l.nofeat} older events have no data yet (run <code>python3 noise_events.py --reprocess</code> once).${l.model?` Current model: ${l.model.n_car} + ${l.model.n_scooter} examples, cross-validated accuracy ${l.model.cv_acc}%.`:''}</p>
 <p><button id=trn>🧠 Learn from my corrections</button> <span id=trnr></span></p>
 <p><b>Storage.</b> WAV 16 kHz mono ≈ 1.9 MB per minute of event. Now: ${i.events} events, ${i.rec_mb} MB in <code>recorded_sound</code>, ${i.free_gb} GB free on the disk. Nothing is deleted automatically.</p>
 <p><b>Blind spots.</b> Sounds less than ${i.delta} dB above the background are not recorded (a distant horn in rush hour). The first ~3 s after start are used to learn the background. Events during a restart are missed. Microphone auto-gain distorts levels. All times are the Pi's local time.</p>`;
 $('#trn').onclick=async()=>{$('#trnr').textContent='training…';const r=await (await fetch('/api/train',{method:'POST'})).json();$('#trnr').textContent=r.msg;if(r.ok)load()};
 $('#inf').showModal()};
$('#infx').onclick=()=>$('#inf').close();
$('#inf').addEventListener('click',e=>{if(e.target===$('#inf'))$('#inf').close()});
const HS={rows:[],busy:0,view:'list'},hdlg=$('#hn'),PAL=['#e53935','#1e88e5','#43a047','#fb8c00','#8e24aa','#00acc1'];
const PAT={none:'no horn tones found',single:'single short beep',single_long:'single long honk',double:'double beep',series:'short series (3–5)',periodic:'regular beeper (reversing / alarm)',long_series:'long series (6+)',error:'analysis failed'};
let HA=null;
function hnOpts(){const k=$('#hnc').value||'car_horn';$('#hnc').innerHTML=cats.map(c=>`<option${c===k?' selected':''}>${esc(c)}</option>`).join('')}
async function hnLoad(){const my=++HS.busy;$('#hnst').textContent='analyzing…';
 for(;;){const d=await (await fetch(`/api/horns?cat=${encodeURIComponent($('#hnc').value)}&hours=${$('#hnr').value}&minl=${MINL}`)).json();
  if(my!==HS.busy)return;HS.rows=d.rows;hnDraw();
  if(!d.pending){$('#hnst').textContent=(HNMSG?HNMSG+' · ':'')+d.rows.length+' events';return}$('#hnst').textContent=`analyzing… ${d.pending} left`}}
function hist(el,labels,counts,color){const W=Math.max(el.clientWidth,260),Hh=140,L=6,B=20,m=Math.max(...counts,1),bw=(W-L)/counts.length;
 el.innerHTML=`<svg width=${W} height=${Hh} style="display:block">`+counts.map((c,i)=>{const h=(Hh-B-16)*c/m,x=L+i*bw;
  return `<rect x=${x+4} y=${Hh-B-h} width=${bw-8} height=${h} fill="${color}"/><text x=${x+bw/2} y=${Hh-B-h-4} text-anchor=middle font-size=12 fill=currentColor>${c||''}</text><text x=${x+bw/2} y=${Hh-5} text-anchor=middle font-size=11 fill=currentColor opacity=.75>${labels[i]}</text>`}).join('')+'</svg>'}
function specCard(r){const dur=r.dur||r.length||1,d=new Date(r.ts*1000),
 bars=(r.beeps||[]).map(b=>`<i style="position:absolute;top:0;bottom:0;left:${(b[0]/dur*100).toFixed(3)}%;width:${Math.max((b[1]-b[0])/dur*100,.4).toFixed(3)}%;background:${PAL[b[3]%6]};opacity:.55"></i>`).join('');
 return `<div class=sc data-id="${r.id}" data-dur="${dur}" style="border:1px solid #8884;border-radius:10px;padding:8px 10px;margin:0 0 10px">
 <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px"><b>${d.toLocaleString()}</b>
  <span style="opacity:.75">${r.length.toFixed(1)} s · ${r.n} beep${r.n===1?'':'s'} · ${r.horn_s.toFixed(1)} s horn</span>
  <button class=hfx data-id="${r.id}" style="margin-left:auto" title="fix beeps of this event">✎ fix</button></div>
 <div class=scw style="position:relative;cursor:pointer"><img src="/spec/${encodeURIComponent(r.file.replace('.wav','.png'))}?t0=0&t1=${dur.toFixed(3)}&w=1000" alt="" draggable=false style="width:100%;height:120px;display:block;border-radius:4px;background:#000">
  <div style="position:absolute;inset:0">${bars}</div><i class=scp style="position:absolute;top:0;bottom:0;width:2px;background:#fff;left:0;display:none;pointer-events:none;box-shadow:0 0 3px #000"></i></div>
 <audio controls preload=none src="/audio/${encodeURIComponent(r.file)}" style="width:100%;margin-top:6px"></audio></div>`}
$('#hnv').addEventListener('click',e=>{const b=e.target.closest('[data-v]');if(!b)return;HS.view=b.dataset.v;
 document.querySelectorAll('#hnv button').forEach(x=>x.classList.toggle('on',x===b));hnDraw()});
$('#hns2').addEventListener('click',e=>{const b=e.target.closest('.hfx');if(b){hornEdit(+b.dataset.id);return}
 const w=e.target.closest('.scw');if(!w)return;const c=w.closest('.sc'),a=c.querySelector('audio'),r=w.getBoundingClientRect();if(!r.width)return;
 const dur=a.duration&&isFinite(a.duration)?a.duration:+c.dataset.dur;a.currentTime=Math.min(Math.max((e.clientX-r.left)/r.width,0),1)*dur;a.play()});
document.addEventListener('timeupdate',e=>{const a=e.target;if(a.tagName!=='AUDIO')return;const c=a.closest('.sc');if(!c)return;
 const p=c.querySelector('.scp');p.style.display='';p.style.left=(a.currentTime/(a.duration||+c.dataset.dur||1)*100)+'%'},true);
function hnDraw(){const R=HS.rows.filter(r=>r.pattern!=='error'),n=R.length,pc=k=>n?` (${Math.round(100*k/n)}%)`:'';
 const sing=R.filter(r=>r.n===1).length,rep=R.filter(r=>r.n>=2).length,multi=R.filter(r=>r.n_pitch>=2).length,tb=R.reduce((s,r)=>s+r.n,0);
 $('#hns').innerHTML=[['events',n],['single beep',sing+pc(sing)],['repeated (2+ beeps)',rep+pc(rep)],['beeps total',tb],
  ['avg beeps / event',n?(tb/n).toFixed(1):0],['max beeps',Math.max(0,...R.map(r=>r.n))],['2+ different pitches',multi+pc(multi)],
  ['horn sound',R.reduce((s,r)=>s+r.horn_s,0).toFixed(1)+' s']].map(([a,b])=>`<div><b>${b}</b><span>${a}</span></div>`).join('');
 const bk=v=>v>=10?7:v>=6?6:v,c1=new Array(8).fill(0),c2=new Array(5).fill(0);
 R.forEach(r=>{c1[bk(r.n)]++;c2[Math.min(r.n_pitch,4)]++});
 hist($('#hnh1'),['0','1','2','3','4','5','6–9','10+'],c1,'#1e88e5');hist($('#hnh2'),['0','1','2','3','4+'],c2,'#fb8c00');
 $('#hnt').innerHTML=HS.rows.map(r=>{const d=new Date(r.ts*1000),W=200;
  const tl=`<svg width=${W} height=20 class=tl data-f="${esc(r.file)}" data-d=${r.dur} style="cursor:pointer;display:block"><rect width=${W} height=20 fill="#8882" rx=3 />`+
   r.beeps.map(b=>`<rect x=${b[0]/r.dur*W} width=${Math.max((b[1]-b[0])/r.dur*W,2)} height=20 fill="${PAL[b[3]%6]}"><title>${b[0]}s–${b[1]}s, ${b[2]} Hz</title></rect>`).join('')+'</svg>';
  return `<tr><td>${d.toLocaleDateString()} <b>${d.toLocaleTimeString()}</b><td><b>${r.n}</b><td>${PAT[r.pattern]||r.pattern}
  <td>${r.clusters.map(([f,k],i)=>`<span style="color:${PAL[i%6]}">●</span> ${f} × ${k}`).join(' &nbsp; ')||'–'}<td>${tl}
  <td><button data-f="${esc(r.file)}" data-t=0>▶</button>`}).join('')}
$('#hnt').addEventListener('click',e=>{const el=e.target.closest('[data-f]');if(!el)return;let t=+(el.dataset.t||0);
 if(el.classList.contains('tl'))t=(e.clientX-el.getBoundingClientRect().left)/el.getBoundingClientRect().width*el.dataset.d;
 if(HA)HA.pause();HA=new Audio('/audio/'+encodeURIComponent(el.dataset.f));HA.addEventListener('loadedmetadata',()=>{HA.currentTime=t;HA.play()},{once:true});HA.load()});
$('#hnb').onclick=()=>{hdlg.showModal();hnOpts();hnLoad()};
let HNMSG='';
$('#hnl').onclick=async()=>{$('#hnst').textContent='learning…';const r=await (await fetch('/api/horn/learn',{method:'POST'})).json();HNMSG=r.msg;hnLoad()};
const HX_L='ABCDEFGH';let HX=null;
$('#tb').addEventListener('click',e=>{const b=e.target.closest('.hfx');if(b)hornEdit(+b.dataset.id)});
async function hornEdit(id){const j=await (await fetch('/api/horn/'+id)).json();
 const bs=j.beeps.map(b=>({s:b[0],e:b[1],g:b[2],f:b[3],keep:!!b[4],auto:!!b[5]}));
 const lens=bs.filter(b=>b.auto&&b.keep).map(b=>b.e-b.s).sort((x,y)=>x-y),med=lens.length?lens[lens.length>>1]:0.3;
 HX={id,file:j.file,dur:Math.max(j.dur,0.5),beeps:bs,defLen:Math.min(Math.max(med,0.05),2),sel:null,zoom:1,z0:0,z1:Math.max(j.dur,0.5)};
 $('#hfxa').src='/audio/'+encodeURIComponent(j.file);hxView(0,1);$('#hfx').showModal()}
const hxX=t=>((t-HX.z0)/(HX.z1-HX.z0)*100).toFixed(4)+'%';
function hxView(z0,zoom){const win=HX.dur/zoom;HX.zoom=zoom;HX.z0=Math.min(Math.max(z0,0),Math.max(HX.dur-win,0));HX.z1=Math.min(HX.z0+win,HX.dur);
 $('#hfxi').src='/spec/'+encodeURIComponent(HX.file.replace('.wav','.png'))+`?t0=${HX.z0.toFixed(4)}&t1=${HX.z1.toFixed(4)}&w=1400`;
 const pan=$('#hfxpan');pan.style.display=zoom>1?'':'none';pan.value=HX.dur-win>0?Math.round(HX.z0/(HX.dur-win)*1000):0;hxDraw()}
function hxOverlay(){const win=HX.z1-HX.z0;
 $('#hfxo').innerHTML=HX.beeps.map((b,k)=>b.e<HX.z0||b.s>HX.z1?'':
  b.keep?`<div data-bk=${k} title="${b.s.toFixed(3)} s · ${(b.e-b.s)*1000|0} ms · ${b.f||'?'} Hz · horn ${HX_L[b.g]}${b.keep?'':' (removed)'}" style="position:absolute;top:0;bottom:0;left:${hxX(b.s)};width:${Math.max((b.e-b.s)/win*100,0.3).toFixed(4)}%;background:${PAL[b.g%6]};opacity:.55;box-sizing:border-box;cursor:pointer"></div>`:'').join('')+
  (HX.sel?`<div style="position:absolute;top:0;bottom:0;left:${hxX(HX.sel[0])};width:${((HX.sel[1]-HX.sel[0])/win*100).toFixed(4)}%;background:#fff4;border:1px solid #fff;pointer-events:none"></div>`:'')}
function hxDraw(){hxOverlay();$('#hfxz').textContent=`${HX.z0.toFixed(2)}–${HX.z1.toFixed(2)} s of ${HX.dur.toFixed(2)} s · ${HX.zoom}×`;
 $('#hfxb').innerHTML=HX.beeps.map((b,k)=>`<tr class="${b.keep?'':'off'}"><td><input type=checkbox class=hk data-k=${k} ${b.keep?'checked':''}>
 <td><button class=hp data-k=${k}>▶</button>
 <td><input type=number step=any class=hs data-k=${k} value="${b.s.toFixed(3)}" style="width:84px"> s
 <td><input type=number step=1 class=hl data-k=${k} value="${Math.round((b.e-b.s)*1000)}" style="width:72px"> ms
 <td>${b.f||'?'} Hz
 <td><select class=hg data-k=${k} style="background:${PAL[b.g%6]};color:${txt(PAL[b.g%6])};font-weight:600">${[0,1,2,3,4,5,6,7].map(g=>`<option value=${g} style="background:${PAL[g%6]};color:${txt(PAL[g%6])}"${g===b.g?' selected':''}>horn ${HX_L[g]}</option>`).join('')}</select>
 <td>${b.auto?'detected':'added'}<td><button class=hd data-k=${k} title="delete this beep from the list">✕</button>`).join('');
 const kept=HX.beeps.filter(b=>b.keep);$('#hfxs').textContent=`${kept.length} beeps · ${new Set(kept.map(b=>b.g)).size} horn(s)`}
const zoomTo=z=>{const c=(HX.z0+HX.z1)/2,win=HX.dur/z;hxView(c-win/2,z)};
$('#hfxzi').onclick=()=>zoomTo(Math.min(HX.zoom*2,64));
$('#hfxzo').onclick=()=>zoomTo(Math.max(HX.zoom/2,1));
$('#hfxzf').onclick=()=>hxView(0,1);
$('#hfxpan').onchange=e=>{const win=HX.dur/HX.zoom;hxView(e.target.value/1000*(HX.dur-win),HX.zoom)};
$('#hfxb').addEventListener('change',e=>{const el=e.target,k=+el.dataset.k,b=HX.beeps[k];if(!b)return;
 if(el.classList.contains('hk'))b.keep=el.checked;
 else if(el.classList.contains('hg'))b.g=+el.value;
 else if(el.classList.contains('hs')){const len=b.e-b.s;b.s=Math.max(+el.value||0,0);b.e=b.s+len}
 else if(el.classList.contains('hl'))b.e=b.s+Math.max(+el.value||1,1)/1000;
 HX.beeps.sort((x,y)=>x.s-y.s);hxDraw()});
$('#hfxb').addEventListener('click',e=>{const dl=e.target.closest('.hd');if(dl){HX.beeps.splice(+dl.dataset.k,1);hxDraw();return}
 const p=e.target.closest('.hp');if(!p)return;const b=HX.beeps[+p.dataset.k];
 $('#hfxa').currentTime=Math.max(b.s-0.3,0);$('#hfxa').play()});
$('#hfxa').addEventListener('timeupdate',()=>{if(!HX)return;const t=$('#hfxa').currentTime,p=$('#hfxp');
 p.style.display=(t>=HX.z0&&t<=HX.z1)?'':'none';p.style.left=hxX(t)});
$('#hfxo').addEventListener('click',e=>{if(HMOVED){HMOVED=false;return}const bar=e.target.closest('[data-bk]');if(!bar)return;const b=HX.beeps[+bar.dataset.bk];b.g=(b.g+1)%6;hxDraw()});
$('#hfxo').addEventListener('contextmenu',e=>{e.preventDefault();const bar=e.target.closest('[data-bk]');if(!bar)return;HX.beeps.splice(+bar.dataset.bk,1);hxDraw()});
let HDRAG=null;
const hxT=e=>{const r=$('#hfxo').getBoundingClientRect();return HX.z0+Math.min(Math.max((e.clientX-r.left)/(r.width||1),0),1)*(HX.z1-HX.z0)};
let HMOVE=null,HMOVED=false;
$('#hfxo').addEventListener('mousedown',e=>{if(!HX||e.button!==0)return;HMOVED=false;
 const bar=e.target.closest('[data-bk]');
 if(bar){const k=+bar.dataset.bk,b=HX.beeps[k];HMOVE={k,x0:e.clientX,s0:b.s,e0:b.e,moved:false};e.preventDefault();return}
 HDRAG={t0:hxT(e),x0:e.clientX,moved:false};e.preventDefault()});
document.addEventListener('mousemove',e=>{if(!HMOVE||!HX)return;const dx=e.clientX-HMOVE.x0;if(Math.abs(dx)>=3)HMOVE.moved=true;
 if(!HMOVE.moved)return;const r=$('#hfxo').getBoundingClientRect(),dt=dx/(r.width||1)*(HX.z1-HX.z0),b=HX.beeps[HMOVE.k],len=HMOVE.e0-HMOVE.s0;
 b.s=Math.min(Math.max(HMOVE.s0+dt,0),Math.max(HX.dur-len,0));b.e=b.s+len;hxOverlay()});
document.addEventListener('mouseup',()=>{if(!HMOVE)return;const m=HMOVE;HMOVE=null;if(m.moved){HMOVED=true;HX.beeps.sort((x,y)=>x.s-y.s);hxDraw()}});
document.addEventListener('mousemove',e=>{if(!HDRAG||!HX)return;if(Math.abs(e.clientX-HDRAG.x0)>=4)HDRAG.moved=true;
 if(HDRAG.moved){const t=hxT(e);HX.sel=[Math.min(HDRAG.t0,t),Math.max(HDRAG.t0,t)];hxOverlay()}});
document.addEventListener('mouseup',e=>{if(!HDRAG||!HX)return;const d=HDRAG;HDRAG=null;const t=hxT(e);
 let s,en;if(d.moved){s=Math.min(d.t0,t);en=Math.max(d.t0,t)}else{s=d.t0;en=s+HX.defLen}
 HX.sel=null;
 if(en-s>=0.02){const near=HX.beeps.filter(b=>b.keep&&b.f).sort((x,y)=>Math.abs(x.s-s)-Math.abs(y.s-s))[0];
  HX.beeps.push({s:+s.toFixed(3),e:+en.toFixed(3),g:near?near.g:0,f:near?near.f:0,keep:true,auto:false});
  HX.beeps.sort((x,y)=>x.s-y.s)}
 hxDraw()});
$('#hfxadd').onclick=()=>{const s=parseFloat(prompt('Start of the missed beep, in seconds from the start of the recording:','0'));
 if(!(s>=0))return;const near=HX.beeps.filter(b=>b.keep&&b.f).sort((x,y)=>Math.abs(x.s-s)-Math.abs(y.s-s))[0];
 HX.beeps.push({s,e:s+HX.defLen,g:near?near.g:0,f:near?near.f:0,keep:true,auto:false});HX.beeps.sort((x,y)=>x.s-y.s);hxDraw()};
async function hxUpdate(r){if(!r.ok){alert('save failed');return}$('#hfx').close();await fetchDay();if(hdlg.open)hnLoad()}
$('#hfxsave').onclick=async()=>{const beeps=HX.beeps.filter(b=>b.keep).map(b=>[+b.s.toFixed(3),+b.e.toFixed(3),b.g,b.f||0]);
 await hxUpdate(await fetch('/api/horn/'+HX.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({beeps})}))};
$('#hfxrst').onclick=async()=>{await hxUpdate(await fetch('/api/horn/'+HX.id,{method:'DELETE'}))};
$('#hfxx').onclick=()=>$('#hfx').close();
$('#hfx').addEventListener('click',e=>{if(e.target===$('#hfx'))$('#hfx').close()});
$('#hnx').onclick=()=>hdlg.close();
hdlg.addEventListener('click',e=>{if(e.target===hdlg)hdlg.close()});
hdlg.addEventListener('close',()=>{if(HA)HA.pause()});
$('#hnc').onchange=$('#hnr').onchange=hnLoad;
addEventListener('resize',()=>{if(hdlg.open)hnDraw()});
async function renameCat(old){let n=prompt(`Rename category "${old}" for all its events (and for future ones):`,old);if(n===null)return;
 n=n.trim().toLowerCase().split(' ').join('_').slice(0,40);if(!n||n===old)return;
 if(cats.includes(n)&&!confirm(`"${n}" already exists. Merge "${old}" into it?`))return;
 const r=await fetch('/api/categories/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({old,new:n})});
 if(!r.ok){alert('rename failed ("false_alarm" is reserved and cannot be renamed)');return}
 cats=[];$('#f').innerHTML='<option value="">all categories</option>';
 (await (await fetch('/api/categories')).json()).forEach(addCat);colors=await (await fetch('/api/colors')).json();await load()}
$('#tb').addEventListener('click',e=>{if(e.target.classList.contains('ren'))renameCat(e.target.closest('tr').querySelector('select.pill').value)});
$('#tb').addEventListener('click',async e=>{if(!e.target.classList.contains('rs'))return;const id=+e.target.closest('tr').dataset.id;
 const r=await fetch('/api/events/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reset:true})});
 if(r.ok){const j=await r.json(),o=rows.find(x=>x.id===id);o.category=j.category;o.corrected=0;addCat(j.category);render()}});
$('#tb').addEventListener('click',async e=>{if(!e.target.classList.contains('ok'))return;const tr=e.target.closest('tr'),id=+tr.dataset.id;
 const r=await fetch('/api/events/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({category:tr.querySelector('select.pill').value})});
 if(r.ok){tr.classList.add('fixed');tr.querySelector('.cf').textContent='✓ manual';rows.find(x=>x.id===id).corrected=1}});
document.addEventListener('timeupdate',e=>{const a=e.target;if(a.tagName!=='AUDIO')return;
 a.closest('td').querySelector('.cur').style.left=(a.currentTime/(a.duration||1)*100)+'%'},true);
$('#tb').addEventListener('click',e=>{const sp=e.target.closest('.sp');if(!sp)return;const a=sp.parentNode.querySelector('audio');
 const x=(e.clientX-sp.getBoundingClientRect().left)/sp.offsetWidth;const go=()=>{a.currentTime=x*a.duration;a.play()};
 a.readyState?go():(a.addEventListener('loadedmetadata',go,{once:true}),a.load())});
let hp=-1;async function hornWatch(){const i=await (await fetch('/api/info?light=1')).json();if(hp>0&&i.horn_pending===0)load();hp=i.horn_pending}
setInterval(hornWatch,15000);hornWatch();
(function(){const root=document.documentElement;
 const set=v=>{if(v==='auto')delete root.dataset.theme;else root.dataset.theme=v;
  document.querySelectorAll('#th button').forEach(b=>b.classList.toggle('on',b.dataset.th===v));
  try{if(v==='auto')localStorage.removeItem('ne_theme');else localStorage.setItem('ne_theme',v)}catch(e){}};
 $('#th').addEventListener('click',e=>{const b=e.target.closest('[data-th]');if(b)set(b.dataset.th)});
 set(root.dataset.theme||'auto')})();
$('#tb').addEventListener('dblclick',async e=>{const t=e.target.closest('.tg');if(!t)return;
 const id=+t.closest('tr').dataset.id,cat=t.dataset.c;
 const r=await fetch('/api/events/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({category:cat})});
 if(r.ok){LAST_SET=cat;addCat(cat);await fetchDay()}else alert('could not change the category')});
// column names stay visible while scrolling: a fixed copy of the header row, aligned with the table
const TW=$('.tw'),TB=TW.querySelector('table'),THEAD=TB.querySelector('thead');
const HCL=document.createElement('div');HCL.id='hcl';HCL.innerHTML='<table></table>';document.body.appendChild(HCL);
function syncHead(){const H=$('header').offsetHeight,tr=THEAD.getBoundingClientRect(),tb=TB.getBoundingClientRect(),
 show=tr.top<H&&tb.bottom>H+tr.height;
 HCL.style.display=show?'':'none';if(!show)return;
 const tbl=HCL.firstChild,src=[...THEAD.querySelectorAll('th')];
 HCL.style.top=H+'px';HCL.style.left=TW.getBoundingClientRect().left+'px';HCL.style.width=TW.clientWidth+'px';
 tbl.style.cssText=`width:${TB.offsetWidth}px;margin-left:${-TW.scrollLeft}px;border-collapse:collapse`;
 tbl.innerHTML='';tbl.appendChild(THEAD.cloneNode(true));
 tbl.querySelectorAll('th').forEach((th,i)=>{th.style.width=src[i].offsetWidth+'px'})}
TW.addEventListener('scroll',syncHead,{passive:true});addEventListener('scroll',syncHead,{passive:true});addEventListener('resize',syncHead);
{const r0=render;render=function(){r0();applyCols();syncHead()}}
{const hd0=hnDraw;hnDraw=function(){hd0();const sp=$('#hns2'),tw=$('#hntw');
 if(HS.view==='spec'){tw.style.display='none';sp.style.display='';sp.innerHTML=HS.rows.map(specCard).join('')||'<p style="opacity:.7">no events in this range</p>'}
 else{tw.style.display='';sp.style.display='none'}}}
fetch('/api/colors').then(r=>r.json()).then(c=>{colors=c;load()});setInterval(poll,10000);
</script>"""

app = Flask(__name__)


@app.route("/")
def index():
    import json
    return PAGE.replace("%CATS%", json.dumps(category_list()))


@app.route("/api/days")
def api_days():
    """Days that have events (server local date), newest first, with counts."""
    days = sql("select date(ts,'unixepoch','localtime') d, count(*) n, coalesce(sum(length),0) s from events "
               "group by d order by d desc")
    tot = sql("select count(*) n, coalesce(sum(length),0) s, coalesce(min(ts),0) f from events")[0]
    last = sql("select coalesce(max(id),0) m from events")[0]["m"]
    return jsonify(days=[dict(day=r["d"], n=r["n"], s=round(r["s"] or 0, 1)) for r in days],
                   today=time.strftime("%Y-%m-%d"), maxId=last, total_n=tot["n"], total_s=round(tot["s"] or 0, 1),
                   first_ts=tot["f"])


@app.route("/api/events")
def api_events():
    """Events newest first. day=YYYY-MM-DD limits to one local day; before=<id> pages back inside it."""
    after = int(request.args.get("after", 0))
    before = int(request.args.get("before", 0))
    limit = min(int(request.args.get("limit", 500)), 5000)
    where, params = "e.id>?", [after]
    if before:
        where += " and e.id<?"
        params.append(before)
    if request.args.get("day"):
        try:
            y, m, d = map(int, request.args["day"].split("-"))
            import datetime
            datetime.date(y, m, d)                       # rejects impossible dates such as month 13
            t0, t1 = time.mktime((y, m, d, 0, 0, 0, 0, 0, -1)), time.mktime((y, m, d + 1, 0, 0, 0, 0, 0, -1))
        except (ValueError, OverflowError):
            abort(400)
        where += " and e.ts>=? and e.ts<?"
        params += [t0, t1]
    out = []
    for r in sql("select e.*, h.data hdata, mn.data md from events e left join horn h on h.event_id=e.id and h.ver=? "
                 "left join horn_manual mn on mn.event_id=e.id where " + where + " order by e.id desc limit ?",
                 [HORN_VER] + params + [limit]):
        d = dict(r)
        hd, md = d.pop("hdata"), d.pop("md")
        for k in ("labels", "timeline"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        h = json.loads(hd) if hd else None
        if h:
            h = horn_effective(h, json.loads(md)["beeps"] if md else None)
        d["horn"] = dict(n=h["n"], s=h["horn_s"], p=h.get("periodic", False), ok=h.get("trusted", True),
                         m=bool(md)) if h else None
        out.append(d)
    return jsonify(out)


@app.route("/api/events/<int:i>", methods=["POST"])
def api_set(i):
    j = request.get_json(force=True) or {}
    if j.get("reset"):                                   # back to the automatic label, and "not reviewed" again
        sql("update events set category=coalesce(orig_category, category), corrected=0 where id=?", (i,), commit=True)
        return jsonify(ok=True, category=sql("select category from events where id=?", (i,))[0]["category"])
    cat = str(j.get("category", "")).strip().lower().replace(" ", "_")[:40]
    if not cat:
        abort(400)
    row = sql("select category, labels, length from events where id=?", (i,))
    if row:
        labs = json.loads(row[0]["labels"] or "[]")
        old = row[0]["category"]
        if old != cat and old != "false_alarm" and not any(l[0] == old for l in labs):
            labs.append([old, None, round(row[0]["length"] or 0, 1), "y"])     # keep the replaced sound visible
        sql("update events set category=?, corrected=1, labels=? where id=?", (cat, json.dumps(labs), i), commit=True)
    return jsonify(ok=True)


@app.route("/api/events/<int:i>", methods=["DELETE"])
def api_del(i):
    for r in sql("select file from events where id=?", (i,)):
        for ext in (".wav", ".png"):
            try:
                os.remove(os.path.join(REC_DIR, os.path.basename(r["file"])[:-4] + ext))
            except OSError:
                pass
    sql("delete from horn where event_id=?", (i,), commit=True)
    sql("delete from horn_manual where event_id=?", (i,), commit=True)
    sql("delete from events where id=?", (i,), commit=True)
    return jsonify(ok=True)


def minl_sql():
    """Level filter from the sliders: only events whose average level reaches minl dB(A) (30 = no filter)."""
    try:
        v = float(request.args.get("minl", 0) or 0)
    except ValueError:
        return ""
    return " and laeq>=%.1f" % v if v > 30 else ""


def lt(fmt):
    return "strftime('" + fmt + "',ts,'unixepoch','localtime')"


def q15(prefix):        # key of the 15-minute bucket: prefix + HH:MM rounded down
    return "(" + lt(prefix + "%H:") + " || printf('%02d',(cast(" + lt("%M") + " as int)/15)*15))"


@app.route("/api/stats")
def api_stats():
    """Aggregates for the analysis window (local time of the machine running this script).
    step: 3600 hourly buckets, 900 = 15-minute buckets, 86400 = calendar days, 0 = free (every event as a point)."""
    mode, now = request.args.get("mode", "recent"), time.time()
    step = {900: 900, 0: 0, 86400: 86400}.get(int(request.args.get("step", 3600)), 3600)
    base, ndays, points, wd = "category!='false_alarm'" + minl_sql(), 1, [], None
    if mode == "recent":
        hours_arg = int(request.args.get("hours", 48))
        if step == 86400 and hours_arg == 0:                 # all days since the first event
            first = sql("select min(ts) m from events")[0]["m"]
            t0 = first if first else now
            hours = None
        else:
            hours = min(max(hours_arg, 1), 24 * (7 if step == 900 else 90 if step == 3600 else 36500))
            t0 = now - hours * 3600
        where, params, t1 = base + " and ts>=?", (t0,), now
        if step == 86400:
            key = lt("%Y-%m-%d")
            day, today, keys = datetime.date.fromtimestamp(t0), datetime.date.fromtimestamp(now), []
            while day <= today:
                keys.append(day.isoformat())
                day += datetime.timedelta(days=1)
        else:
            key = q15("%Y-%m-%d ") if step == 900 else lt("%Y-%m-%d %H:00")
            keys = [] if step == 0 else list(dict.fromkeys(
                "%s %02d:%02d" % (time.strftime("%Y-%m-%d", lc), lc.tm_hour, lc.tm_min // 15 * 15 if step == 900 else 0)
                for lc in (time.localtime(now - i * (step or 3600)) for i in range(hours * 3600 // (step or 3600), -1, -1))))
    else:
        days = max(int(request.args.get("days", 0)), 0)
        wd = [int(x) for x in request.args.get("wd", "").split(",") if x.strip().isdigit() and int(x) < 7]
        where, params, t0, t1 = base + " and " + lt("%w") + " in (" + ",".join("'%d'" % d for d in wd) + ")", (), 0, 86400
        if days:
            where, params = where + " and ts>=?", (now - days * 86400,)
        key = q15("") if step == 900 else lt("%H:00")
        keys = [] if step == 0 else ["%02d:%02d" % (h, m) for h in range(24) for m in range(0, 60, 15 if step == 900 else 60)]
    rows = []
    if wd is not None and not wd:
        pass                                              # no weekday selected -> empty result
    elif step == 0:
        tod = "(cast(%s as int)*3600+cast(%s as int)*60+cast(%s as int))" % (lt("%H"), lt("%M"), lt("%S"))
        points = [dict(r) for r in sql("select id,ts,category c,length l,laeq a,peak_db pk," + tod + " tod from events "
                                       "where " + where + " order by ts desc limit 5001", params)]
    else:
        agg = {}
        for r in sql("select " + key + " b, category c, length l, laeq a, peak_db pk from events where " + where, params):
            g = agg.setdefault((r["b"], r["c"]), [0, 0.0, 0.0, -999.0])
            g[0] += 1
            g[1] += r["l"] or 0
            g[2] += 10 ** ((r["a"] or 0) / 10)          # energy sum -> average level per bucket on the page
            g[3] = max(g[3], r["pk"] or 0)
        rows = [dict(b=b, c=c, n=g[0], s=g[1], pw=g[2], pk=g[3]) for (b, c), g in agg.items()]
        if mode != "recent":
            ndays = sql("select count(distinct date(ts,'unixepoch','localtime')) d from events where " + where,
                        params)[0]["d"]
    return jsonify(keys=keys, ndays=ndays, rows=rows, points=points[:5000], truncated=len(points) > 5000,
                   t0=t0, t1=t1, tz=time.localtime(now).tm_gmtoff)


@app.route("/api/sources")
def api_sources():
    """Which sound weighs most in the noise: per category total sound time, energy-mean level, peak, energy share."""
    now = time.time()
    days = max(int(request.args.get("days", 0)), 0)
    wd = [int(x) for x in request.args.get("wd", "").split(",") if x.strip().isdigit() and int(x) < 7]
    where, params = "category!='false_alarm'" + minl_sql() + " and " + lt("%w") + " in (" + ",".join("'%d'" % d for d in wd or [9]) + ")", []
    if days:
        where += " and ts>=?"
        params.append(now - days * 86400)
    hr = "cast(" + lt("%H") + " as int)"
    where += {"day": " and " + hr + " between 7 and 18", "evening": " and " + hr + " between 19 and 22",
              "night": " and (" + hr + ">=23 or " + hr + "<7)"}.get(request.args.get("period", "all"), "")
    extras, agg, dist = request.args.get("ex") == "1", {}, {}
    TH = list(range(40, 95, 5))
    for r in sql("select category c, length l, laeq a, peak_db pk, labels from events where " + where, params):
        parts = [(r["c"], r["l"] or 0)]
        if extras and r["labels"]:                      # time each sound was present, the event's level applies
            labs = [(l[0], l[2]) for l in json.loads(r["labels"]) if l[2]]
            if labs:
                parts = labs + ([] if any(c == r["c"] for c, _ in labs) else [(r["c"], r["l"] or 0)])
        for c, secs in parts:
            g = agg.setdefault(c, [0, 0.0, 0.0, -999.0])
            g[0] += 1
            g[1] += secs
            g[2] += secs * 10 ** ((r["a"] or 0) / 10)
            g[3] = max(g[3], r["pk"] or 0)
            dist.setdefault(c, []).append((r["a"] or 0, r["pk"] or 0, secs))
    te, ts = sum(g[2] for g in agg.values()) or 1, sum(g[1] for g in agg.values()) or 1
    def exceeded(levels, secs, frac):                   # level that is exceeded for `frac` of the sound time
        o = np.argsort(levels)[::-1]
        cum = np.cumsum(secs[o]) / secs.sum()
        return float(levels[o][min(int(np.searchsorted(cum, frac)), len(o) - 1)])
    cats = []
    for c, g in agg.items():
        if g[1] <= 0:
            continue
        a, pk, sc = (np.array(v, dtype=float) for v in zip(*dist[c]))
        th = dict(avg_s=[round(float(sc[a >= t].sum()), 1) for t in TH], avg_n=[int((a >= t).sum()) for t in TH],
                  pk_s=[round(float(sc[pk >= t].sum()), 1) for t in TH], pk_n=[int((pk >= t).sum()) for t in TH])
        cats.append(dict(c=c, n=g[0], s=round(g[1], 1), avg=round(10 * float(np.log10(g[2] / g[1])), 1), pk=g[3],
                         share=g[2] / te, tshare=g[1] / ts, th=th, l10=exceeded(a, sc, 0.1), l50=exceeded(a, sc, 0.5),
                         l90=exceeded(a, sc, 0.9)))
    return jsonify(thr=TH, cats=sorted(cats, key=lambda c: -c["share"]))


@app.route("/api/heat")
def api_heat():
    """Weekday x time-of-day grid: per cell and category the events, sound seconds, energy sum and peak."""
    now = time.time()
    days = max(int(request.args.get("days", 0)), 0)
    step = 900 if request.args.get("step") == "900" else 3600
    where, params = "category!='false_alarm'" + minl_sql(), []
    if days:
        where += " and ts>=?"
        params.append(now - days * 86400)
    keys = ["%02d:%02d" % (h, m) for h in range(24) for m in range(0, 60, step // 60)]
    ix = {k: i for i, k in enumerate(keys)}
    agg = {}
    for r in sql("select " + lt("%w") + " w, " + (q15("") if step == 900 else lt("%H:00")) + " b, category c, length l, "
                 "laeq a, peak_db pk from events where " + where, params):
        if r["b"] not in ix:
            continue
        g = agg.setdefault((int(r["w"]), ix[r["b"]], r["c"]), [0, 0.0, 0.0, -999.0])
        l = r["l"] or 0
        g[0] += 1
        g[1] += l
        g[2] += l * 10 ** ((r["a"] or 0) / 10)
        g[3] = max(g[3], r["pk"] or 0)
    wd = {int(r["w"]): r["n"] for r in sql("select " + lt("%w") + " w, count(distinct date(ts,'unixepoch','localtime')) n "
                                           "from events where " + where + " group by w", params)}
    alld = sql("select count(distinct date(ts,'unixepoch','localtime')) n from events where " + where, params)[0]["n"]
    return jsonify(keys=keys, wdays=[wd.get(d, 0) for d in range(7)], alldays=alld,
                   cells=[[w, i, c, g[0], round(g[1], 1), g[2], g[3]] for (w, i, c), g in agg.items()])


def learn_status():
    m = load_vehicle_model()
    n = {r["c"]: r["n"] for r in sql("select category c, count(*) n from events where corrected=1 and feat is not null "
                                     "and category in ('car','scooter_moto') group by c")}
    return dict(model={k: m[k] for k in ("n_car", "n_scooter", "cv_acc")} if m else None, car=n.get("car", 0),
                scooter=n.get("scooter_moto", 0), nofeat=sql("select count(*) c from events where feat is null")[0]["c"])


@app.route("/api/info")
def api_info():
    pend = sql("select count(*) c from events where id not in (select event_id from horn where ver=?)", (HORN_VER,))[0]["c"]
    if request.args.get("light"):
        return jsonify(horn_pending=pend)
    files = [os.path.join(REC_DIR, f) for f in os.listdir(REC_DIR)] if os.path.isdir(REC_DIR) else []
    return jsonify(dict(SETTINGS, block_s=round(BLK_S, 3), events=sql("select count(*) c from events")[0]["c"],
                        rec_mb=round(sum(os.path.getsize(f) for f in files) / 1e6, 1),
                        free_gb=round(shutil.disk_usage(HERE).free / 1e9, 1), learn=learn_status(),
                        horn_pending=pend))


@app.route("/api/train", methods=["POST"])
def api_train():
    return jsonify(train_vehicle_model())


@app.route("/api/horns")
def api_horns():
    """Beep analysis of events in a category (default car_horn). Results are cached in table `horn`;
    new events are analysed lazily, a batch per request ('pending' tells the page to ask again)."""
    cat = request.args.get("cat", "car_horn")
    hours = max(int(request.args.get("hours", 0)), 0)
    where, params = "(e.category=? or e.labels like ?)" + minl_sql(), [cat, '%"' + cat + '"%']
    if hours:
        where += " and e.ts>=?"
        params.append(time.time() - hours * 3600)
    todo = ("select e.id,e.file from events e where " + where +
            " and e.id not in (select event_id from horn where ver=?) order by e.id desc")
    t0 = time.time()
    for r in sql(todo + " limit 40", params + [HORN_VER]):
        try:
            res = analyze_horn(os.path.join(REC_DIR, os.path.basename(r["file"])))
        except Exception as e:
            print("horn analysis failed:", r["file"], e)
            res = dict(n=0, horn_s=0, n_pitch=0, pattern="error", dur=0, beeps=[], clusters=[])
        sql("insert or replace into horn(event_id,ver,data) values(?,?,?)", (r["id"], HORN_VER, json.dumps(res)),
            commit=True)
        if time.time() - t0 > 6:
            break
    pending = len(sql(todo, params + [HORN_VER]))
    rows = sql("select e.id,e.ts,e.length,e.category,e.file,h.data,mn.data md from events e join horn h "
               "on h.event_id=e.id and h.ver=? left join horn_manual mn on mn.event_id=e.id where " + where +
               " order by e.id desc limit 500", [HORN_VER] + params)
    out = []
    for r in rows:
        man = json.loads(r["md"])["beeps"] if r["md"] else None
        out.append(dict(horn_effective(json.loads(r["data"]), man), id=r["id"], ts=r["ts"], length=r["length"],
                        file=r["file"]))
    return jsonify(pending=pending, rows=out)


@app.route("/api/horn/learn", methods=["POST"])
def api_horn_learn():
    return jsonify(learn_horn())


@app.route("/api/horn/<int:i>")
def api_horn_get(i):
    """Beeps of one event for the fix window: detected beeps (kept or not) and beeps you added."""
    r = sql("select e.file, e.length, h.data hd, mn.data md from events e left join horn h on h.event_id=e.id and h.ver=? "
            "left join horn_manual mn on mn.event_id=e.id where e.id=?", (HORN_VER, i))
    if not r:
        abort(404)
    det = json.loads(r[0]["hd"]) if r[0]["hd"] else {}
    man = json.loads(r[0]["md"])["beeps"] if r[0]["md"] else None
    return jsonify(file=r[0]["file"], dur=det.get("dur") or r[0]["length"], beeps=editor_beeps(det, man),
                   manual=man is not None)


@app.route("/api/horn/<int:i>", methods=["POST"])
def api_horn_set(i):
    """Save your beep list: [[start_s, end_s, horn_index 0-7, Hz], ...]. Only kept beeps are sent."""
    try:
        raw = (request.get_json(force=True) or {}).get("beeps", [])
        beeps = sorted([[round(float(b[0]), 3), round(float(b[1]), 3), int(b[2]), round(float(b[3]))] for b in raw])
    except (TypeError, ValueError, IndexError):
        abort(400)
    if any(b[1] <= b[0] or not 0 <= b[2] <= 7 for b in beeps):
        abort(400)
    if not sql("select 1 from events where id=?", (i,)):
        abort(404)
    sql("insert or replace into horn_manual(event_id,data,updated) values(?,?,?)",
        (i, json.dumps({"beeps": beeps}), time.time()), commit=True)
    sync_horn_labels(i)
    return jsonify(ok=True, horn=horn_summary(i))


@app.route("/api/horn/<int:i>", methods=["DELETE"])
def api_horn_reset(i):
    sql("delete from horn_manual where event_id=?", (i,), commit=True)
    sync_horn_labels(i)
    return jsonify(ok=True, horn=horn_summary(i))


@app.route("/api/categories")
def api_categories():
    return jsonify(category_list())


@app.route("/api/categories/rename", methods=["POST"])
def api_rename():
    """Rename a category for all its events (merges if the new name exists); the colour follows the name."""
    j = request.get_json(force=True) or {}

    def clean(v):
        return str(v or "").strip().lower().replace(" ", "_")[:40]
    old, new = clean(j.get("old")), clean(j.get("new"))
    if not old or not new or old == new or "false_alarm" in (old, new):
        abort(400)
    n = sql("select count(*) c from events where category=?", (old,))[0]["c"]
    sql("update events set category=? where category=?", (new, old), commit=True)
    sql("update events set orig_category=? where orig_category=?", (new, old), commit=True)
    with _colors_lock:
        d = load_colors()
        col = d.pop(old, DEFAULT_COLORS.get(old))
        if col and new not in d and new not in DEFAULT_COLORS:
            d[new] = col
        save_json(COLORS_PATH, d)
        al = load_json(ALIASES_PATH)
        for k, v in list(al.items()):
            if v == old:
                al[k] = new
        if old in BASE_CATS or old in al:
            al[old] = new
        save_json(ALIASES_PATH, {k: v for k, v in al.items() if k != v})
    return jsonify(ok=True, changed=n)


@app.route("/api/colors")
def api_colors():
    d = dict(DEFAULT_COLORS)
    d.update(load_colors())
    return jsonify(d)


@app.route("/api/colors", methods=["POST"])
def api_set_color():
    j = request.get_json(force=True) or {}
    cat = str(j.get("category", "")).strip().lower().replace(" ", "_")[:40]
    col = str(j.get("color", ""))
    if not cat or not re.match(r"^#[0-9a-fA-F]{6}$", col):
        abort(400)
    with _colors_lock:
        d = load_colors()
        d[cat] = col.lower()
        with open(COLORS_PATH + ".tmp", "w") as f:
            json.dump(d, f, indent=1)
        os.replace(COLORS_PATH + ".tmp", COLORS_PATH)
    return jsonify(ok=True)


@app.route("/spec/<path:fn>")
def spec(fn):
    fn = os.path.basename(fn)
    wav, png = (os.path.join(REC_DIR, fn[:-4] + ext) for ext in (".wav", ".png"))
    if not fn.endswith(".png") or not os.path.exists(wav):
        abort(404)
    if "t0" in request.args:                         # zoomed view for the fix window: rendered on demand
        try:
            t0, t1 = float(request.args["t0"]), float(request.args["t1"])
            W = min(max(int(request.args.get("w", 1000)), 200), 2000)
        except ValueError:
            abort(400)
        x = read_wav(wav)
        a, b = int(max(t0, 0) * SR), int(min(t1, len(x) / SR) * SR)
        if b - a < 512:
            abort(400)
        return Response(_png(np.ascontiguousarray(spectro_rgb(x[a:b], W, 120, nperseg=256, noverlap=192))),
                        mimetype="image/png")
    if not os.path.exists(png):                      # recordings made before spectrograms existed
        make_spectrogram(wav, png)
    return send_from_directory(REC_DIR, fn)


@app.route("/audio/<path:fn>")
def audio(fn):
    return send_from_directory(REC_DIR, fn)


@app.route("/export.csv")
def export():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp", "length_s", "category", "likelihood_pct", "avg_db", "peak_db", "other_labels", "top_labels",
                "horn_beeps", "horn_sound_s", "file", "manually_corrected"])
    for r in sql("select e.*, h.data hdata, mn.data md from events e left join horn h on h.event_id=e.id and h.ver=? "
                 "left join horn_manual mn on mn.event_id=e.id order by e.id", (HORN_VER,)):
        hd = horn_effective(json.loads(r["hdata"]), json.loads(r["md"])["beeps"] if r["md"] else None) if r["hdata"] else {}
        lab = "; ".join("%s %ss" % (l[0], l[2]) for l in json.loads(r["labels"] or "[]") if l[0] != r["category"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])), r["length"], r["category"], r["pct"],
                    r["laeq"], r["peak_db"], lab, r["raw"], hd.get("n", ""), hd.get("horn_s", ""), r["file"],
                    r["corrected"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=events.csv"})


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=4)
    p.add_argument("--offset", type=float, default=94.0, help="dBFS(A) -> dB(A) calibration offset")
    p.add_argument("--delta", type=float, default=8.0, help="dB above background that starts an event")
    p.add_argument("--hold", type=float, default=10.0, help="seconds of quiet before an event is closed")
    p.add_argument("--max-len", type=float, default=60.0, help="max event length, seconds")
    p.add_argument("--split", type=float, default=3.0,
                   help="split an event when a different dominant sound lasts this many seconds (0 = off)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--web-only", "--muted", dest="web_only", action="store_true",
                   help="muted mode: no microphone and no model, only the web UI on the saved data")
    p.add_argument("--reprocess", nargs="?", const="missing", choices=["missing", "all"],
                   help="re-analyse saved recordings (labels, strip, car-vs-scooter data) and exit; 'all' redoes every event")
    args = p.parse_args()
    os.makedirs(REC_DIR, exist_ok=True)
    init_db()
    if args.reprocess:
        reprocess(args, args.reprocess)
        raise SystemExit
    threading.Thread(target=horn_backfill, daemon=True).start()
    SETTINGS.update(device=args.device, offset=args.offset, delta=args.delta, hold=args.hold, max_len=args.max_len,
                    split=args.split,
                    recording=not args.web_only)
    if not args.web_only:
        threading.Thread(target=recorder, args=(args,), daemon=True).start()
    print(f"Web UI: http://{lan_ip()}:{args.port}")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
