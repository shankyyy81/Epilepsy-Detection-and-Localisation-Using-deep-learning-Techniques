"""
Format-agnostic EEG dataset discovery, loading, validation and reporting.

Supports per-segment files (.txt/.csv/.tsv/.dat/.npy/.npz/.mat/.edf) in a directory tree
and bulk arrays (a single .npy/.npz/.mat holding [N, ...] plus labels).

Internal representation: X float32 [N, C, T], y int64 [N] (1 = focal, 0 = non-focal).
Nothing is silently discarded: malformed files are listed in the report (or raise if strict).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEGMENT_EXTS = (".txt", ".csv", ".tsv", ".dat", ".npy", ".npz", ".mat", ".edf")
FS_KEYS = ("fs", "Fs", "FS", "srate", "sampling_rate", "SampleRate", "sfreq", "sample_rate")
X_KEYS = ("X", "x", "data", "signals", "eeg", "EEG", "signal")
Y_KEYS = ("y", "Y", "labels", "label", "target", "targets")

BERN_RE = re.compile(r"Data_([FN])_Ind(\d+)", re.IGNORECASE)


@dataclass
class EEGData:
    X: np.ndarray                       # [N, C, T] float32
    y: np.ndarray                       # [N] int64
    names: List[str]
    pair_index: np.ndarray              # [N] int (-1 if unknown)
    patient_ids: Optional[np.ndarray]   # [N] str or None
    fs: float
    fs_source: str
    channel_names: List[str]
    groups: np.ndarray                  # [N] int duplicate-cluster id
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def seq_len(self) -> int:
        return int(self.X.shape[2])


# ---------------------------------------------------------------------------
# label / metadata parsing
# ---------------------------------------------------------------------------
def parse_label(path: str) -> Tuple[Optional[int], int]:
    """Return (label, pair_index) from a file path. label None if undeterminable."""
    base = os.path.basename(path)
    m = BERN_RE.search(base)
    if m:
        return (1 if m.group(1).upper() == "F" else 0), int(m.group(2))
    low = path.replace("\\", "/").lower()
    parts = low.split("/")
    for token in ("nonfocal", "non_focal", "non-focal", "non focal"):
        if token in low:
            return 0, -1
    if "focal" in low:
        return 1, -1
    for p in reversed(parts[:-1]):
        if p in ("n", "nf", "nonf"):
            return 0, -1
        if p in ("f",):
            return 1, -1
    return None, -1


def load_patient_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"patient_map not found: {path}")
    mapping = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        fcol = cols.get("file") or cols.get("filename") or cols.get("name")
        pcol = cols.get("patient_id") or cols.get("patient") or cols.get("subject")
        if not fcol or not pcol:
            raise ValueError("patient_map CSV needs columns 'file' and 'patient_id'")
        for row in reader:
            key = os.path.splitext(os.path.basename(row[fcol].strip()))[0]
            mapping[key] = str(row[pcol]).strip()
    return mapping


# ---------------------------------------------------------------------------
# per-format readers -> (array [C, T] or [N, C, T], fs or None, channel names or None)
# ---------------------------------------------------------------------------
def _sniff_delimiter(line: str) -> Optional[str]:
    for d in (",", ";", "\t"):
        if d in line:
            return d
    return None  # whitespace


def _is_numeric_row(tokens: List[str]) -> bool:
    try:
        [float(t) for t in tokens if t.strip() != ""]
        return True
    except ValueError:
        return False


def read_text(path: str):
    with open(path, "r") as fh:
        first = ""
        while first.strip() == "":
            first = fh.readline()
            if not first:
                raise ValueError("empty file")
    delim = _sniff_delimiter(first)
    tokens = first.split(delim) if delim else first.split()
    header = None if _is_numeric_row(tokens) else 0
    names = [t.strip() for t in tokens] if header == 0 else None
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=delim if delim else r"\s+", header=header,
                         skipinitialspace=True, engine="c" if delim else "python")
        arr = df.to_numpy(dtype=np.float64)
    except ImportError:
        arr = np.loadtxt(path, delimiter=delim, skiprows=1 if header == 0 else 0)
    return arr, None, names


def read_npy(path: str):
    arr = np.load(path, allow_pickle=False)
    return np.asarray(arr, dtype=np.float64), None, None


def _pick(d: Dict[str, Any], keys) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def read_npz(path: str):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    x = _pick(d, X_KEYS)
    if x is None:
        x = max((v for v in d.values() if np.issubdtype(v.dtype, np.number)), key=lambda v: v.size)
    fs = _pick(d, FS_KEYS)
    return np.asarray(x, dtype=np.float64), (float(np.ravel(fs)[0]) if fs is not None else None), None


def _read_mat_dict(path: str) -> Dict[str, Any]:
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        return {k: v for k, v in d.items() if not k.startswith("__")}
    except NotImplementedError:  # MATLAB v7.3 -> HDF5
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                try:
                    out[k] = np.array(f[k]).T
                except Exception:
                    pass
        return out


def read_mat(path: str):
    d = _read_mat_dict(path)
    numeric = {k: np.asarray(v) for k, v in d.items()
               if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)}
    fs = _pick(numeric, FS_KEYS)
    x = _pick(numeric, X_KEYS)
    if x is None:
        cands = [v for k, v in numeric.items() if k not in FS_KEYS]
        if not cands:
            raise ValueError("no numeric array in .mat")
        x = max(cands, key=lambda v: v.size)
    return np.asarray(x, dtype=np.float64), (float(np.ravel(fs)[0]) if fs is not None else None), None


def read_edf(path: str):
    try:
        import mne
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        return raw.get_data().astype(np.float64), float(raw.info["sfreq"]), list(raw.ch_names)
    except ImportError:
        pass
    try:
        import pyedflib
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        sigs = [f.readSignal(i) for i in range(n)]
        fs = float(f.getSampleFrequency(0))
        names = f.getSignalLabels()
        f.close()
        L = min(len(s) for s in sigs)
        return np.stack([s[:L] for s in sigs]).astype(np.float64), fs, names
    except ImportError as e:
        raise ImportError("Reading .edf requires `mne` or `pyedflib`") from e


READERS = {".txt": read_text, ".csv": read_text, ".tsv": read_text, ".dat": read_text,
           ".npy": read_npy, ".npz": read_npz, ".mat": read_mat, ".edf": read_edf}


def to_channels_first(a: np.ndarray) -> np.ndarray:
    """Standardise a single segment to [C, T] (time assumed to be the longer axis)."""
    a = np.squeeze(a)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim != 2:
        raise ValueError(f"segment has {a.ndim} dims after squeeze: {a.shape}")
    return a.T if a.shape[0] > a.shape[1] else a


# ---------------------------------------------------------------------------
# discovery + loading
# ---------------------------------------------------------------------------
def discover_files(root: str) -> List[str]:
    files = []
    for ext in SEGMENT_EXTS:
        files += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
        files += glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True)
    files = sorted(set(f for f in files if not os.path.basename(f).startswith(".")))
    return files


def _cache_key(files: List[str], extra: str = "") -> str:
    h = hashlib.md5()
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(extra.encode())
    return h.hexdigest()[:12]


def _load_one(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        arr, fs, names = READERS[ext](path)
        seg = to_channels_first(arr)
        return path, seg.astype(np.float32), fs, names, None
    except Exception as e:  # recorded, never silently dropped
        return path, None, None, None, f"{type(e).__name__}: {e}"


def _load_bulk(path: str, labels_path: Optional[str]):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        X = np.load(path)
        fs, y = None, None
    elif ext == ".npz":
        z = np.load(path)
        d = {k: z[k] for k in z.files}
        X, y, fs = _pick(d, X_KEYS), _pick(d, Y_KEYS), _pick(d, FS_KEYS)
    elif ext == ".mat":
        d = _read_mat_dict(path)
        X, y, fs = _pick(d, X_KEYS), _pick(d, Y_KEYS), _pick(d, FS_KEYS)
    else:
        raise ValueError(f"Unsupported bulk file {path}")
    if X is None:
        raise ValueError(f"Could not find signal array in {path}")
    if labels_path:
        y = np.load(labels_path) if labels_path.endswith(".npy") else np.loadtxt(labels_path)
    if y is None:
        raise ValueError("Bulk array requires labels (in file or data.labels_path)")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"Bulk X must be [N,C,T] or [N,T,C]; got {X.shape}")
    if X.shape[1] > X.shape[2]:  # [N, T, C] -> [N, C, T]
        X = np.transpose(X, (0, 2, 1))
    y = np.asarray(y).astype(np.int64).ravel()
    fs = float(np.ravel(fs)[0]) if fs is not None else None
    names = [f"sample_{i:05d}" for i in range(len(y))]
    return np.ascontiguousarray(X), y, names, fs


# ---------------------------------------------------------------------------
# duplicate detection (union-find over shared channels / shared sample blocks)
# ---------------------------------------------------------------------------
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def find_duplicate_groups(X: np.ndarray, mode: str = "block", block: int = 64,
                          decimals: int = 3) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Group segments that share an identical channel (exact) or an identical `block`-sample
    run at any offset in any channel (block). Returns group ids and the list of
    multi-member clusters. Used as `groups` for StratifiedGroupKFold.
    """
    n = X.shape[0]
    uf = _UF(n)
    if mode == "none":
        return np.arange(n), []
    R = np.round(X, decimals).astype(np.float32)
    seen: Dict[bytes, int] = {}
    for i in range(n):
        for c in range(R.shape[1]):
            key = hashlib.md5(R[i, c].tobytes()).digest()
            if key in seen and seen[key] != i:
                uf.union(seen[key], i)
            else:
                seen[key] = i
    if mode == "block":
        T = R.shape[2]
        table: Dict[bytes, int] = {}
        for i in range(n):
            for c in range(R.shape[1]):
                s = R[i, c]
                for st in range(0, T - block + 1, block):
                    table.setdefault(s[st:st + block].tobytes(), i)
        for i in range(n):
            for c in range(R.shape[1]):
                s = R[i, c]
                v = np.lib.stride_tricks.sliding_window_view(s, block)
                # skip flat runs (e.g. zero padding) which would create spurious matches
                for st in range(v.shape[0]):
                    w = v[st]
                    key = w.tobytes()
                    j = table.get(key)
                    if j is not None and j != i and w.max() != w.min():
                        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    clusters = defaultdict(list)
    for i, g in enumerate(groups):
        clusters[g].append(i)
    multi = [v for v in clusters.values() if len(v) > 1]
    return groups.astype(np.int64), multi


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def load_dataset(cfg, logger=None) -> EEGData:
    log = logger.info if logger else print
    dcfg = cfg.data
    root = dcfg.root
    if not os.path.exists(root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    os.makedirs(dcfg.cache_dir, exist_ok=True)
    malformed: List[Dict[str, str]] = []
    unlabeled: List[str] = []
    fs_found: List[float] = []
    ch_names = None

    if os.path.isfile(root):
        X, y, names, fs_file = _load_bulk(root, dcfg.labels_path)
        pair_index = np.arange(len(y))
        if fs_file:
            fs_found.append(fs_file)
        n_files_total = 1
        groups_cached = None
    else:
        files = discover_files(root)
        n_files_total = len(files)
        if not files:
            raise FileNotFoundError(f"No supported EEG files ({SEGMENT_EXTS}) under {root}")
        exts = Counter(os.path.splitext(f)[1].lower() for f in files)
        log(f"Discovered {len(files)} files under {root}: {dict(exts)}")
        key = _cache_key(files, f"{dcfg.dup_check}|{dcfg.dup_block}")
        cache = os.path.join(dcfg.cache_dir, f"dataset_{key}.npz")
        groups_cached = None
        if dcfg.use_cache and os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            X, y, names = z["X"], z["y"], list(z["names"])
            pair_index = z["pair_index"]
            groups_cached = z["groups"]
            malformed = json.loads(str(z["malformed"]))
            unlabeled = json.loads(str(z["unlabeled"]))
            fs_found = list(z["fs_found"])
            ch_names = list(z["ch_names"]) if z["ch_names"].size else None
            log(f"Loaded cached arrays from {cache}")
        else:
            labels, segs, kept, pidx = [], [], [], []
            with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
                results = list(ex.map(_load_one, files))
            for path, seg, fs, names_, err in results:
                if err is not None:
                    malformed.append({"file": path, "reason": err})
                    continue
                lab, pi = parse_label(path)
                if lab is None:
                    unlabeled.append(path)
                    continue
                if not np.isfinite(seg).all():
                    malformed.append({"file": path, "reason": "non-finite values"})
                    continue
                if fs:
                    fs_found.append(fs)
                if names_ and ch_names is None:
                    ch_names = names_
                labels.append(lab); segs.append(seg); kept.append(path); pidx.append(pi)
            if not segs:
                raise RuntimeError("No valid labelled segments were loaded.")
            shapes = Counter(s.shape for s in segs)
            C_mode = Counter(s.shape[0] for s in segs).most_common(1)[0][0]
            T_mode = Counter(s.shape[1] for s in segs).most_common(1)[0][0]
            if len(shapes) > 1:
                log(f"WARNING: heterogeneous segment shapes {dict(shapes)}")
                if dcfg.length_policy == "crop_min":
                    T_min = min(s.shape[1] for s in segs)
                    keep = [i for i, s in enumerate(segs) if s.shape[0] == C_mode]
                    for i in range(len(segs)):
                        if i not in set(keep):
                            malformed.append({"file": kept[i], "reason": f"channel count {segs[i].shape[0]} != {C_mode}"})
                    segs = [segs[i][:, :T_min] for i in keep]
                    labels = [labels[i] for i in keep]; kept = [kept[i] for i in keep]; pidx = [pidx[i] for i in keep]
                else:
                    keep = [i for i, s in enumerate(segs) if s.shape == (C_mode, T_mode)]
                    for i in range(len(segs)):
                        if segs[i].shape != (C_mode, T_mode):
                            malformed.append({"file": kept[i], "reason": f"shape {segs[i].shape} != {(C_mode, T_mode)}"})
                    segs = [segs[i] for i in keep]
                    labels = [labels[i] for i in keep]; kept = [kept[i] for i in keep]; pidx = [pidx[i] for i in keep]
            X = np.ascontiguousarray(np.stack(segs).astype(np.float32))
            y = np.asarray(labels, dtype=np.int64)
            names = [os.path.basename(p) for p in kept]
            pair_index = np.asarray(pidx, dtype=np.int64)
            if dcfg.use_cache:
                np.savez(cache, X=X, y=y, names=np.array(names), pair_index=pair_index,
                         groups=np.array([]), malformed=json.dumps(malformed),
                         unlabeled=json.dumps(unlabeled), fs_found=np.array(fs_found),
                         ch_names=np.array(ch_names or []))
                log(f"Cached arrays to {cache}")
        if (malformed or unlabeled) and dcfg.strict:
            raise RuntimeError(f"strict=True and {len(malformed)} malformed / {len(unlabeled)} unlabeled files")

    # ---- sampling frequency -------------------------------------------------
    if fs_found:
        vals = Counter(round(f, 3) for f in fs_found)
        fs = float(vals.most_common(1)[0][0])
        fs_source = f"file metadata ({dict(vals)})"
        if len(vals) > 1:
            log(f"WARNING: multiple sampling rates found {dict(vals)}; using {fs}")
    else:
        fs = float(dcfg.fs)
        fs_source = "config (not stored in files)"

    # ---- duplicate groups (computed on the full set, cached) -----------------
    groups = None
    if groups_cached is not None and groups_cached.size == len(y):
        groups = groups_cached.astype(np.int64)
    if groups is None:
        log(f"Running duplicate detection (mode={dcfg.dup_check}) ...")
        groups, _ = find_duplicate_groups(X, mode=dcfg.dup_check, block=dcfg.dup_block)
        if not os.path.isfile(root) and dcfg.use_cache:
            z = dict(np.load(cache, allow_pickle=True))
            z["groups"] = groups
            np.savez(cache, **z)
    clusters = defaultdict(list)
    for i, g in enumerate(groups):
        clusters[int(g)].append(i)
    dup_clusters = [[names[i] for i in v] for v in clusters.values() if len(v) > 1]

    # ---- patient ids ---------------------------------------------------------
    pmap = load_patient_map(dcfg.patient_map)
    patient_ids = None
    if pmap is not None:
        missing = [n for n in names if os.path.splitext(n)[0] not in pmap]
        if missing:
            raise ValueError(f"patient_map is missing {len(missing)} files, e.g. {missing[:3]}")
        patient_ids = np.array([pmap[os.path.splitext(n)[0]] for n in names])

    # ---- channel handling ------------------------------------------------------
    C = X.shape[1]
    ch_names = ch_names if ch_names and len(ch_names) == C else [f"ch{c}" for c in range(C)]
    if dcfg.channel_mode == "select":
        sel = list(dcfg.channels or range(C))
        X = np.ascontiguousarray(X[:, sel])
        ch_names = [ch_names[c] for c in sel]
    elif dcfg.channel_mode == "average":
        X = X.mean(axis=1, keepdims=True)
        ch_names = ["mean"]

    # ---- optional stratified subset (smoke tests) ------------------------------
    if dcfg.max_samples and dcfg.max_samples < len(y):
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(y)), train_size=int(dcfg.max_samples),
                                  stratify=y, random_state=int(cfg.seed))
        idx = np.sort(idx)
        X, y, groups = X[idx], y[idx], groups[idx]
        names = [names[i] for i in idx]
        pair_index = pair_index[idx]
        if patient_ids is not None:
            patient_ids = patient_ids[idx]
        log(f"Using stratified subset of {len(y)} samples (data.max_samples)")

    report = build_report(X, y, names, fs, fs_source, ch_names, patient_ids, malformed,
                          unlabeled, dup_clusters, n_files_total)
    data = EEGData(X=X, y=y, names=names, pair_index=pair_index, patient_ids=patient_ids,
                   fs=fs, fs_source=fs_source, channel_names=ch_names, groups=groups,
                   report=report)
    print_report(report, log)
    return data


def build_report(X, y, names, fs, fs_source, ch_names, patient_ids, malformed, unlabeled,
                 dup_clusters, n_files_total) -> Dict[str, Any]:
    N, C, T = X.shape
    rep: Dict[str, Any] = {
        "n_files_discovered": int(n_files_total),
        "n_samples": int(N),
        "n_channels": int(C),
        "channel_names": list(ch_names),
        "n_timepoints": int(T),
        "sampling_rate_hz": fs,
        "sampling_rate_source": fs_source,
        "duration_s": round(T / fs, 4),
        "tensor_shape_[N,C,T]": [int(N), int(C), int(T)],
        "class_distribution": {"non_focal(0)": int((y == 0).sum()), "focal(1)": int((y == 1).sum())},
        "patient_ids_available": patient_ids is not None,
        "patient_distribution": (dict(Counter(map(str, patient_ids))) if patient_ids is not None
                                 else "NOT AVAILABLE (patient identity not provided in the files)"),
        "malformed_files": malformed,
        "unlabeled_files": unlabeled,
        "duplicate_clusters": dup_clusters,
        "amplitude": {},
    }
    for lab, name in ((1, "focal"), (0, "non_focal")):
        sel = X[y == lab]
        if len(sel) == 0:
            continue
        std = sel.std(axis=2)
        amax = np.abs(sel).max(axis=2)
        rep["amplitude"][name] = {
            "per_channel_std_mean": std.mean(0).round(3).tolist(),
            "per_channel_std_median": np.median(std, 0).round(3).tolist(),
            "per_channel_absmax_median": np.median(amax, 0).round(3).tolist(),
            "flat_channels": int((std < 1e-8).sum()),
        }
    return rep


def print_report(rep: Dict[str, Any], log) -> None:
    log("-" * 70)
    log("DATASET REPORT")
    log(f"  samples: {rep['n_samples']} | tensor [N,C,T] = {rep['tensor_shape_[N,C,T]']}")
    log(f"  channels: {rep['n_channels']} {rep['channel_names']}")
    log(f"  sampling rate: {rep['sampling_rate_hz']} Hz ({rep['sampling_rate_source']})")
    log(f"  duration: {rep['duration_s']} s")
    log(f"  classes: {rep['class_distribution']}")
    log(f"  patients: {rep['patient_distribution']}")
    log(f"  malformed files: {len(rep['malformed_files'])} | unlabeled: {len(rep['unlabeled_files'])}")
    for m in rep["malformed_files"][:10]:
        log(f"    MALFORMED {m['file']}: {m['reason']}")
    log(f"  duplicate clusters (kept in same fold): {len(rep['duplicate_clusters'])} {rep['duplicate_clusters'][:5]}")
    for k, v in rep["amplitude"].items():
        log(f"  amplitude[{k}]: std mean {v['per_channel_std_mean']} | absmax median {v['per_channel_absmax_median']}")
    log("-" * 70)


def subset_arrays(data: EEGData, idx: np.ndarray) -> EEGData:
    return EEGData(X=data.X[idx], y=data.y[idx], names=[data.names[i] for i in idx],
                   pair_index=data.pair_index[idx],
                   patient_ids=None if data.patient_ids is None else data.patient_ids[idx],
                   fs=data.fs, fs_source=data.fs_source, channel_names=data.channel_names,
                   groups=data.groups[idx], report=data.report)


def load_inference_files(paths: List[str], channel_mode: str = "all", channels=None):
    """Load unlabeled files for inference -> (X [N,C,T], names, fs or None)."""
    segs, names, fss = [], [], []
    for p in paths:
        path, seg, fs, _, err = _load_one(p)
        if err:
            raise RuntimeError(f"Could not read {p}: {err}")
        segs.append(seg); names.append(os.path.basename(p))
        if fs:
            fss.append(fs)
    X = np.stack(segs).astype(np.float32)
    if channel_mode == "select":
        X = X[:, list(channels)]
    elif channel_mode == "average":
        X = X.mean(1, keepdims=True)
    return X, names, (fss[0] if fss else None)
