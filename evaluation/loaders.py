"""統一 preprocessing layer：把不同 diffusion model 的輸出格式轉成同一種形狀。

**不修改任何原始生成資料**，只在讀取時做格式轉換與對齊檢查。

輸出的標準形狀：
    real : np.ndarray (T, 10)      rclose，欄序 = config.TICKERS
    syn  : np.ndarray (P, T, 10)   P = 路徑數（單路徑模型 P=1）
    dates: pd.DatetimeIndex (T,)
"""
import os
import glob
import json
import warnings

import numpy as np
import pandas as pd

from . import config as C
from output_layout import model_directories


# ===========================================================
# 真實資料
# ===========================================================
def load_real(start=None, end=None):
    """回傳 (dates, R) — R shape (T, 10) 的 rclose，欄序 = TICKERS。"""
    df = pd.read_csv(C.REAL_RETURNS_CSV, parse_dates=["date"]).set_index("date")
    R = df[[f"{t}_rclose" for t in C.TICKERS]].copy()
    R.columns = C.TICKERS
    if start is not None:
        R = R[R.index >= pd.Timestamp(start)]
    if end is not None:
        R = R[R.index <= pd.Timestamp(end)]
    if R.isna().any().any():
        raise ValueError(f"真實資料在 {start}~{end} 有缺值，需先處理")
    return R.index, R.values.astype(np.float64)


# ===========================================================
# 模型掃描
# ===========================================================
def discover_models(outputs_dir=None, verbose=True):
    """掃描分類後的模型目錄（並相容舊扁平目錄），依檔案辨識 protocol。

    回傳 list[dict]，每個 dict 描述一個候選模型：
        tag, protocol, n_paths, date_range, files, eligible_for_primary, reason
    """
    outputs_dir = outputs_dir or C.OUTPUTS_DIR
    found = []

    for directory in model_directories(outputs_dir):
        d = str(directory)
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)

        # ── Family B：walk-forward rolling generation ──
        ret_files = sorted(glob.glob(os.path.join(d, "paths", "returns", "path_*_returns.csv")))
        if not ret_files:   # 相容舊的扁平結構
            ret_files = sorted(glob.glob(os.path.join(d, "paths", "path_*_returns.csv")))
        legacy_single = os.path.join(d, "wf_generated_returns.csv")

        if ret_files:
            dates = pd.read_csv(ret_files[0], usecols=["date"], parse_dates=["date"])["date"]
            found.append(dict(
                tag=tag, protocol="walk_forward", n_paths=len(ret_files),
                start=str(dates.iloc[0].date()), end=str(dates.iloc[-1].date()),
                T=len(dates), files=ret_files, has_dates=True))
            continue
        if os.path.exists(legacy_single):
            dates = pd.read_csv(legacy_single, usecols=["date"], parse_dates=["date"])["date"]
            found.append(dict(
                tag=tag, protocol="walk_forward", n_paths=1,
                start=str(dates.iloc[0].date()), end=str(dates.iloc[-1].date()),
                T=len(dates), files=[legacy_single], has_dates=True))
            continue

        # ── Family A：單次 2024 生成（無 date 欄）──
        f2024 = os.path.join(d, "dow10_generated_returns_2024.csv")
        if os.path.exists(f2024):
            n = sum(1 for _ in open(f2024)) - 1
            found.append(dict(
                tag=tag, protocol="single_shot_2024", n_paths=1,
                start="2024-01-02", end="2024-12-31", T=n,
                files=[f2024], has_dates=False))
            continue

        # ── Family C：10yr ablation（非 Dow 10）──
        f10 = os.path.join(d, "generated_log_returns_10yr.csv")
        if os.path.exists(f10):
            cols = pd.read_csv(f10, nrows=1).columns
            tick = sorted({c.split("_")[0] for c in cols})
            found.append(dict(
                tag=tag, protocol="other_universe", n_paths=1,
                start=None, end=None, T=None, files=[f10],
                has_dates=False, universe=tick))

    # ── 判定是否可進入主表 ──
    ps, pe = pd.Timestamp(C.PRIMARY_EVAL_START), pd.Timestamp(C.PRIMARY_EVAL_END)
    for m in found:
        if m["protocol"] == "other_universe":
            m["eligible_for_primary"] = False
            m["reason"] = f"ticker universe 非 Dow 10（{','.join(m.get('universe', []))}）"
        elif m["protocol"] == "single_shot_2024":
            m["eligible_for_primary"] = False
            m["reason"] = ("只有 2024 輸出，未涵蓋 primary window；"
                           "需以 walk-forward 協定重新生成（Option 1）")
        elif pd.Timestamp(m["start"]) > ps or pd.Timestamp(m["end"]) < pe:
            m["eligible_for_primary"] = False
            m["reason"] = f"日期範圍 {m['start']}~{m['end']} 未涵蓋 primary window"
        else:
            m["eligible_for_primary"] = True
            m["reason"] = ""

    if verbose:
        print(f"掃描 {outputs_dir}：找到 {len(found)} 個候選目錄")
        for proto in ["walk_forward", "single_shot_2024", "other_universe"]:
            sub = [m for m in found if m["protocol"] == proto]
            if sub:
                print(f"  [{proto}] {len(sub)} 個")
    return found


def select_models(discovered, whitelist=None, verbose=True):
    """依 config 的 whitelist / protocol 規則挑出要評估的模型。"""
    whitelist = whitelist if whitelist is not None else C.MODELS_WHITELIST
    sel = []
    for m in discovered:
        if whitelist is not None and m["tag"] not in whitelist:
            continue
        if C.REQUIRE_WALKFORWARD_PROTOCOL and not m["eligible_for_primary"]:
            warnings.warn(f"{m['tag']} 不符合 primary window 要求：{m['reason']}")
            continue
        m = dict(m)
        m["display"] = C.MODEL_DISPLAY_NAMES.get(m["tag"], m["tag"])
        sel.append(m)
    if verbose:
        print(f"選入評估的模型：{[m['display'] for m in sel]}")
    return sel


# ===========================================================
# Synthetic 載入
# ===========================================================
def load_synthetic(model, start=None, end=None, real_dates=None):
    """回傳 (dates, S) — S shape (P, T, 10)。

    walk_forward     : 直接依 date 欄篩選
    single_shot_2024 : 無 date 欄，以序位對映到真實 2024 交易日（並 assert 長度）
    """
    proto = model["protocol"]

    if proto == "walk_forward":
        paths, dates = [], None
        for f in model["files"]:
            df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            if start is not None:
                df = df[df.index >= pd.Timestamp(start)]
            if end is not None:
                df = df[df.index <= pd.Timestamp(end)]
            x = df[[f"{t}_rclose" for t in C.TICKERS]].values.astype(np.float64)
            if dates is None:
                dates = df.index
            elif not dates.equals(df.index):
                raise ValueError(f"{f} 的日期與其他路徑不一致")
            paths.append(x)
        return dates, np.stack(paths, axis=0)

    if proto == "single_shot_2024":
        if real_dates is None:
            raise ValueError("single_shot_2024 需要 real_dates 才能對映序位")
        d2024 = real_dates[(real_dates >= pd.Timestamp("2024-01-01")) &
                           (real_dates <= pd.Timestamp("2024-12-31"))]
        df = pd.read_csv(model["files"][0])
        if len(df) != len(d2024):
            raise ValueError(f"{model['tag']}: 生成 {len(df)} 列 != 真實 2024 交易日 {len(d2024)}")
        df.index = d2024
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        x = df[[f"{t}_rclose" for t in C.TICKERS]].values.astype(np.float64)
        return df.index, x[np.newaxis]

    raise ValueError(f"不支援的 protocol: {proto}")


# ===========================================================
# 對齊檢查（§25）
# ===========================================================
def assert_alignment(real_dates, syn_dates, real, syn, model_tag=""):
    """日期、ticker、長度三重檢查。任何一項不符就 raise，不容許靜默對齊。"""
    if len(real_dates) != len(syn_dates):
        raise AssertionError(f"{model_tag}: 長度不符 real={len(real_dates)} syn={len(syn_dates)}")
    if not pd.DatetimeIndex(real_dates).equals(pd.DatetimeIndex(syn_dates)):
        diff = set(map(str, real_dates)) ^ set(map(str, syn_dates))
        raise AssertionError(f"{model_tag}: 日期不對齊，差異 {len(diff)} 天，例如 {sorted(diff)[:3]}")
    if real.shape[1] != C.N_TICKER or syn.shape[-1] != C.N_TICKER:
        raise AssertionError(f"{model_tag}: ticker 數不符")
    if not np.isfinite(real).all():
        raise AssertionError("真實資料含 NaN/Inf")
    if not np.isfinite(syn).all():
        n = int((~np.isfinite(syn)).sum())
        raise AssertionError(f"{model_tag}: synthetic 含 {n} 個 NaN/Inf")
    return True


def window_slice(dates, X, start, end):
    """依日期切窗。X 可為 (T,10) 或 (P,T,10)。"""
    dates = pd.DatetimeIndex(dates)
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    if X.ndim == 2:
        return dates[mask], X[mask]
    return dates[mask], X[:, mask]
