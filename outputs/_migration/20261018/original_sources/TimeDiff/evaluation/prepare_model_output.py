"""把任意模型的輸出轉成 evaluation framework 的標準格式。

標準格式（wide，log return of close）：
    outputs/<tag>/paths/returns/path_XX_returns.csv
    date,AAPL_rclose,AMGN_rclose,...,WMT_rclose      # 10 欄 + date
    2018-01-02,...                                    # 1,761 列，日期須與真實交易日完全一致

支援的輸入
----------
  --kind   return | price
  --layout wide   | long

  wide  + return : date + {TICKER}_rclose  或  {TICKER}_return / {TICKER}
  wide  + price  : date + {TICKER}_Close   或  {TICKER}_close / {TICKER}
  long  + return : date, tic, (rclose | return | ret)
  long  + price  : date, tic, (close | Close | price)

price -> return 用 log(C_t / C_{t-1})。若價格序列的第一天就是 evaluation 起始日，
會以**真實資料**在該日前一交易日的 Close 作為 seed（與 walk_forward_dow10.py 的作法一致，
且所有模型共用同一個真實 seed，不會造成模型間的不公平）。

用法
----
  python -m evaluation.prepare_model_output --tag nsdiff \
      --input /path/to/nsdiff_output.csv --kind price --layout long

  # 多路徑放在同一個檔（有一欄標示路徑編號）
  python -m evaluation.prepare_model_output --tag armd \
      --input armd.csv --kind return --layout long --path-col path

  # 多路徑分散在多個檔
  python -m evaluation.prepare_model_output --tag dva \
      --input "dva_run_*.csv" --kind return --layout wide

轉完後把 tag 加進 evaluation/config.py 的 MODELS_WHITELIST，再跑
  python -m evaluation.run_all
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import loaders


RET_ALIASES = ["rclose", "return", "ret", "log_return", "logret"]
PX_ALIASES = ["close", "Close", "adj_close", "price"]


def _find_col(cols, ticker, aliases):
    """在 wide 表中找出某個 ticker 的欄位名。"""
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        for cand in (f"{ticker}_{a}", f"{a}_{ticker}"):
            if cand.lower() in lower:
                return lower[cand.lower()]
    if ticker in cols:                      # 只有 ticker 名當欄名
        return ticker
    if ticker.lower() in lower:
        return lower[ticker.lower()]
    return None


def _long_to_wide(df, value_aliases, value_col=None):
    """long (date, tic, value) -> wide (date x ticker)。"""
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("datetime") or cols.get("time")
    tcol = cols.get("tic") or cols.get("ticker") or cols.get("symbol")
    if dcol is None or tcol is None:
        raise ValueError(f"long format 需要 date 與 tic 欄，實際欄位: {list(df.columns)}")
    if value_col is not None:
        if value_col not in df.columns:
            raise ValueError(f"--value-col '{value_col}' 不在欄位中: {list(df.columns)}")
        vcol = value_col
    else:
        vcol = None
        for a in value_aliases:
            if a.lower() in cols:
                vcol = cols[a.lower()]
                break
    if vcol is None:
        raise ValueError(f"找不到數值欄（試過 {value_aliases}），實際欄位: {list(df.columns)}\n"
                         f"若數值欄名稱不在上列（例如 return 值放在名為 'close' 的欄位），"
                         f"請用 --value-col 明確指定。")
    w = df.pivot(index=dcol, columns=tcol, values=vcol)
    w.index = pd.to_datetime(w.index)
    return w.sort_index()


def _wide_normalize(df, aliases, value_col=None):
    """wide -> (T, 10) DataFrame，欄序 = TICKERS。value_col 在 wide 版不適用，僅為介面一致。"""
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or cols.get("datetime") or cols.get("time")
    if dcol is None:
        raise ValueError(f"wide format 需要 date 欄，實際欄位: {list(df.columns)[:8]}")
    out = pd.DataFrame(index=pd.to_datetime(df[dcol]))
    missing = []
    for t in C.TICKERS:
        c = _find_col(df.columns, t, aliases)
        if c is None:
            missing.append(t)
        else:
            out[t] = df[c].values
    if missing:
        raise ValueError(f"找不到這些 ticker 的欄位: {missing}\n"
                         f"可用欄位: {list(df.columns)[:20]}")
    return out.sort_index()


def price_to_return(px, start, end):
    """log(C_t / C_{t-1})。第一天缺前值時，用真實資料前一交易日的 Close 當 seed。"""
    px = px.sort_index()
    start_ts = pd.Timestamp(start)
    if px.index[0] >= start_ts:
        # 需要一個 seed：取真實資料在 start 之前最後一個交易日的 Close
        ohlc = pd.read_csv(os.path.join(C.DATA_DIR, "dow10_wf_ohlc.csv"),
                           parse_dates=["date"]).set_index("date")
        prior = ohlc[ohlc.index < start_ts]
        if prior.empty:
            raise ValueError(f"真實資料中找不到 {start} 之前的交易日可作為 seed")
        seed_date = prior.index[-1]
        seed = prior.loc[seed_date, [f"{t}_Close" for t in C.TICKERS]].values.astype(float)
        print(f"  價格序列第一天就是 {px.index[0].date()}，以真實 {seed_date.date()} 的 Close 作為 seed")
        px = pd.concat([pd.DataFrame([seed], index=[seed_date], columns=C.TICKERS), px])
    r = np.log(px / px.shift(1)).dropna(how="all")
    return r[(r.index >= start_ts) & (r.index <= pd.Timestamp(end))]


def convert_one(df, kind, layout, start, end, value_col=None):
    """單一路徑 -> (T, 10) 的 rclose DataFrame。"""
    aliases = RET_ALIASES if kind == "return" else PX_ALIASES
    if layout == "long":
        w = _long_to_wide(df, aliases, value_col=value_col)
        missing = [t for t in C.TICKERS if t not in w.columns]
        if missing:
            raise ValueError(f"long 表缺少這些 ticker: {missing}")
        w = w[C.TICKERS]
    else:
        w = _wide_normalize(df, aliases, value_col=value_col)

    if kind == "price":
        w = price_to_return(w, start, end)
    else:
        w = w[(w.index >= pd.Timestamp(start)) & (w.index <= pd.Timestamp(end))]
    return w


def validate(w, real_dates, tag, path_id):
    """對齊檢查。任何一項不符就 raise —— 不容許靜默對齊。"""
    problems = []
    if len(w) != len(real_dates):
        problems.append(f"列數 {len(w)} != 真實交易日 {len(real_dates)}")
    extra = sorted(set(w.index) - set(real_dates))
    miss = sorted(set(real_dates) - set(w.index))
    if extra:
        problems.append(f"多出 {len(extra)} 個非交易日，例如 {[str(d.date()) for d in extra[:3]]}")
    if miss:
        problems.append(f"缺少 {len(miss)} 個交易日，例如 {[str(d.date()) for d in miss[:3]]}")
    bad = int((~np.isfinite(w.values)).sum())
    if bad:
        problems.append(f"含 {bad} 個 NaN/Inf")
    if problems:
        raise ValueError(f"[{tag} path {path_id}] 對齊失敗：\n    " + "\n    ".join(problems))
    # 合理性提醒（不阻擋）
    s = float(np.nanstd(w.values))
    if s > 0.5:
        print(f"  ⚠ path {path_id}: 日報酬標準差 {s:.3f} 異常大，確認單位是 log return 而非百分比?")
    if s < 1e-5:
        print(f"  ⚠ path {path_id}: 日報酬標準差 {s:.2e} 異常小，確認不是常數序列?")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="模型代號，會建立 outputs/<tag>/")
    ap.add_argument("--input", required=True, help="輸入檔（可用 glob，如 'run_*.csv'）")
    ap.add_argument("--kind", choices=["return", "price"], required=True)
    ap.add_argument("--layout", choices=["long", "wide"], required=True)
    ap.add_argument("--path-col", default=None,
                    help="單一檔內含多路徑時，標示路徑編號的欄位名")
    ap.add_argument("--value-col", default=None,
                    help="明確指定數值欄名（long format）。當數值欄的名稱會誤導時使用，"
                         "例如 return 值放在名為 'close' 的欄位裡")
    ap.add_argument("--start", default=C.PRIMARY_EVAL_START)
    ap.add_argument("--end", default=C.PRIMARY_EVAL_END)
    ap.add_argument("--dry-run", action="store_true", help="只驗證，不寫檔")
    a = ap.parse_args(argv)

    real_dates, _ = loaders.load_real(a.start, a.end)
    print(f"目標：{a.start} ~ {a.end}，{len(real_dates)} 個交易日，{C.N_TICKER} 檔")

    files = sorted(glob.glob(a.input)) or [a.input]
    print(f"輸入：{len(files)} 個檔案")

    frames = []
    for f in files:
        raw = pd.read_csv(f)
        if a.path_col:
            if a.path_col not in raw.columns:
                raise ValueError(f"{f} 沒有 --path-col 指定的欄位 '{a.path_col}'")
            for pid, sub in raw.groupby(a.path_col):
                frames.append(convert_one(sub.drop(columns=[a.path_col]),
                                          a.kind, a.layout, a.start, a.end, a.value_col))
        else:
            frames.append(convert_one(raw, a.kind, a.layout, a.start, a.end, a.value_col))

    for i, w in enumerate(frames):
        validate(w, real_dates, a.tag, i)
    print(f"✓ {len(frames)} 條路徑全部通過對齊檢查")

    if a.dry_run:
        print("(--dry-run，未寫檔)")
        return frames

    out_dir = os.path.join(C.OUTPUTS_DIR, a.tag, "paths", "returns")
    os.makedirs(out_dir, exist_ok=True)
    for i, w in enumerate(frames):
        out = w.copy()
        out.columns = [f"{t}_rclose" for t in C.TICKERS]
        out.index.name = "date"
        out.reset_index().to_csv(os.path.join(out_dir, f"path_{i:02d}_returns.csv"), index=False)
    print(f"✓ 已寫入 {out_dir}/path_XX_returns.csv")
    print(f"\n下一步：")
    print(f"  1) 把 \"{a.tag}\" 加進 evaluation/config.py 的 MODELS_WHITELIST")
    print(f"     （並在 MODEL_DISPLAY_NAMES 給它一個論文用的顯示名稱）")
    print(f"  2) python -m evaluation.run_all")
    return frames


if __name__ == "__main__":
    main()
