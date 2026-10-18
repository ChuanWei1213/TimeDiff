"""Distribution, pairing, and within-condition path diagnostics (no PPO)."""
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd

from data_provider.dow10_cond import CLOSE_INDICES, portfolio_log_returns, portfolio_statistics
from dow10_cond_experiment import mean_pairwise_corr


def path_statistics(raw):
    rc = np.asarray(raw[..., CLOSE_INDICES], dtype=np.float64)
    rp = portfolio_log_returns(rc)
    values = np.concatenate([np.ones((len(rc), 1)), np.exp(np.cumsum(rp, axis=1))], axis=1)
    stats = portfolio_statistics(rc)
    a, b = np.square(rp[:, :-1]), np.square(rp[:, 1:])
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(a).sum(axis=1) * np.square(b).sum(axis=1))
    squared_acf = np.divide((a * b).sum(axis=1), denominator,
                            out=np.full(len(rc), np.nan), where=denominator > 0)
    return pd.DataFrame({
        "return_21d": stats[:, 0], "volatility_ann": stats[:, 1],
        "max_drawdown": (values / np.maximum.accumulate(values, axis=1) - 1).min(axis=1),
        "mean_asset_correlation": mean_pairwise_corr(rc),
        "max_abs_asset_daily_log_return": np.abs(rc).max(axis=(1, 2)),
        "portfolio_squared_return_lag1_correlation": squared_acf,
    }), values


def diversity_statistics(raw, references, path_metadata, feature_scale):
    """Measure shape diversity separately from deliberately fixed R/sigma labels.

    Ratios use the checkpoint's fixed training feature scale, not realized scale.
    Exact-copy checks only compare to the corresponding original future.
    """
    close = np.asarray(raw[..., CLOSE_INDICES], dtype=np.float64)
    ref = np.asarray(references[..., CLOSE_INDICES], dtype=np.float64)
    scale = np.asarray(feature_scale)[CLOSE_INDICES]
    normalized = close / scale
    rp = portfolio_log_returns(close)
    realized = portfolio_statistics(close)
    rows = []
    for window_id, group in path_metadata.groupby("window_id", sort=True):
        ids = group.index.to_numpy()
        rms, shape_rms, corr = [], [], []
        centered = normalized[ids] - normalized[ids].mean(axis=1, keepdims=True)
        hashes = [hashlib.sha256(close[j].tobytes()).hexdigest() for j in ids]
        for a, b in combinations(range(len(ids)), 2):
            rms.append(float(np.sqrt(np.mean((normalized[ids[a]] - normalized[ids[b]]) ** 2))))
            shape_rms.append(float(np.sqrt(np.mean((centered[a] - centered[b]) ** 2))))
            if rp[ids[a]].std() > 0 and rp[ids[b]].std() > 0:
                corr.append(float(np.corrcoef(rp[ids[a]], rp[ids[b]])[0, 1]))
        rows.append({
            "window_id": int(window_id), "anchor_date": group.anchor_date.iloc[0],
            "target_return": group.target_return.iloc[0],
            "target_volatility": group.target_volatility.iloc[0],
            "n_paths": len(ids), "n_unique_close_paths": len(set(hashes)),
            "within_condition_return_std": float(realized[ids, 0].std(ddof=1)) if len(ids) > 1 else np.nan,
            "within_condition_vol_std": float(realized[ids, 1].std(ddof=1)) if len(ids) > 1 else np.nan,
            "mean_pairwise_normalized_rmse": float(np.mean(rms)) if rms else np.nan,
            "mean_pairwise_centered_rmse": float(np.mean(shape_rms)) if shape_rms else np.nan,
            "mean_pairwise_portfolio_return_correlation": float(np.mean(corr)) if corr else np.nan,
            "mean_normalized_rmse_to_original_future": float(np.sqrt(np.mean((normalized[ids] - ref[ids] / scale) ** 2, axis=(1, 2))).mean()),
        })
    table = pd.DataFrame(rows)
    hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in close]
    original_error = np.max(np.abs(close - ref), axis=(1, 2))
    summary = {
        "n_paths": len(raw), "n_unique_close_paths": len(set(hashes)),
        "exact_duplicate_close_path_fraction": 1 - len(set(hashes)) / len(raw),
        "n_conditions_with_replicates": int((table.n_paths >= 2).sum()),
        "n_conditions_all_close_paths_identical": int(((table.n_paths >= 2) & (table.n_unique_close_paths == 1)).sum()),
        "mean_pairwise_normalized_rmse": float(table.mean_pairwise_normalized_rmse.mean()),
        "median_pairwise_centered_rmse": float(table.mean_pairwise_centered_rmse.median()),
        "median_pairwise_portfolio_return_correlation": float(table.mean_pairwise_portfolio_return_correlation.median()),
        "original_future_copy_count_at_1e_minus_7": int((original_error <= 1e-7).sum()),
        "limitations": [
            "Nonzero differences alone do not establish realistic diversity.",
            "Small R/sigma dispersion is expected under fixed conditions; inspect path-shape differences.",
            "Near-copy check covers only the paired original future, not every generator-training window.",
            "Overlapping windows and repeated conditions are dependent.",
        ],
    }
    return table, summary


def _quantile_rows(label, values):
    rows = []
    for k, name in enumerate(("return_21d", "volatility_ann")):
        x = np.asarray(values)[:, k]
        rows.append({"source": label, "metric": name, "n": len(x),
                     "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                     **{f"q{int(q * 100):02d}": float(np.quantile(x, q))
                        for q in (.01, .10, .30, .50, .70, .90, .99)}})
    return rows


def write_diagnostics(out, plan, raw, references, path_metadata, feature_scale, condition_scale,
                      high_clip, low_clip):
    metrics, values = path_statistics(raw)
    ref_metrics, ref_values = path_statistics(references)
    targets = path_metadata[["target_return", "target_volatility"]].to_numpy()
    realized = metrics[["return_21d", "volatility_ann"]].to_numpy()
    errors = realized - targets
    per_path = pd.concat([path_metadata.reset_index(drop=True), metrics], axis=1)
    per_path["return_error"] = errors[:, 0]
    per_path["volatility_error"] = errors[:, 1]
    per_path.to_csv(out / "path_metrics.csv", index=False)
    ref_metrics.to_csv(out / "paired_real_metrics.csv", index=False)
    diversity, div_summary = diversity_statistics(raw, references, path_metadata, feature_scale)
    diversity.to_csv(out / "diversity_by_condition.csv", index=False)
    train = plan.population[["target_return", "target_volatility"]].to_numpy()
    draws = plan.draws[["target_return", "target_volatility"]].to_numpy()
    quantiles = pd.DataFrame(_quantile_rows("training_population", train)
                             + _quantile_rows("sampled_conditions", draws)
                             + _quantile_rows("generated_paths", realized))
    quantiles.to_csv(out / "distribution_quantiles.csv", index=False)
    cutoff = plan.metadata["tail_return_cutoff"]
    group_errors = []
    for label, mask in [("all", np.ones(len(raw), dtype=bool)),
                        ("target_tail", targets[:, 0] <= cutoff),
                        ("target_non_tail", targets[:, 0] > cutoff)]:
        if not mask.any():
            continue
        e = errors[mask]
        group_errors.append({
            "group": label, "n_paths": int(mask.sum()),
            "return_bias": float(e[:, 0].mean()), "return_mae": float(np.abs(e[:, 0]).mean()),
            "volatility_bias": float(e[:, 1].mean()), "volatility_mae": float(np.abs(e[:, 1]).mean()),
            "standardized_condition_mse": float(np.square(e / condition_scale).mean()),
            "hit_1pp_return_2pp_vol": float(((np.abs(e[:, 0]) <= .01) & (np.abs(e[:, 1]) <= .02)).mean()),
            "hit_2pp_return_5pp_vol": float(((np.abs(e[:, 0]) <= .02) & (np.abs(e[:, 1]) <= .05)).mean()),
        })
    pd.DataFrame(group_errors).to_csv(out / "condition_errors.csv", index=False)
    # Retain joint-bin comparisons: matching marginals alone is insufficient.
    bins = [np.r_[-np.inf, np.unique(np.quantile(train[:, k], [.1, .3, .5, .7, .9])), np.inf]
            for k in range(2)]
    joint, joint_histograms = [], {}
    for label, data, weights in [("expected_mixture", train, plan.population.mixture_probability.to_numpy()),
                                 ("sampled_conditions", draws, np.full(len(draws), 1 / len(draws))),
                                 ("generated_paths", realized, np.full(len(raw), 1 / len(raw)))]:
        hist = np.histogram2d(data[:, 0], data[:, 1], bins=bins, weights=weights)[0]
        joint_histograms[label] = hist
        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                joint.append({"source": label, "return_bin": i, "volatility_bin": j, "mass": hist[i, j]})
    pd.DataFrame(joint).to_csv(out / "joint_distribution_bins.csv", index=False)
    years = []
    for year in sorted(set(plan.population.target_start.str[:4])):
        pop_mask = plan.population.target_start.str[:4] == year
        draw_mask = plan.draws.target_start.str[:4] == year
        selected = plan.draws.loc[draw_mask]
        years.append({"target_start_year": year, "n_population": int(pop_mask.sum()),
                      "expected_mixture_mass": float(plan.population.loc[pop_mask, "mixture_probability"].sum()),
                      "n_draws": len(selected), "n_unique_windows": selected.window_id.nunique(),
                      "n_full_pool_draws": int((selected.source_pool == "full").sum()),
                      "n_tail_pool_draws": int((selected.source_pool == "tail").sum())})
    pd.DataFrame(years).to_csv(out / "sampling_by_year.csv", index=False)
    summary = {
        "status": "generated_and_diagnosed", "sampling": plan.metadata,
        "realized_tail_share": float((realized[:, 0] <= cutoff).mean()),
        "target_tail_share": float((targets[:, 0] <= cutoff).mean()),
        "return_vol_correlation": {
            label: float(np.corrcoef(data.T)[0, 1])
            for label, data in [("training_population", train), ("sampled_conditions", draws), ("generated_paths", realized)]},
        "condition_errors": group_errors, "diversity": div_summary,
        "joint_binned_total_variation": {
            "sampled_vs_expected_mixture": float(np.abs(joint_histograms["sampled_conditions"] - joint_histograms["expected_mixture"]).sum() / 2),
            "generated_vs_sampled_conditions": float(np.abs(joint_histograms["generated_paths"] - joint_histograms["sampled_conditions"]).sum() / 2),
        },
        "high_clip_fraction": float(high_clip.mean()), "low_clip_fraction": float(low_clip.mean()),
        "path_comparison": {
            name: {"generated_mean": float(metrics[name].mean()), "paired_real_mean": float(ref_metrics[name].mean())}
            for name in ("max_drawdown", "mean_asset_correlation", "max_abs_asset_daily_log_return",
                         "portfolio_squared_return_lag1_correlation")},
        "joint_bin_edges": [[None if not np.isfinite(x) else float(x) for x in b] for b in bins],
        "caveats": [
            "This is an in-training-support generation diagnostic, not out-of-sample validation or PPO evidence.",
            "No paths were rejected or retuned based on realized conditions.",
            "The original +/-2pp return and +/-5pp volatility tolerance is descriptive, not an acceptance standard.",
            "Multiple generated paths per draw do not increase the number of independently sampled conditions.",
            "Lag-1 squared-return correlation within a 21-day path is noisy; it is descriptive only.",
        ],
    }
    # JSON null represents undefined correlations (e.g. a tiny constant fixture).
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    summary = finite_json(summary)
    (out / "diagnostics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _plots(out, plan, realized, values, ref_values)
    e = group_errors[0]
    tail_label = f"PR{100 * plan.metadata['tail_quantile']:g}"
    text = [
        "# Dow 10 聯合條件取樣：生成診斷", "",
        f"已生成 **{len(raw):,} 條** 21 日路徑；{len(plan.draws):,} 次條件抽樣，涵蓋 {plan.metadata['n_unique_windows']:,} 個不同歷史視窗。",
        "本輪未執行 PPO；結果只衡量取樣、生成條件與路徑多樣性。", "",
        f"- 條件母體：{plan.metadata['train_start']}～{plan.metadata['train_end']}，{len(train):,} 個完整視窗。",
        f"- 來源池：完整分布 {plan.metadata['n_full_draws']} 組、{tail_label} 以下 {plan.metadata['n_tail_draws']} 組；完整池也包含尾部。",
        f"- {tail_label} 報酬門檻：{cutoff:.4%}。理論尾部占比 {plan.metadata['expected_sampled_tail_share']:.2%}；抽中條件 {summary['target_tail_share']:.2%}；實際生成 {summary['realized_tail_share']:.2%}。",
        f"- 報酬條件 MAE：{e['return_mae'] * 100:.3f} 個百分點；年化波動 MAE：{e['volatility_mae'] * 100:.3f} 個百分點。",
        f"- 報酬偏差：{e['return_bias'] * 100:+.3f} 個百分點；年化波動偏差：{e['volatility_bias'] * 100:+.3f} 個百分點。",
        f"- 收盤路徑完全重複比例：{div_summary['exact_duplicate_close_path_fraction']:.2%}；同條件所有路徑完全相同的組數：{div_summary['n_conditions_all_close_paths_identical']}。",
        f"- 同條件兩兩標準化 RMSE 平均：{div_summary['mean_pairwise_normalized_rmse']:.4f}；去除每條路徑各資產均值後的 RMSE 組中位數：{div_summary['median_pairwise_centered_rmse']:.4f}。",
        "標準化使用 checkpoint 的訓練特徵尺度；RMSE 無單位。條件固定時，期末報酬與波動的低變異本身不代表路徑塌縮。", "",
        "![條件分布](condition_distributions.png)", "",
        "![同條件路徑](within_condition_paths.png)", "",
        "## 限制與來源", "",
        "- 重疊歷史視窗並非獨立事件；本批取樣比例是實驗設計，並非市場發生機率。",
        "- 零完全重複不能證明多樣性充分；請連同路徑形狀、原始歷史路徑及數值距離解讀。",
        "- 原始未來僅用於條件建構與事後診斷，未送進模型 decoder；所有生成路徑均保留，沒有依結果篩選。",
        "- 本次近似複製檢查只比對各自配對的真實未來，並非對生成器全部訓練資料進行最近鄰稽核。",
        "- 配對與機率：population.csv、sampling_plan.csv；逐路徑：path_metrics.csv；多樣性：diversity_by_condition.csv。",
        "- 完整來源、checkpoint／資料／程式雜湊與版本見 generation.json；分位數見 distribution_quantiles.csv；聯合分布見 joint_distribution_bins.csv。",
    ]
    (out / "README.md").write_text("\n".join(text) + "\n")
    return summary


def _plots(out, plan, realized, values, ref_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    for family in ("Arial Unicode MS", "PingFang TC", "Noto Sans CJK TC"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            break
        except ValueError:
            continue
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"訓練分布": "#a0a7b0", "抽中條件": "#275c96", "生成路徑": "#b06b22"}
    train = plan.population[["target_return", "target_volatility"]].to_numpy()
    draws = plan.draws[["target_return", "target_volatility"]].to_numpy()
    tail_label = f"訓練 PR{100 * plan.metadata['tail_quantile']:g}"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for k, title in enumerate(("21 日累積報酬", "年化波動率")):
        ax = axes[0, k]
        # ECDF shows all observations, including outliers, without bin choices.
        for label, data in [("訓練分布", train), ("抽中條件", draws), ("生成路徑", realized)]:
            x = np.sort(data[:, k])
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=label, color=colors[label])
        ax.set(xlabel=title, ylabel="累積比例", title=f"{title}的累積分布")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.15)
        if k == 0:
            ax.axvline(plan.metadata["tail_return_cutoff"], color="#444444", linestyle="--", linewidth=1, label=tail_label)
        ax.legend(fontsize=9)
    for ax, title, data in [(axes[1, 0], "抽中的報酬與波動配對", draws),
                            (axes[1, 1], "生成後的報酬與波動", realized)]:
        ax.scatter(train[:, 0], train[:, 1], s=8, alpha=.2, color=colors["訓練分布"], label="訓練分布")
        ax.scatter(data[:, 0], data[:, 1], s=10, alpha=.35, color=colors["抽中條件"] if title.startswith("抽") else colors["生成路徑"], label=title)
        ax.axvline(plan.metadata["tail_return_cutoff"], color="#444444", linestyle="--", linewidth=1)
        ax.set(xlabel="21 日累積報酬", ylabel="年化波動率", title=title)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.15)
    combined = np.vstack([train, draws, realized])
    for ax in axes[1]:
        for setter, k in ((ax.set_xlim, 0), (ax.set_ylim, 1)):
            lo, hi = combined[:, k].min(), combined[:, k].max()
            pad = max((hi - lo) * .05, .001)
            setter(lo - pad, hi + pad)
    fig.suptitle("Dow 10｜歷史、報酬與波動配對取樣", fontsize=15)
    fig.savefig(out / "condition_distributions.png", dpi=160)
    plt.close(fig)
    # Deterministic selection by target-return ranks, never by attractive outputs.
    ordered = plan.draws.drop_duplicates("window_id").sort_values(["target_return", "window_id"])
    chosen = ordered.iloc[np.unique(np.linspace(0, len(ordered) - 1, min(6, len(ordered))).round().astype(int))]
    reps = len(values) // len(plan.draws)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True, sharex=True, sharey=True)
    for ax, row in zip(axes.flat, chosen.itertuples()):
        d = int(row.draw_id)
        for j in range(d * reps, (d + 1) * reps):
            ax.plot(np.arange(22), values[j], color="#275c96", alpha=.55, linewidth=1.2)
        ax.plot(np.arange(22), ref_values[d * reps], color="#333333", linestyle="--", linewidth=1.3,
                label="配對真實路徑")
        ax.set_title(f"{row.anchor_date}\nR={row.target_return:.1%}；波動={row.target_volatility:.1%}", fontsize=10)
        ax.set(xlabel="生成後交易日", ylabel="等權組合淨值（起點 1）")
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle("同一歷史與條件下，不同 diffusion noise 的路徑", fontsize=14)
    fig.savefig(out / "within_condition_paths.png", dpi=160)
    plt.close(fig)
