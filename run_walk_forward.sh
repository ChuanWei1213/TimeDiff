#!/usr/bin/env bash
# ============================================================
# run_walk_forward.sh
# DOW10 Walk-Forward（滾動重訓）實驗：2018-01-02 ~ 2024-12-31
#   - 每 63 個共同交易日重訓一次，共 28 期（最後一期 60 日）
#   - 每期訓練窗 [origin-1260, origin)；選模 1008/252 → 選 best step
#     → 重新初始化用完整 1260 日訓練到該 step
#   - 每期生成 21/21/21；價格自 2017 最後真實 Close 起全段串接、不重置
#
# 既有檔案完全未被修改（本實驗只用 walk_forward_dow10.py +
# data_provider/preprocess_walkforward_dow10.py 兩支新檔）。
# ============================================================
set -e
cd "$(dirname "$0")"

TAG="dow10_wf"
OUT_DIR="./outputs/dow10/walk_forward/timediff/${TAG}"

# ---- 訓練預算（先小規模跑通，再放大）----
MAX_STEPS=2000        # 選模階段 step 上限，同時是兩階段 LR/annealing 基準
EVAL_EVERY=100        # 每 100 step 算一次 validation MSE（→ 每期 20 次驗證）
VAL_BATCHES=0         # 0 = 用完整 232 個驗證窗口；設 4 可大幅省時
PRETRAIN_STEPS=500    # DLinear 預訓
VOL_SCALE=1.0         # 既有 DOW10 校準值為 3.86
N_PATHS=20            # 一次生成幾份資料（同一批模型，只差取樣噪聲；幾乎不加時間）

# 之後要在同一批模型上「再追加」幾份資料（不重訓，秒級）：
#   python walk_forward_dow10.py --tag ${TAG} --out_dir ${OUT_DIR} \
#       --generate_only --n_paths 20 --gen_seed 1
# 想要連「模型隨機性」都不同的獨立資料集，就換 seed 完整重跑：
#   python walk_forward_dow10.py --tag ${TAG}_s7 --seed 7 --n_paths 20 --save_ckpt（僅修振幅）

echo "========================================="
echo "  DOW10 Walk-Forward 實驗"
echo "  tag        : ${TAG}"
echo "  max_steps  : ${MAX_STEPS}  eval_every: ${EVAL_EVERY}"
echo "  預算警告   : 28 期 × 2 次訓練 = 56 次完整訓練"
echo "========================================="

# ---- Step 1: 資料前處理（2012-2024，含 OHLC）----
if [ ! -f "./datasets/prediction/data_finance/dow10_wf_returns.csv" ]; then
  echo ""
  echo "[Step 1] 下載並前處理 DOW10 (2012-2024) ..."
  python data_provider/preprocess_walkforward_dow10.py
else
  echo ""
  echo "[Step 1] 已存在 dow10_wf_returns.csv，跳過前處理。"
fi

# ---- Step 2: Walk-Forward 訓練 + 生成 ----
# --resume：已完成的期別會直接沿用 cycles/cycle_XX_returns.csv，可中斷續跑
echo ""
echo "[Step 2] Walk-Forward 訓練與生成 ..."
python walk_forward_dow10.py \
    --tag ${TAG} \
    --out_dir ${OUT_DIR} \
    --max_steps ${MAX_STEPS} \
    --eval_every ${EVAL_EVERY} \
    --val_batches ${VAL_BATCHES} \
    --pretrain_steps ${PRETRAIN_STEPS} \
    --batch_size 32 \
    --test_batch_size 16 \
    --learning_rate 1e-3 \
    --diff_steps 100 \
    --type_sampler dpm \
    --parameterization x_start \
    --use_window_normalization False \
    --use_volatility_loss True \
    --use_leverage_loss True \
    --lambda_vol_max 0.1 \
    --lambda_lev_max 0.1 \
    --vol_scale ${VOL_SCALE} \
    --n_paths ${N_PATHS} \
    --save_ckpt \
    --use_amp True \
    --seed 2024 \
    --resume

echo ""
echo "✅ 完成。結果：${OUT_DIR}/"
echo "   paths/path_XX_returns.csv、path_XX_prices.csv（${N_PATHS} 份下游資料）"
echo "   wf_paths_summary.csv（各份的 vol_ratio 與 5 項 stylized facts）"
echo "   wf_generated_returns.csv / wf_generated_prices.csv / wf_price.png（= path 00）"
echo "   wf_stylized_facts.txt / cycles/cycle_XX_meta.json（各期 best_step 與驗證曲線）"
