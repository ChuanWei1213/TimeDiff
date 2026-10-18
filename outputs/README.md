# 實驗產物索引

更新：2026-10-18。先依資料市場與實驗協定分類，再依模型、取樣方法與批次區分。
完整設定、現有檔案數與舊新路徑對照見 [catalog.yaml](catalog.yaml)。

```text
outputs/
├── README.md / catalog.yaml
├── dow10/
│   ├── walk_forward/
│   │   ├── timediff/<模型 tag>/
│   │   ├── statistical/<模型 tag>/
│   │   ├── external/<模型 tag>/
│   │   └── analyses/
│   └── conditional/
│       ├── fixed_scenarios/legacy/
│       └── empirical_joint/run_20261017/
│           ├── train/
│           ├── plans/full70_tail30_pr30/
│           ├── batches/full70_tail30_pr30/
│           ├── batches/full70_tail30_pr10/
│           ├── analyses/generation_quality/
│           ├── analyses/pr30_vs_pr10/
│           └── designs/random_stock_pr30/
├── spx/                         原位置保留
└── _migration/20261018/          搬移紀錄、原始碼備份與讀取驗證
```

`single_shot` 尚無現存資料；後續 `generate_dow10.py` 預設寫入 `dow10/single_shot/<tag>/`。
19 個舊空目錄已清理，提交前確認均不存在，見 [清理查核](_migration/20261018/empty_directory_cleanup.json)。
搬移當下受主機限制而暫留空目錄的紀錄，仍保存在 [relocation.json](_migration/20261018/relocation.json)。

## 條件式 Dow 10 資料

| 批次 | 生成方式 | 現存資料 | 模型／用途 |
|---|---|---|---|
| [舊固定情境](dow10/conditional/fixed_scenarios/legacy/rl_paths/paths.npz) | 人工指定三組報酬／波動，實際條件為 −10%／50%、−23%／100%、−36%／140% | 6,039 × 21 日 × 10 檔 | 舊 Scenario Mix 使用；原 checkpoint 本機未留存 |
| [70% 完整＋30% PR30](dow10/conditional/empirical_joint/run_20261017/batches/full70_tail30_pr30/README.md) | 保留歷史與其後報酬、波動的聯合配對；尾部門檻 −0.1841% | 2,000 × 21 日 × 10 檔 | 共用下列模型，尚未跑 PPO |
| [70% 完整＋30% PR10](dow10/conditional/empirical_joint/run_20261017/batches/full70_tail30_pr10/README.md) | 同一取樣方法改用 PR10；尾部門檻 −3.4763% | 2,000 × 21 日 × 10 檔 | 共用下列模型，尚未跑 PPO |

PR30 與 PR10 共用 [run_20261017 的模型](dow10/conditional/empirical_joint/run_20261017/train/model.pt)：訓練目標期間 2005–2017、
2018–2019 選模，最佳 step 3,500。條件母體為 2010–2017，共 1,993 個視窗。
每批抽 500 次，每次生成 4 條；70／30 是來源池配額，完整池也含尾部。
原訓練紀錄見 [metadata.json](dow10/conditional/empirical_joint/run_20261017/train/metadata.json)。

| 配套內容 | 位置與意義 |
|---|---|
| PR30 原始分析 | [實驗說明](dow10/conditional/empirical_joint/run_20261017/README.md) |
| 相關性、high／low 與尾部診斷 | [診斷 notebook](dow10/conditional/empirical_joint/run_20261017/analyses/generation_quality/diagnostic_review.ipynb) |
| PR30 對 PR10 比較 | [比較 notebook](dow10/conditional/empirical_joint/run_20261017/analyses/pr30_vs_pr10/pr10_comparison.ipynb) |
| 隨機單股 PR30 取樣構想 | [設計資料夾](dow10/conditional/empirical_joint/run_20261017/designs/random_stock_pr30/)；屬於設計分析，沒有生成批次 |

PR10 的抽樣計畫保留在自己的 `batches/full70_tail30_pr10/sampling_plan.csv`，不複製第二份。
`generation.json`、`metadata.json`、舊命令與既有結果 JSON 保留生成當時的原文，包括當時的舊路徑。

## Walk-forward 模型

模型 tag 保持不變，`MODELS_WHITELIST` 不需要加上分類前綴。
以下為本次實際讀取狀態，不把當年預定的路徑數當成目前檔案數。

| 分類 | 模型 tag | 本次讀取狀態 |
|---|---|---|
| [timediff](dow10/walk_forward/timediff/) | dow10_wf_v2、dow10_wf_vpred、v_cross、v_cross_ga、v_whiten、v_3layer、v_4layer、v_4layer_fix | 個別 CSV 可讀；同模型不同檔案有既存長度／日期不一致，整批載入會報錯 |
| [statistical](dow10/walk_forward/statistical/) | gauss_sigma | 可載入 10 條、799 天；未涵蓋完整 1,761 天評估期間 |
| statistical | garch_null | 個別 CSV 可讀；既存日期不一致，整批載入會報錯 |
| [external](dow10/walk_forward/external/) | cndiff、nsdiff | 各可載入 1 條、1,761 天 |
| external | armd、dva | 保留原始資料，缺少 loader 可辨識的標準化 returns 檔 |

上述缺漏在搬移前已存在，原 CSV 的 SHA-256 全部未變；本次未補值、截短或重建它們。
逐檔列數、日期與錯誤訊息見 [讀取驗證](_migration/20261018/readability_audit.json)。
既有 [P0/P1 報告](dow10/walk_forward/analyses/p0p1_results.txt) 也已保留。

## 程式引用與保存驗證

- [output_layout.py](../output_layout.py) 統一新位置、模型探索與歷史路徑解析；原本的模型 tag 不變。
- 訓練／生成入口的預設輸出、評估 loader、分析程式與 notebook 的來源引用均使用新位置。
- FinRL 的 `SyntheticPaths` 相容舊 `outputs/dow10_cond/rl_paths/paths.npz` 設定，原實驗 config 不必改寫。
- 455 個原始產出檔均有 [舊新位置與 SHA-256 紀錄](_migration/20261018/inventory.json)；數據、權重、圖表與歷史 JSON 保持原始位元組。
- 只修改搬移所需的程式、文件與 notebook 來源，原始版本另存 [original_sources](_migration/20261018/original_sources/)；notebook 已保存的輸出與執行次數維持原樣。
- 34 項相關測試通過；三本分析 notebook 的 24 個非空程式單元以同一行程、依序且遇錯即停的 IPython 方式通過。18 個檢查產物改寫到暫存目錄，見 [notebook_smoke.json](_migration/20261018/notebook_smoke.json)。此檢查未啟動真正的 Jupyter kernel，也不驗證互動元件。
- 本次僅搬移與讀取驗證，沒有重新訓練、重新生成資料、執行回測或 PPO。讀取成功不等於生成品質已符合後續訓練需求。

本次 Git 追蹤包含搬移後的 Dow 10 walk-forward 資料、固定情境資料、PR30／PR10 模型與批次、
診斷／設計分析，以及搬移稽核與原始碼備份。未搬動的 `spx/` 產物與 Python 快取維持本機保存；
完整搬移稽核包含這些本機檔案，因此需在保有原始產物的工作目錄執行。

從 TimeDiff 根目錄可重新執行唯讀資料檢查（僅更新搬移稽核報告）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-spx/bin/python scripts/verify_outputs_layout.py
```

未來同模型的新取樣方法放到新的 `batches/<取樣設定>/`；模型重訓則建立新的 `run_<日期>/`。
設計構想放 `designs/`，跨批次比較放 `analyses/`，讓資料集與研究說明各有清楚位置。
