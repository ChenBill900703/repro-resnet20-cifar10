# CIFAR-10 Plain-20 / ResNet-20 重現規格

> 主要證據：He et al., *Deep Residual Learning for Image Recognition*, arXiv:1512.03385v1，`references/1512.03385v1.pdf`。頁碼均指 PDF 頁碼。狀態使用 **EXPLICIT**（原文明示）、**DERIVED**（由原文直接推導）、**ASSUMPTION**（專案必須選定但非論文明示）與 **UNKNOWN**（無法可靠確定）。

## 1. 重現目標

### 論文研究主張

- **EXPLICIT**：深度增加後，plain network 的 accuracy 先飽和再快速下降；更深模型甚至有更高 training error，因此 degradation problem 不是單純 overfitting，而是最佳化困難（Introduction，PDF p.1；Figure 1）。
- **EXPLICIT**：若新增層能表成 identity mapping，較深模型理論上存在不劣於淺層模型的建構解；既有 solver 未必能在可行時間找到它（Introduction，PDF pp.1-2）。
- **EXPLICIT**：Residual learning 將目標映射改寫為 `F(x) := H(x) - x`，輸出為 `F(x) + x`，假設 residual mapping 較容易最佳化（Section 3.1，PDF p.3）。
- **EXPLICIT**：CIFAR-10 實驗顯示更深 plain nets 有更高 training error，而 ResNets 可克服此最佳化困難並從深度取得 accuracy gain（Section 4.2，PDF p.7；Figure 6，PDF p.8）。

### 本專案縮小後的重現範圍

- **ASSUMPTION／專案規則**：只實作與比較 Plain-20、CIFAR-10 post-activation ResNet-20（option A）。
- **ASSUMPTION／專案規則**：每個模型先各做一個正式 64,000-update run，以 final checkpoint 報告 test error，而非搜索最佳 checkpoint。
- 最小研究問題是：在相同 depth、width、parameter count 與訓練 recipe 下，residual shortcut 是否改善最佳化行為；不企圖重現整篇論文的所有 depth 或 ImageNet 結果。

## 2. 資料集與評估資料切分

| 項目 | 規格 | 狀態 | 來源與說明 |
| --- | --- | --- | --- |
| Dataset | CIFAR-10 | EXPLICIT | Section 4.2，PDF p.7 |
| Training images | 50,000 | EXPLICIT | Section 4.2，PDF p.7 |
| Test images | 10,000 | EXPLICIT | Section 4.2，PDF p.7 |
| Classes | 10 | EXPLICIT | Section 4.2，PDF p.7 |
| 訓練／評估方式 | training set 訓練，test set 評估 | EXPLICIT | Section 4.2，PDF p.7 |
| Validation split 是否為正式固定 split | 否；論文只說 45k/5k train/val split 用於決定訓練時程 | EXPLICIT | Section 4.2，PDF p.7 |
| 45k/5k 的角色 | 決定 32k、48k、64k 的 schedule／termination | EXPLICIT | Section 4.2，PDF p.7 |
| Table 6 最終模型是否重新以完整 50k 訓練 | 論文語句暗示使用 training set，但未明寫 schedule 決定後的重訓程序 | UNKNOWN | 見 `open_questions.md` |
| 本專案正式訓練 split | 官方 50k train / 10k test；不得用 test set 調參 | ASSUMPTION／專案規則 | `AGENTS.md` |

## 3. Plain-20 模型結構

Plain-20 沿用 Section 4.2 的 CIFAR plain architecture，沒有 shortcut。為與 ResNet-20 對照，下表把每兩個 convolution 視為一個「two-convolution group」；這是 **DERIVED** 的整理單位，不表示 plain network 本身具有 residual block。

| 區段 | 輸入尺寸 | 輸出尺寸 | 通道數 | 卷積規格 | stride | 卷積層數 | block／group 數 | shortcut |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| Input | `32x32x3` | `32x32x3` | 3 | - | - | 0 | 0 | 無 |
| Initial convolution | `32x32x3` | `32x32x16` | 16 | `3x3` | `1`（DERIVED，由輸出尺寸約束） | 1 | 0 | 無 |
| Stage 1 | `32x32x16` | `32x32x16` | 16 | 每層 `3x3` | `1`（DERIVED） | 6 | 3 groups（DERIVED） | 無 |
| Stage 2 | `32x32x16` | `16x16x32` | 32 | 每層 `3x3` | 首層 `2`，其餘 `1`（下採樣位置為 DERIVED） | 6 | 3 groups（DERIVED） | 無 |
| Stage 3 | `16x16x32` | `8x8x64` | 64 | 每層 `3x3` | 首層 `2`，其餘 `1`（下採樣位置為 DERIVED） | 6 | 3 groups（DERIVED） | 無 |
| Global average pooling | `8x8x64` | `1x1x64` | 64 | - | - | 0 | 0 | 無 |
| Fully connected | 64 | 10 logits | 10 | 10-way FC | - | 1 | 0 | 無 |
| Softmax | 10 logits | 10 class scores | 10 | softmax | - | 0 | 0 | 無 |

- **EXPLICIT**：輸入 `32x32`；initial convolution 為 `3x3`；三個 feature-map size 為 `{32,16,8}`；filters 為 `{16,32,64}`；下採樣由 stride-2 convolution 執行；結尾為 global average pooling、10-way FC 與 softmax（Section 4.2，PDF p.7）。
- **DERIVED**：`n=3` 時，三個 stage 各有 `2n=6` 個 convolution，共 18；加 initial convolution 與 FC 得 `1+18+1=20` weighted layers，即 19 convolution + 1 FC。
- Convolution padding、convolution bias、FC bias 與精確 activation/BN 框架細節未在 CIFAR 段落完整明示；不得由此表擅自視為論文事實，見 `assumptions.md`。

## 4. ResNet-20 模型結構

| 區段 | 輸入尺寸 | 輸出尺寸 | 通道數 | 卷積規格 | stride | 卷積層數 | block 數 | shortcut |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| Input | `32x32x3` | `32x32x3` | 3 | - | - | 0 | 0 | 無 |
| Initial convolution | `32x32x3` | `32x32x16` | 16 | `3x3` | `1`（DERIVED） | 1 | 0 | 無 |
| Stage 1 | `32x32x16` | `32x32x16` | 16 | 每個 block 兩個 `3x3` | `1` | 6 | 3 | identity option A |
| Stage 2 | `32x32x16` | `16x16x32` | 32 | 每個 block 兩個 `3x3` | 第一個 block 的第一個 convolution 為 `2`（DERIVED）；其餘 `1` | 6 | 3 | 第一個為 stride-2 + zero-padding option A；其餘 identity |
| Stage 3 | `16x16x32` | `8x8x64` | 64 | 每個 block 兩個 `3x3` | 第一個 block 的第一個 convolution 為 `2`（DERIVED）；其餘 `1` | 6 | 3 | 第一個為 stride-2 + zero-padding option A；其餘 identity |
| Global average pooling | `8x8x64` | `1x1x64` | 64 | - | - | 0 | 0 | 無 |
| Fully connected | 64 | 10 logits | 10 | 10-way FC | - | 1 | 0 | 無 |
| Softmax | 10 logits | 10 class scores | 10 | softmax | - | 0 | 0 | 無 |

核對結果：

- **EXPLICIT**：CIFAR depth formula 為 `6n+2`；論文比較 `n={3,5,7,9}`，分別得到 20/32/44/56 layers（Section 4.2，PDF p.7）。
- **DERIVED**：ResNet-20 使用 `n=3`，stage blocks 為 `[3,3,3]`，每個 block 含兩個 `3x3` convolution。
- **EXPLICIT**：shortcut 連接每一對 `3x3` layers，共 `3n=9` 個 shortcuts；CIFAR 所有情況採 identity shortcut，即 option A（Section 4.2，PDF p.7）。
- **EXPLICIT**：維度增加時 option A 以額外 zero entries padding，沒有額外 parameter；跨不同 feature-map size 時 shortcut stride 為 2（Section 3.3，PDF p.4）。
- **EXPLICIT**：Figure 2 與 Section 3.2 指出兩層 residual function 可寫成 `F=W2 sigma(W1x)`，element-wise addition 後採用第二個 nonlinearity，即 post-activation（PDF pp.2-3）。
- **DERIVED**：結合「BN right after each convolution and before activation」（Section 3.4，PDF p.4）、CIFAR 採 BN（Section 4.2，PDF p.7）及 Figure 7 的 response 定義（PDF p.8），目標 block 的論文級操作關係為 `conv -> BN -> ReLU -> conv -> BN -> addition -> ReLU`。
- **UNKNOWN**：option A 的具體 tensor 索引、zero-padding 左右分配、odd-size 行為屬框架實作細節；論文只規定 stride 2、identity 與 zero entries。

## 5. 每個 Stage 的輸出尺寸與 Block 數量

| Stage | Output spatial size | Output channels | Convolution layers | Two-convolution blocks | Downsampling |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial | `32x32` | 16 | 1 | 0 | 無 |
| Stage 1 | `32x32` | 16 | 6 | 3 | 無 |
| Stage 2 | `16x16` | 32 | 6 | 3 | 第一個 convolution stride 2；ResNet shortcut 同時 stride 2（後者 EXPLICIT，前者位置 DERIVED） |
| Stage 3 | `8x8` | 64 | 6 | 3 | 第一個 convolution stride 2；ResNet shortcut 同時 stride 2（後者 EXPLICIT，前者位置 DERIVED） |

20 層計算為：initial convolution `1` + 三個 stage 的 convolution `3 x 6 = 18` + final FC `1` = `20` weighted layers。Global average pooling、BN、ReLU、element-wise addition 與 shortcut 不含 learned weights，均不計入 `6n+2` weighted-layer depth。

## 6. 資料預處理

| 項目 | 規格 | 狀態 | 來源／備註 |
| --- | --- | --- | --- |
| Input image size | `32x32` | EXPLICIT | Section 4.2，PDF p.7 |
| Per-pixel mean subtraction | 使用 | EXPLICIT | Section 4.2，PDF p.7 |
| Mean 的統計集合 | CIFAR 段落未明寫；本專案規則指定 training set | UNKNOWN（論文）／ASSUMPTION（專案） | 不得宣稱論文明示 training-set mean |
| Mean 的形狀與精確計算流程 | 未明寫 | UNKNOWN | 可為完整 mean image 或其他實作，需審核 |
| Standard-deviation normalization | 未提及 | UNKNOWN | 不得加入常見 channel-wise mean/std recipe 並標成論文設定 |
| Test image 使用何種 mean | 未明寫 | UNKNOWN | 合理實作應重用 training-derived mean，但仍是 ASSUMPTION |
| Tensor 數值範圍 | 未明寫 | UNKNOWN | `0..255` 或 `0..1` 均不可視為論文明示 |

## 7. 資料增強

| 項目 | 規格 | 狀態 | 來源／備註 |
| --- | --- | --- | --- |
| Padding | 每側 4 pixels | EXPLICIT | Section 4.2，PDF p.7 |
| Random crop | 從 padded image 取 `32x32` crop | EXPLICIT | Section 4.2，PDF p.7 |
| Horizontal flip | 從 padded image 或其 horizontal flip 隨機取樣 | EXPLICIT | Section 4.2，PDF p.7 |
| Padding mode | 未明寫 | UNKNOWN | 見 `assumptions.md` |
| Test-time views | 原始 `32x32` image 的 single view | EXPLICIT | Section 4.2，PDF p.7 |
| 其他 augmentation | CIFAR 段落未列其他項目 | UNKNOWN | 不得把 ImageNet color/scale augmentation 套入 |

## 8. Optimizer 與訓練設定

| 設定 | 論文值 | 狀態 | 原文位置 | 備註 |
| --- | --- | --- | --- | --- |
| Optimizer | SGD | EXPLICIT | Abstract/Introduction 與 Section 3.4，PDF pp.1,4 | CIFAR 段落沿用該訓練描述 |
| Global batch size | 論文稱 mini-batch size 128 | EXPLICIT（數值）；global 解讀需 ASSUMPTION | Section 4.2，PDF p.7 | 在 two GPUs 上訓練；未寫 per-device split |
| GPU 數量 | 2 | EXPLICIT | Section 4.2，PDF p.7 | 本專案單 GPU 是 ASSUMPTION／刻意偏離 |
| Initial learning rate | `0.1` | EXPLICIT | Section 4.2，PDF p.7 | ResNet-20 無 warm-up 敘述 |
| LR adjustments | 32k、48k iterations | EXPLICIT | Section 4.2，PDF p.7 | milestone 邊界的 step-before/after 未明寫 |
| Adjustment factor | 每次除以 10 | EXPLICIT | Section 4.2，PDF p.7 | LR 序列為 0.1、0.01、0.001（DERIVED） |
| Total iterations | 64k，之後 terminate | EXPLICIT | Section 4.2，PDF p.7 | final/best checkpoint reporting form 未明寫 |
| Momentum | `0.9` | EXPLICIT | Section 4.2，PDF p.7 | 具體 optimizer implementation 未明寫 |
| Weight decay | `0.0001` | EXPLICIT | Section 4.2，PDF p.7 | 套用至 bias/BN 與否未明寫 |
| Dropout | 不使用 | EXPLICIT | Section 4.2，PDF p.7 | Figure 8 附近也重申 no maxout/dropout |
| Loss / softmax | 網路末端 softmax；loss 未命名 | EXPLICIT／UNKNOWN | Section 4.2，PDF p.7 | 不得把特定 cross-entropy API 當原文明示 |
| Warm-up | 只對 ResNet-110 描述 `0.01` 約 400 iterations，再回 `0.1` | EXPLICIT | Section 4.2，PDF p.7 | **不得套用至 ResNet-20** |

由論文數值計算：`64,000 x 128 = 8,192,000` 次 training-example presentations；若每個等效 epoch 以 50,000 個 training examples 計，約為 `163.84` epochs。這是 **DERIVED**，不是論文直接陳述；random sampling/shuffle 使其不代表每張影像恰好出現相同次數。

## 9. 權重初始化

- **EXPLICIT**：論文說採用文獻 `[13]` 的 weight initialization（Sections 3.4、4.2，PDF pp.4,7）。
- `[13]` 是 He, Zhang, Ren, Sun, “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification,” ICCV 2015（References，PDF p.9）。
- **UNKNOWN**：本論文沒有重新提供完整初始化公式，也未在 CIFAR 段落分別交代 convolution、FC、bias 的 distribution、fan mode、gain 或常數。
- 因此「採用 [13]」不可直接展開成 normal/uniform、fan_in/fan_out 等已確定細節；須先查主要引用或列為實作 ASSUMPTION。

## 10. Batch Normalization

| 項目 | 論文資訊 | 狀態 |
| --- | --- | --- |
| BN placement | 每個 convolution 後、activation 前 | EXPLICIT（Section 3.4，PDF p.4） |
| CIFAR 是否採 BN | 採用 BN `[16]` | EXPLICIT（Section 4.2，PDF p.7） |
| CIFAR 每個 convolution 都有 BN | 由 Section 3.4 的通用 implementation 與 CIFAR「adopt BN」合併 | DERIVED |
| Residual block 相對順序 | `conv-BN-ReLU-conv-BN-add-ReLU` | DERIVED；Figure 2、Sections 3.2/3.4、Figure 7 |
| Dropout | 不使用 | EXPLICIT |
| Epsilon | 未寫 | UNKNOWN |
| Running-statistics momentum | 未寫 | UNKNOWN |
| Affine gamma/beta | 未寫 | UNKNOWN |
| 兩張 GPU 是否同步 BN | 未寫 | UNKNOWN |
| Evaluation statistics 行為 | 未寫 | UNKNOWN |

Figure 7 的 layer response 定義是「每個 `3x3` layer 經 BN 後、其他 nonlinearity（ReLU/addition）之前的輸出」；Figure 7 上半按原 layer order，下半按 magnitude 降序排列（PDF p.8）。

## 11. 評估指標與測試協定

- **EXPLICIT**：Table 6 報告 CIFAR-10 test-set classification error（test error），不是 accuracy（PDF p.7）。Accuracy 可由 `100% - test error` 推導，但不是表中直接欄位。
- **EXPLICIT**：test 僅評估原始 `32x32` image 的 single view，不使用 multi-crop（Section 4.2，PDF p.7）。
- **UNKNOWN**：評估時 mean 的確切套用流程、BN evaluation statistics、test loss 實作與 evaluation interval。
- **UNKNOWN**：ResNet-20 的 8.75% 是 best、last、單次 run 的哪個 checkpoint，論文未說明。
- **EXPLICIT**：只有 ResNet-110 明說執行 5 次並報告 `best (mean +/- std)`；不得把該 reporting form 套用到 ResNet-20。
- **ASSUMPTION／專案規則**：主要結果固定取第 64,000 update 的 final checkpoint；不得依 test set 選 checkpoint。

## 12. 論文原始結果

| Model | Depth | Parameters | CIFAR-10 test error | Reporting form | Source |
| --- | ---: | ---: | ---: | --- | --- |
| Plain-20 | 20 | `0.27M`（DERIVED：plain/residual 同參數 + ResNet-20 列值） | **UNKNOWN：Table 6 未提供精確值** | Figure 6 只有低解析度曲線，不得估成精確 final error | Section 4.2 與 Figure 6，PDF pp.7-8 |
| ResNet-20 | 20 | `0.27M` | `8.75%` | 單一表列值；best/last/mean 未說明 | Table 6，PDF p.7 |
| ResNet-110 | 110 | `1.7M` | `6.43%` best；`6.61 +/- 0.16%` mean +/- std | 5 runs，`best (mean +/- std)` | Table 6 與 caption，PDF p.7 |

Figure 6 明確區分 dashed training error 與 bold testing error，但 Plain-20 的精確原論文最終 test error 不存在於 Table 6。視覺核對只能支持相對曲線行為，不足以產生可引用的精確數值。

## 13. 最小可行重現範圍

### 第一階段包含

- CIFAR-10 official 50k train / 10k test split。
- Plain-20。
- ResNet-20、option A、9 個無參數 shortcuts。
- 每個模型一個正式 64k run。
- 第 64,000 update final checkpoint 的 final test error。
- Training/test error curves；固定頻率評估但不得用於選模型。
- Parameter count 與 architecture/shape/shortcut/forward/backward/smoke tests。
- ResNet-20 與論文 Table 6 的 `8.75%` 比較；不得承諾逐位相同。
- Plain 與 Residual 的相對最佳化行為比較；Plain-20 不與虛構的精確 paper target 比較。

### 第一階段暫不做

- ResNet-32/44/56/110/1202。
- ImageNet、object detection。
- 多 GPU 或 SyncBN 重現。
- 現代 augmentation、AMP、TF32、`torch.compile`。
- 超參數搜尋、seed 搜尋、test-set tuning、early stopping。
- 為追求更高 accuracy 而修改論文 recipe。

## 14. 重現成功判定

1. **Architecture fidelity**：20 weighted layers、19 convolution + 1 FC、stage blocks `[3,3,3]`、spatial sizes `[32,16,8]`、channels `[16,32,64]`、ResNet option A 無參數 shortcut、post-activation 順序與參數量測試全部通過。
2. **Training-protocol fidelity**：SGD、batch 128、LR 0.1 並於 32k/48k 各除 10、64k 結束、momentum 0.9、weight decay 0.0001、BN、no dropout、指定 preprocessing/augmentation 與 final-checkpoint 規則皆可由 frozen config 和 logs 稽核。
3. **Numerical-result proximity**：ResNet-20 final test error 與論文 `8.75%` 做透明比較，並比較 Plain/Residual training dynamics。論文沒有提供容許誤差；任何 tolerance 必須在正式 run 前由本專案預先制定並標成 **ASSUMPTION**，不得稱為原論文標準。Plain-20 沒有精確 paper target。

目前 numerical tolerance 尚未核准，不能以事後調整門檻宣稱成功。

## 15. 必須保存的設定與輸出

- Frozen config（含每個欄位的 evidence status）。
- Source commit／dirty-worktree 狀態。
- Environment 紀錄與有效的本機 package lock。
- `assumptions.md` 的審核版本與 open-question resolution。
- 預先選定的 random seed；不得依 test 表現挑 seed。
- Model summary、逐 stage shapes、weighted-layer count、parameter count。
- Architecture、shape、parameter count、shortcut、forward、backward、smoke-test 輸出。
- Per-update 或固定週期 logs、global step、learning rate、train loss、train error、test loss、test error。
- Checkpoints：model weights、optimizer state、scheduler/global step、BN running statistics、RNG states。
- 第 64,000 update final checkpoint 與 final predictions（sample order、labels、logits或 probabilities、predicted classes）。
- 與論文規格和 ResNet-20 8.75% 的 comparison report；清楚揭露所有 ASSUMPTION、UNKNOWN 與偏離。
