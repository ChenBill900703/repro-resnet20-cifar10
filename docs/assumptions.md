# 實作假設清單

本文件只列出「為了實作必須決定，但主要來源未完整說明」的項目。`APPROVED_BY_USER`表示使用者已批准專案採用，不會改變「論文狀態」欄，也不得被描述成原作者明示設定。2026-08-03的主要來源查證見`docs/primary_source_review.md`，正式決策以`docs/decision_log.md`與frozen config為準。

| ID | 項目 | 論文狀態 | 建議主要假設 | 替代方案 | 可能影響 | 決定狀態 |
| --- | --- | --- | --- | --- | --- | --- |
| ASSUMP-RNG-001 | Random seed | UNKNOWN | 正式run固定base seed=`1`並保存Python/framework/CUDA與DataLoader RNG state；不得依test表現選seed | 多個預先註冊 seeds；完全不固定 | 初始化、shuffle 與結果變異 | APPROVED_BY_USER |
| ASSUMP-FW-001 | Framework | UNKNOWN | 使用專案既有PyTorch環境，自行實作Plain-20/ResNet-20 | 原作者 Caffe；其他框架 | 算子語意、初始化與 BN 差異 | APPROVED_BY_USER |
| ASSUMP-GPU-001 | Single GPU vs paper two GPUs | EXPLICIT：論文為 two GPUs；專案硬體選擇非論文設定 | 使用單張RTX 3070 Ti，保持global batch 128；明列為project decision | 模擬 two GPUs；另取得雙 GPU | BN 與數值路徑、吞吐量 | APPROVED_BY_USER |
| ASSUMP-BATCH-001 | Per-device batch size | UNKNOWN | 單GPU per-device batch 128，使global batch等於論文mini-batch 128 | Gradient accumulation；拆成 64+64 雙 GPU | BN batch statistics、最佳化 | APPROVED_BY_USER |
| ASSUMP-BN-001 | BN 是否跨 GPU 同步 | UNKNOWN | single GPU、non-SyncBN；在報告揭露與論文two-GPU情境不同 | 假設原作者每 GPU local BN；假設同步 BN | Running statistics 與 accuracy | APPROVED_BY_USER |
| ASSUMP-SCALE-001 | Pixel-value scaling | UNKNOWN（作者 CIFAR pipeline）；Caffe b590 default 已確認 | 保留`[0,255]`float尺度、`scale=1`，在同一尺度減mean | 先縮放至 `[0,1]`；其他線性尺度 | 有效初始化與最佳化尺度 | APPROVED_BY_USER |
| ASSUMP-MEAN-001 | Per-pixel mean image 精確流程 | UNKNOWN：只明寫 per-pixel mean subtraction | 用完整50k official training set按每個`(channel,y,x)`平均，得到`3x32x32`mean image，train/test共用 | 45k subset；channel-wise scalar mean | 輸入分布與可比性 | APPROVED_BY_USER |
| ASSUMP-AUGPAD-001 | Augmentation padding mode | EXPLICIT（[24] 明寫 zero padding） | 每側4 pixels採constant-zero padding | 無；除非找到更高優先級作者來源推翻 | 邊界樣本分布 | APPROVED_BY_USER |
| ASSUMP-AUGORDER-001 | Geometric augmentation order | PARTIALLY_CONFIRMED | 幾何子序列固定為constant-zero pad 4→random`32x32`crop→random horizontal flip | flip與crop的等價次序；其他RNG程序 | 資料分布與可重現性 | APPROVED_BY_USER |
| ASSUMP-MEANORDER-001 | Mean subtraction相對於augmentation的位置 | UNKNOWN | Training採`[0,255]`float→subtract`3x32x32`mean→zero-pad→crop→flip；test只轉float並減同一mean | augmentation→mean；其他座標對齊方案 | 邊界像素與per-pixel mean對齊 | APPROVED_BY_USER |
| ASSUMP-CONVPAD-001 | Convolution padding | UNKNOWN；輸出尺寸只形成約束 | 所有`3x3`convolution採padding 1，以符合`{32,16,8}`sizes | 其他顯式 padding/cropping | Shape、感受野與參數對齊 | APPROVED_BY_USER |
| ASSUMP-CONVBIAS-001 | Convolution bias | UNKNOWN | BN前convolution不建立bias | 使用 bias | Parameter count、weight decay、微小數值差異 | APPROVED_BY_USER |
| ASSUMP-FCBIAS-001 | FC bias | UNKNOWN | 10-way FC使用bias並初始化為0 | 無 bias | Parameter count 與 logits | APPROVED_BY_USER |
| ASSUMP-INITDIST-001 | Exact convolution initialization distribution | EXPLICIT（[13]） | zero-mean normal，std`sqrt(2/fan_in)`；不得用uniform | 無；fan mode另列 | 收斂與結果變異，風險高 | APPROVED_BY_USER |
| ASSUMP-FAN-001 | Fan mode / nonlinearity gain | UNKNOWN（[13] 同時允許 fan-in與fan-out） | 採`fan_in`，gain對應ReLU`sqrt(2)`；明列為project decision | `fan_out` Eq. (14) | Activation/gradient variance | APPROVED_BY_USER |
| ASSUMP-FCINIT-001 | FC weight initialization | UNKNOWN | zero-mean normal、std`0.01`；保留`LOW_CONFIDENCE_ASSUMPTION`標記，不得因test結果更換 | Caffe default；沿用convolution recipe；其他預先註冊值 | 初始logits與早期收斂 | APPROVED_BY_USER |
| ASSUMP-BNEPS-001 | BN epsilon | UNKNOWN | `1e-5`，並在frozen config顯式保存 | Caffe/其他 framework default | 正規化數值，通常影響較小 | APPROVED_BY_USER |
| ASSUMP-BNMOM-001 | BN running-statistics momentum | UNKNOWN（作者 CIFAR）；Caffe b590 default已確認 | Update coefficient採PyTorch-style`momentum=0.001`對應Caffe`.999`；正式訓練前mandatory compatibility test | PyTorch 0.1；cumulative average；訓練後recalibration | Evaluation statistics 與跨框架差異 | APPROVED_BY_USER |
| ASSUMP-BNCOMP-001 | Caffe-compatible BN accumulator/eval implementation | UNKNOWN（作者 CIFAR）；Caffe b590 framework semantics已確認 | 使用scaled mean accumulator、scaled unbiased variance accumulator及running scale；eval以accumulator/scale；checkpoint保存全部buffers | 標準PyTorch BatchNorm；post-training recalibration | Early running statistics、evaluation output與exact checkpoint restore | APPROVED_BY_USER |
| ASSUMP-BNAFF-001 | BN affine | PARTIALLY_CONFIRMED | 啟用trainable gamma/beta，初始gamma=1、beta=0 | 關閉 affine；覆寫初始值 | Parameter count 與表現 | APPROVED_BY_USER |
| ASSUMP-WD-BIAS-001 | Weight decay 是否套用 bias | UNKNOWN（作者 CIFAR）；Caffe ParamSpec default已確認 | `caffe-default-derived assumption`：對存在的FC bias套用decay；不得稱為paper-faithful fact | 排除所有bias；只decay weights | Regularization 與可比性 | APPROVED_BY_USER |
| ASSUMP-WD-BN-001 | Weight decay 是否套用 BN gamma/beta | UNKNOWN（作者 CIFAR）；Caffe ParamSpec default已確認 | `caffe-default-derived assumption`：對BN gamma/beta都套用decay；不得稱為paper-faithful fact | 排除BN affine；只排除beta | 可能顯著影響 accuracy | APPROVED_BY_USER |
| ASSUMP-LOSS-001 | Loss implementation | UNKNOWN：只寫 softmax | 以logits計算multiclass cross-entropy、reduction mean，不在模型內重複softmax | 顯式 softmax + NLL；其他等價形式 | 數值穩定性與 loss 值 | APPROVED_BY_USER |
| ASSUMP-SHUFFLE-001 | Training/test DataLoader sampling policy | UNKNOWN | Training使用`shuffle=true`且不replacement，每完成一次完整training-set traversal後重新產生排列；使用base-seed-derived generator及`StatefulBatchSampler`。Test使用`shuffle=false`、保持official dataset order且不使用random sampler | Replacement sampling；固定training順序；test random sampler | Optimization dynamics、checkpoint resume重現性與test順序 | APPROVED_BY_USER |
| ASSUMP-WORKERS-001 | DataLoader workers | UNKNOWN | 固定為4 | 0、2、8 等 | 吞吐量、資料順序與 worker RNG | APPROVED_BY_USER |
| ASSUMP-WSEED-001 | Worker seeding | UNKNOWN | DataLoader worker base seed由獨立base-seed-derived generator產生，並同步設定Python/NumPy/torch CPU worker RNG；不得以固定`base_seed+worker_id`在每次重啟重置相同stream | Framework implicit seeding；每次worker啟動固定相同stream | Crop/flip 可重現性 | APPROVED_BY_USER |
| ASSUMP-RNG-002 | Per-sample augmentation RNG與prefetch-safe exact resume | UNKNOWN | Seed由base seed、epoch、official sample index派生；Dataset顯式持有epoch且batch攜帶indices；optimizer update成功後才移動consumed cursor；checkpoint保存permutation/cursor/epoch/generator state | 只存generator state；依賴worker-local RNG；以issued cursor保存 | Mid-epoch resume樣本跳過／重複、augmentation漂移 | APPROVED_BY_USER |
| ASSUMP-DET-001 | Deterministic algorithms | UNKNOWN | 正式主要run啟用deterministic algorithms；不支援時必須停止並重新決策 | 允許 nondeterministic kernels | 重跑一致性與效能 | APPROVED_BY_USER |
| ASSUMP-CUDNN-001 | cuDNN benchmark | UNKNOWN | disabled | enabled | 效能與 bit-level repeatability | APPROVED_BY_USER |
| ASSUMP-TF32-001 | TF32 | UNKNOWN（2015 論文未涉及） | disabled | enabled | 數值路徑與速度 | APPROVED_BY_USER |
| ASSUMP-LOG-001 | Logging interval | UNKNOWN | 每100 updates記錄聚合train metrics；每個update記錄LR/global step | 每 update；每 epoch | 曲線解析度與 I/O | APPROVED_BY_USER |
| ASSUMP-EVAL-001 | Evaluation interval | UNKNOWN | 固定每1,000 updates加final 64k；結果不得用於調參或checkpoint選擇 | 僅 final；每 epoch | 曲線可比性與 test-set 接觸頻率 | APPROVED_BY_USER |
| ASSUMP-CKPTINT-001 | Checkpoint interval | UNKNOWN | 在32k、48k、64k保存；只有64k是主要結果 | 每1k/5k；只存final | 故障恢復、儲存量 | APPROVED_BY_USER |
| ASSUMP-FINAL-001 | Final checkpoint definition | UNKNOWN（論文 reporting form）；專案已有規則 | 完成optimizer update #64,000後保存的狀態定義為final，不得以test表現替換 | milestone 前保存；best checkpoint | 主要結果的完整性 | APPROVED_BY_USER |
| ASSUMP-LRSTEP-001 | 本專案採用的LR milestone boundary | ASSUMPTION；Caffe b590 framework semantics另已明確確認 | 完成#32,000後切至0.01，#32,001首次使用新LR；完成#48,000後切至0.001。不得稱為作者CIFAR設定已證實 | #32,000本身使用新LR | 邊界各差一個update | APPROVED_BY_USER |
| ASSUMP-SHORT-001 | Option A 空間下採樣 tensor 操作 | UNKNOWN：只明寫 stride 2 | shortcut使用固定偶數索引的`::2`空間subsampling | Average pooling；奇數索引；其他無參數方式 | 對齊與輸出數值，屬架構 fidelity blocker | APPROVED_BY_USER |
| ASSUMP-SHORTPAD-001 | Option A channel zero-padding 分配 | UNKNOWN：只明寫 extra zero entries | channel差額前後對稱zero-padding（16→32各補8；32→64各補16） | 單側 padding；交錯 padding | Channel 對應與結果 | APPROVED_BY_USER |
| ASSUMP-ODD-001 | Odd-size handling | UNKNOWN | shortcut遇到odd spatial size時明確報錯，不做靜默rounding | Floor/ceil subsampling；顯式 crop | 防止隱性 shape bug；CIFAR 正常路徑預期不觸發 | APPROVED_BY_USER |

## 審核原則

- `PENDING_REVIEW` 不得在沒有決策紀錄時改成已核准。
- `DEFERRED` 不得寫入frozen config；必須由後續使用者決定轉為可審查候選或明確核准方案。`APPROVED_BY_USER`仍須遵守對應preflight execution gate。
- 高影響項目（initialization、BN、weight-decay scope、option A tensor semantics、LR boundary）必須在正式訓練前決定並測試。
- 不得用 CIFAR-10 test 結果在替代方案間做選擇；選擇應依主要來源、原作者補充材料或預先聲明的工程理由。

## 主要來源查證對候選假設的影響

| Assumption ID | 查證結論 | 影響 | 對應 decision |
| --- | --- | --- | --- |
| ASSUMP-INITDIST-001 | CONFIRMED_PRIMARY_SOURCE | normal/Gaussian與std公式不再是未查證候選；只剩fan mode與逐層適用範圍。 | DEC-INIT-001 |
| ASSUMP-FAN-001 | PROJECT_DECISION_REQUIRED | [13] 同時認可fan-in與fan-out，無法宣稱唯一原設定。 | DEC-INIT-002 |
| ASSUMP-FCINIT-001 | STILL_UNKNOWN | std`0.01`已批准為`LOW_CONFIDENCE_ASSUMPTION`，不能升格為CIFAR原始事實。 | DEC-FC-001（APPROVED_BY_USER） |
| ASSUMP-AUGPAD-001 | CONFIRMED_PRIMARY_SOURCE | [24] 已確認constant zero padding。 | DEC-AUG-001 |
| ASSUMP-AUGORDER-001 | PROJECT_DECISION_REQUIRED | 幾何順序獨立為raw→zero-pad→crop→flip候選。 | DEC-AUG-001 |
| ASSUMP-MEANORDER-001 | STILL_UNKNOWN | mean-first順序已批准為project assumption；主要來源仍未確認作者順序。 | DEC-MEANORDER-001（APPROVED_BY_USER） |
| ASSUMP-CONVPAD-001 | PARTIALLY_CONFIRMED | shape constraint與作者ImageNet deploy支持pad1，仍無CIFAR prototxt。 | DEC-CONV-001 |
| ASSUMP-CONVBIAS-001 / ASSUMP-FCBIAS-001 | PARTIALLY_CONFIRMED | 作者ImageNet deploy支持conv no-bias、FC with-bias；project choices已批准，CIFAR paper fact仍未知。 | DEC-INIT-003 / DEC-FCBIAS-001 |
| ASSUMP-WD-BIAS-001 / ASSUMP-WD-BN-001 | PROJECT_DECISION_REQUIRED | Caffe default decay_mult=1可確認；作者CIFAR overrides未知。 | DEC-WD-001 |
| ASSUMP-SCALE-001 / ASSUMP-MEAN-001 | PARTIALLY_CONFIRMED | Caffe default與官方CIFAR example支持0..255、full-shape train mean、train/test共用；作者pipeline未知。 | DEC-PRE-001 |
| ASSUMP-LRSTEP-001 | PROJECT_DECISION_REQUIRED | Caffe framework boundary已確認；本專案採用同一boundary仍是ASSUMPTION。 | DEC-LR-001 |
| ASSUMP-BNEPS-001 / ASSUMP-BNAFF-001 | PARTIALLY_CONFIRMED | Caffe defaults與作者ImageNet deploy/README已知；single GPU/non-SyncBN與evaluation policy是project decision。 | DEC-BN-001A |
| ASSUMP-BNMOM-001 / ASSUMP-BNCOMP-001 | PROJECT_DECISION_REQUIRED | 作者CIFAR BN細節仍未知；專案已批准`momentum=0.001`加scaled-accumulator/scale-factor語意，mandatory compatibility test仍為正式訓練gate。不得把project PASS改寫成paper fact。 | DEC-BN-001B / DEC-BN-001C（APPROVED_BY_USER） |
| ASSUMP-WSEED-001 / ASSUMP-RNG-002 | PROJECT_DECISION_REQUIRED | Paper未指定worker RNG或mid-epoch resume；專案採per-sample identity seed與consumed cursor以消除worker/prefetch依賴。 | DEC-RNG-002（APPROVED_BY_USER） |
| ASSUMP-SHORT-001 / ASSUMP-SHORTPAD-001 | STILL_UNKNOWN | 無作者官方CIFAR Option A tensor實作可查。 | DEC-SHORT-001 |
