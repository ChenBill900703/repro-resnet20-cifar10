# 主要來源補充查證

## 1. 範圍、方法與證據等級

本文件只使用專案允許的主要來源，針對 `docs/open_questions.md` 的 `BLOCKER` 與 `HIGH` 問題補充查證。未使用部落格、第三方 PyTorch CIFAR ResNet 實作或不明 repository，也沒有用現代 PyTorch 預設值反推原論文設定。

結論狀態只使用：

- `CONFIRMED_PRIMARY_SOURCE`：允許的主要來源直接確認該主張。
- `PARTIALLY_CONFIRMED`：主要來源確認部分語意，但不足以證明原作者 CIFAR-10 實驗的完整設定。
- `STILL_UNKNOWN`：查閱允許來源後仍沒有足夠證據。
- `PROJECT_DECISION_REQUIRED`：原始事實無法確認，必須由專案明確選擇；是否已批准以`decision_log.md`為準，不會因此把paper fact升格為已確認。

重要限制：作者官方 repository 只發布 ImageNet 的 ResNet-50/101/152 deploy prototxt，README 並明示釋出模型是由作者自己的實作轉換到 Caffe b590f1d、不是以該版 Caffe 訓練。故 Caffe b590 原始碼可以確認「該版 Caffe 的語意」，ImageNet deploy prototxt可以確認「釋出模型的 deploy 結構」，但兩者都不能單獨證明原作者 CIFAR-10 Plain-20/ResNet-20 的訓練設定。

## 2. 查閱的主要來源

| Source ID | 來源 | 官方 URL / repository path | 本次使用範圍 |
| --- | --- | --- | --- |
| PS-01 | He et al., *Deep Residual Learning for Image Recognition*, arXiv v1 | `references/1512.03385v1.pdf`；<https://arxiv.org/pdf/1512.03385v1> | CIFAR 架構、Option A 文字定義、訓練與資料處理、BN placement |
| PS-02 | He et al., *Delving Deep into Rectifiers*（引用 [13]） | <https://arxiv.org/pdf/1502.01852> | 初始化公式、分布、fan-in/fan-out、bias；該文自身 FC 例外 |
| PS-03 | Lee et al., *Deeply-Supervised Nets*（引用 [24]） | <https://arxiv.org/pdf/1409.5185> | CIFAR augmentation 的 zero padding、crop、flip 與 test crop |
| PS-04 | Kaiming He 官方 `deep-residual-networks` repository | <https://github.com/KaimingHe/deep-residual-networks> | 官方發布範圍、Caffe commit、BN/Scale 說明、釋出模型轉換限制 |
| PS-05 | 作者官方 ResNet-50 deploy prototxt | <https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt> | ImageNet deploy 的 conv padding/bias、BN/Scale placement、test BN、FC 欄位；僅作旁證 |
| PS-06 | BVLC Caffe 官方 repository，commit `b590f1d27eb5cbd9bc7b9157d447706407c68682` | <https://github.com/BVLC/caffe/tree/b590f1d27eb5cbd9bc7b9157d447706407c68682> | solver iteration/LR/snapshot、MSRA filler、ParamSpec、BN、Scale、多 GPU、data transformer、官方 CIFAR example |

本次沒有找到作者正式 supplementary material、CIFAR-10 training prototxt、solver、training script、log 或 Option A 原始實作。作者 repository README 所列的第三方實作未被採用為證據。

## A. Initialization

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 BLOCKER |
| --- | --- | --- | --- | --- | --- |
| A-01 | [13] 對 ReLU convolution 的 forward derivation 給出 `n = k^2 c`，令每個權重為零均值 Gaussian，標準差 `sqrt(2/n)`。 | CONFIRMED_PRIMARY_SOURCE | PS-02，Section 2.2，Eq. (10)，PDF pp.2-3；`n=k^2c` 與「zero-mean Gaussian distribution whose standard deviation is sqrt(2/n)」 | 明確是 normal/Gaussian，不是 uniform；此處 `n` 是 fan-in。 | 部分：解除 distribution 與 forward formula 子問題。 |
| A-02 | [13] 的 backward derivation另給 `n_hat = k^2 d` 與標準差 `sqrt(2/n_hat)`，並明說 Eq. (10) 或 Eq. (14) 任一種都足以讓模型收斂。 | CONFIRMED_PRIMARY_SOURCE | PS-02，Section 2.2，Eq. (14) 後文字，PDF p.3 | fan-out 版本也被原文認可；[13] 沒有唯一指定 CIFAR ResNet 必須使用 fan-in 或 fan-out。 | 否：原作者 ResNet 實際 fan mode 仍未知。 |
| A-03 | Caffe b590 的 `MSRAFiller` 使用零均值 Gaussian、`sqrt(2/n)`，預設 `FAN_IN`，也可顯式選 `FAN_OUT`/`AVERAGE`。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`include/caffe/filler.hpp` L181-L207；`src/caffe/proto/caffe.proto` 的 `FillerParameter.variance_norm` | 確認 Caffe 實作候選的語意；不能證明作者 CIFAR prototxt 使用了 `MSRAFiller` 或其預設。 | 否。 |
| A-04 | [13] 的 convolution derivation令 bias `b=0`；這支持 convolution bias 的零初始化，但不證明 CIFAR ResNet 是否建立 bias parameter。 | PARTIALLY_CONFIRMED | PS-02，Section 2.2，forward derivation，PDF p.2 | 「bias 初值為零」與「完全沒有 bias parameter」是不同主張。 | 否。 |
| A-05 | [13] 沒有給出可直接套用到 CIFAR ResNet 單一 10-way FC 的通用初始化規則；該文在自己的 ImageNet model 對前兩個 FC 使用 std `0.01`、最後 FC 使用 std `0.001`，並說這是因輸入沒有 normalization、避免 overflow。 | STILL_UNKNOWN | PS-02，Section 3，Implementation Details，PDF p.4 | 該 FC 設定是 [13] 特定 ImageNet 架構的例外，不能無條件移植；Caffe b590 `MSRAFiller` 註解也明說其 shape 假設目前不適合 InnerProduct。 | 否：`Q-FCINIT-001` 未解除。 |
| A-06 | 作者官方 repository 無 CIFAR-10 Plain-20/ResNet-20 training prototxt，因此不能逐層確認 filler、bias 或 FC 初始化。 | STILL_UNKNOWN | PS-04，README 的 repository scope 與 prototxt 清單；PS-05 僅為 ImageNet deploy | deploy prototxt不含 training filler，且 README 明說不是以該版 Caffe 訓練。 | 否：`Q-INIT-001` 仍為部分未解除。 |
| A-07 | 專案已選定fan-in、FC normal std`0.01`、conv no-bias及FC bias0。 | PROJECT_DECISION_REQUIRED | A-01至A-06；相關decisions均APPROVED_BY_USER | FC std`0.01`保留LOW_CONFIDENCE_ASSUMPTION；不得標成原論文事實。 | Paper fact仍未知；須initialization tests。 |

## B. Option A shortcut

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 BLOCKER |
| --- | --- | --- | --- | --- | --- |
| B-01 | ResNet 論文明示跨 feature-map size 的 shortcut 使用 stride 2，option A 以額外 zero entries 增加維度且不引入參數；CIFAR 全部 shortcut 採 option A。 | CONFIRMED_PRIMARY_SOURCE | PS-01，Sections 3.3、4.2，PDF pp.4,7；Eqs. (2)/(3) 附近的 options A/B/C 描述 | 確認無參數、stride 2、zero entries 與 CIFAR 全用 option A。 | 部分。 |
| B-02 | 允許來源沒有說 stride-2 identity shortcut 是固定 `::2`、pooling 或其他無參數操作。 | STILL_UNKNOWN | PS-01，Sections 3.3、4.2；PS-04 repository 無 CIFAR 原始實作 | 「performed with a stride of 2」不足以唯一決定 tensor 索引。 | 否。 |
| B-03 | 允許來源沒有說新增 channel 的 zero padding 是前後對稱、單側或其他配置，也沒有說空間抽樣與 channel padding 的實作順序。 | STILL_UNKNOWN | PS-01，Section 3.3「padding extra zero entries」；PS-04 無相關 CIFAR source | 只能確認多出的 entries 為 zero，不能確定位置。 | 否。 |
| B-04 | 作者官方 repository 沒有可驗證 Option A 的 CIFAR prototxt/source。 | STILL_UNKNOWN | PS-04，README/prototxt directory；只含 ResNet-50/101/152 ImageNet deploy | ImageNet deploy 使用 projection branches，不能作為 CIFAR option A 證據。 | 否：`Q-SHORT-001` 未解除。 |
| B-05 | 專案已批准偶數位置抽樣、對稱channel zero padding及odd-size報錯。 | PROJECT_DECISION_REQUIRED | B-01至B-04；`DEC-SHORT-001`（APPROVED_BY_USER） | 遵守論文明示約束，但tensor細節仍是project assumption。 | Paper fact仍未知；實作須通過shortcut tests。 |

## C. Convolution details

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 BLOCKER |
| --- | --- | --- | --- | --- | --- |
| C-01 | CIFAR 每個 stage 內的 `3x3` convolution 必須維持對應的 `32/16/8` spatial size；padding 1 可由 shape constraint 推導。作者 ImageNet deploy 的所有 `3x3` convolution 亦明列 `pad: 1`。 | PARTIALLY_CONFIRMED | PS-01，Section 4.2 architecture；PS-05，例如 `res2a_branch2b` 等 `3x3` layers 的 `pad: 1` | shape 推導加官方 deploy 慣例高度支持 padding 1，但沒有 CIFAR training prototxt 的直接欄位證據。 | 部分；仍需把實作選擇記為專案決定。 |
| C-02 | 作者 ImageNet deploy 的 convolution 明列 `bias_term: false`，而 BN 後接 Scale；這不能直接證明 CIFAR training convolution 也無 bias。 | PARTIALLY_CONFIRMED | PS-05，所有 Convolution layer 的 `bias_term: false`；README 的 BatchNorm+Scale 說明 | 只有 ImageNet deploy 旁證；Caffe convolution 預設其實是 `bias_term: true`。 | 否。 |
| C-03 | 作者 ImageNet deploy 的 `fc1000` 未覆寫 `bias_term`；Caffe `InnerProductParameter.bias_term` 預設 `true`，因此該 deploy 的 FC 有 bias。CIFAR FC 是否相同仍無直接證據。 | PARTIALLY_CONFIRMED | PS-05，layer `fc1000`；PS-06，`src/caffe/proto/caffe.proto` L779-L783 | 可確認 deploy/Caffe default，不可確認 CIFAR training。 | 否。 |
| C-04 | 作者 README 說 BN 由 Caffe BatchNorm（無可訓練 gamma/beta）加 Scale（學習 gamma/beta）組成；Caffe Scale 預設 gamma=1，若 `bias_term:true` 則 beta filler 預設 0。 | PARTIALLY_CONFIRMED | PS-04，README「The BN layers are implemented using BatchNorm and Scale」；PS-06，`scale_layer.cpp` L24-L64、`caffe.proto` L997-L1031 | 確認 Caffe b590 預設與官方 deploy 組合，未直接確認 CIFAR training 是否覆寫 filler。 | 否。 |
| C-05 | 原論文整體 implementation 明說在每個 convolution 後、activation 前採 BN；CIFAR 段落明說採 BN，因此 CIFAR 的每個 convolution 後都有 BN。 | CONFIRMED_PRIMARY_SOURCE | PS-01，Section 3.4「We adopt batch normalization right after each convolution and before activation」；Section 4.2「We adopt batch normalization but no dropout」 | 這項不需依賴第三方實作。 | 是：解除 `Q-PLAINBN-001`。 |
| C-06 | 專案已批准padding 1、conv no-bias、FC with-bias、gamma=1/beta=0作為實作設定。 | PROJECT_DECISION_REQUIRED | C-01至C-04；相關decisions均APPROVED_BY_USER | Bias欄位仍只是project assumption；須parameter/initialization tests。 | Paper fact仍未知。 |

## D. Weight decay

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 HIGH |
| --- | --- | --- | --- | --- | --- |
| D-01 | Caffe b590 對每個 learnable parameter 使用 `local_decay = solver.weight_decay * ParamSpec.decay_mult`；`decay_mult` 預設 1。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`src/caffe/solvers/sgd_solver.cpp` L145-L160；`src/caffe/proto/caffe.proto` L281-L303 | 確認該版 Caffe 的 L2 scope/default 語意。 | 部分。 |
| D-02 | 若 training prototxt 未覆寫 `decay_mult`，convolution weight、FC weight、FC bias、Scale gamma、Scale beta 都會使用全域 weight decay；但原作者 CIFAR training prototxt不可得，無法知道是否覆寫。 | PARTIALLY_CONFIRMED | PS-06，ParamSpec default 與 SGDSolver regularization；PS-04 缺少 CIFAR training files | Caffe default 不是原作者實際設定的證明。 | 否。 |
| D-03 | 作者 ImageNet deploy prototxt沒有 training ParamSpec/solver 證據可決定 CIFAR weight、bias 或 BN affine 的 decay scope。 | STILL_UNKNOWN | PS-05（deploy only）；PS-04 README 的 conversion caveat | 不能從 deploy 欄位推回 training decay multipliers。 | 否：`Q-WD-001` 未解除。 |
| D-04 | 專案已批准對所有learnable parameters套用weight decay。 | PROJECT_DECISION_REQUIRED | D-01至D-03；`DEC-WD-001`（APPROVED_BY_USER） | 仍是`caffe-default-derived assumption`，不得稱為paper-faithful fact。 | Paper fact仍未知；須optimizer-group test。 |

## E. Input preprocessing

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 HIGH |
| --- | --- | --- | --- | --- | --- |
| E-01 | Caffe b590 `DataTransformer` 將 byte datum 轉為 `uint8` 數值後做 `(pixel - mean) * scale`，`scale` 預設 1，因此其預設數值路徑是 `0..255` 而非先除以 255。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`src/caffe/data_transformer.cpp` L42-L123；`src/caffe/proto/caffe.proto` L401-L415 | 只確認 Caffe default；沒有作者 CIFAR data layer/prototxt 證明實驗未覆寫 scale。 | 部分。 |
| E-02 | Caffe 官方 CIFAR example 從 training LMDB 計算 `mean.binaryproto`，train/test data layer 使用同一 mean file；DataTransformer 要求 mean file channel/height/width 與 datum 相同，即 per-pixel `C x H x W` mean。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`examples/cifar10/create_cifar10.sh` L14-L17；`cifar10_quick_train_test.prototxt` L2-L34；`data_transformer.cpp` L60-L65 | 確認官方 Caffe CIFAR example 的做法，不是作者 ResNet CIFAR pipeline 的直接證據。 | 部分。 |
| E-03 | 原 ResNet 論文只明說 per-pixel mean subtraction；原始 pixel scale、mean 是 45k 或 50k、mean artifact shape，以及 test 是否使用同一 artifact 仍沒有作者 CIFAR 設定檔直接證據。 | PARTIALLY_CONFIRMED | PS-01，Section 4.2；E-01/E-02 | Caffe官方慣例提供可信候選，但不能升格為原實驗事實。 | 否：`Q-MEAN-001` 仍未解除。 |
| E-04 | [24] 明說 CIFAR augmentation 是每側 zero padding 4 pixels，再做 cropping 與 random flipping；因此 ResNet 所稱 follow [24] 的 padding mode 可確認為 zero。 | CONFIRMED_PRIMARY_SOURCE | PS-03，Section 4.1 CIFAR-10/100，PDF p.6：「zero padding 4 pixels on each side, then do corner cropping and random flipping」；PS-01，Section 4.2 | ResNet 已自行重述 random `32x32` crop，故只採 [24] 的 zero-padding 細節，不採其 global contrast normalization 或 corner-only crop。 | 是：解除 `Q-PAD-001`。 |
| E-05 | 主要來源支持zero padding、random crop及horizontal flip；專案已批准幾何子序列zero-pad→crop→flip。 | PROJECT_DECISION_REQUIRED | PS-03；PS-01；`DEC-AUG-001`（APPROVED_BY_USER） | 只處理幾何augmentation；完整順序見下一列。 | Paper細節仍未知。 |
| E-06 | Mean相對位置未被主要來源確認；專案批准mean-first：float`[0,255]`→subtract mean→zero-pad→crop→flip。 | STILL_UNKNOWN | PS-01；PS-04無CIFAR script；`DEC-MEANORDER-001`（APPROVED_BY_USER） | 為保留`3x32x32`mean座標語意的project assumption；padding 0代表centered zero。 | Paper fact仍未知；config已凍結。 |
| E-07 | Raw scale、完整50k`3x32x32`mean、train/test共用mean及完整transform順序均已由使用者批准。 | PROJECT_DECISION_REQUIRED | `DEC-PRE-001`、`DEC-AUG-001`、`DEC-MEANORDER-001` | 不得包裝成原作者pipeline已確認。 | 須preprocessing tests。 |

## F. Learning-rate boundary

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 BLOCKER |
| --- | --- | --- | --- | --- | --- |
| F-01 | Caffe b590 solver 的 `iter_` 初始化為 0，且註解明定其值表示「weights 已更新的次數」。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`src/caffe/solver.cpp` L45-L64、L194-L258 | `ApplyUpdate()` 後才 `++iter_`。 | 是，作為 Caffe 語意。 |
| F-02 | `multistep` 在 `GetLearningRate()` 中於 `iter_ >= stepvalue` 先增加 step，再計算 LR；`GetLearningRate()` 在當次 `ApplyUpdate()` 前呼叫。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`src/caffe/solvers/sgd_solver.cpp` L27-L50、L102-L115 | `stepvalue: 32000` 時，`iter_=32000` 的下一次更新第一次用新 LR。 | 是。 |
| F-03 | 在Caffe b590 framework semantics下，updates #1..#32,000使用0.1，#32,001..#48,000使用0.01，#48,001..#64,000使用0.001。 | CONFIRMED_PRIMARY_SOURCE | F-01/F-02與PS-01的base LR、milestones、gamma | 只確認「若使用該Caffe multistep語意」時的boundary，不證明原作者CIFAR solver實際採用同一語意。 | 解除framework-semantics子問題。 |
| F-04 | `Solve()` 執行 `Step(max_iter - iter_)`；當 `max_iter=64000` 時正好完成 64,000 次更新後停止。Snapshot interval 在 `++iter_` 後檢查；若最後 iteration 沒有 interval snapshot，`snapshot_after_train` 預設 true 會在 optimization 後另存 snapshot。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`solver.cpp` L194-L300；`caffe.proto` 的 `snapshot_after_train` default | Snapshot filename iteration 對應已完成更新數。 | 是。 |
| F-05 | 原作者CIFAR是否使用Caffe b590相同boundary仍無直接證據；本專案已批准採用該boundary。 | PROJECT_DECISION_REQUIRED | PS-04；`DEC-LR-001`（APPROVED_BY_USER） | framework fact與project adoption分列，不宣稱paper-faithful fact。 | 須boundary test。 |

## G. Batch Normalization

| ID | 結論 | 狀態 | 主要來源與定位 | 證據摘要 | 能否解除 HIGH |
| --- | --- | --- | --- | --- | --- |
| G-01 | Caffe b590 的多 GPU `P2PSync` 為每張 GPU 建立 solver/net replica，在 backward 後只匯總 learnable-parameter gradients；沒有跨 replica 聚合 forward BN activation statistics 的路徑。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`src/caffe/parallel.cpp` L268-L283、L325-L378；`batch_norm_layer.cpp` L94-L134 | 對「Caffe b590 P2PSync」而言，training BN statistics 是每 replica/local。 | 部分。 |
| G-02 | 不能據此確認原作者 two-GPU CIFAR 實驗使用 local BN，因作者 README 明說釋出模型不是使用 repository 指向的 Caffe b590 訓練；也沒有原作者 CIFAR training source。 | STILL_UNKNOWN | PS-04 README conversion caveat；PS-01 只寫 two GPUs | 原作者自有實作可能有不同 multi-GPU BN 行為。 | 否：`Q-BN-001` 未解除。 |
| G-03 | Caffe b590 BatchNorm 預設training使用batch statistics並以`moving_average_fraction=.999`累積，`eps=1e-5`；TEST預設`use_global_stats=true`，以stored statistics評估。 | CONFIRMED_PRIMARY_SOURCE | PS-06，`caffe.proto` L491-L500；`batch_norm_layer.cpp` L10-L21、L86-L139 | 只確認Caffe b590語意；PyTorch`momentum=0.001`與測試流程已批准，但running variance與scale-factor等價性仍須mandatory preflight證明。 | 部分。 |
| G-04 | 作者 repository 的 ImageNet released models 使用 BatchNorm+Scale，且 README 說釋出的 BN mean/variance 是訓練後以足夠大的 training batch 重新平均計算，不是 moving average。這不證明 CIFAR-10 報告模型也做了相同 recalibration。 | PARTIALLY_CONFIRMED | PS-04 README 的 BN paragraphs | 顯示作者釋出 ImageNet model 的 eval-stat procedure與純 Caffe default不同。 | 否。 |
| G-05 | 原作者 CIFAR 實驗的同步 BN、有無 post-training recalibration、epsilon、moving-average設定與 test running-statistics來源仍未知。 | STILL_UNKNOWN | G-01 至 G-04 | 只可把 Caffe default 作為候選，不能冒充實驗事實。 | 否：`Q-BNHP-001` 未完全解除。 |
| G-06A | single GPU、non-SyncBN、gamma=1/beta=0、eps`1e-5`、running-stat evaluation且不做recalibration已批准為project decision。 | PROJECT_DECISION_REQUIRED | `DEC-BN-001A`（APPROVED_BY_USER） | 不是作者two-GPU CIFAR設定；須揭露偏離。 | Paper fact仍未知。 |
| G-06B | PyTorch`momentum=0.001`與mandatory compatibility test已批准；它只近似Caffe`.999`。 | PROJECT_DECISION_REQUIRED | `DEC-BN-001B`（APPROVED_BY_USER） | Test未通過禁止正式訓練；approval不代表已證明完全等價。 | Execution gate仍有效。 |

## 3. BLOCKER / HIGH 結果摘要

### 已解除（非 BLOCKER）

- `Q-PAD-001`（HIGH）：[24] 明確寫 zero padding 4 pixels on each side。
- `Q-PLAINBN-001`（MEDIUM）：原 ResNet 論文 Section 3.4 與 CIFAR Section 4.2 結合可確認每個 convolution 後都有 BN。
- `Q-CAFFE-001`（HIGH，來源資格/可取得性問題）：作者官方 repository 與其指定 Caffe commit 已定位；同時確認它們沒有 CIFAR training files，且不可把 Caffe default 倒推為作者實驗設定。

### 部分解除但仍須決策或更多原作者材料

- `Q-LR-001`：Caffe framework semantics已確認；原作者CIFAR是否相同仍未知，本專案已批准相同boundary為assumption。
- `Q-INIT-001`：normal/Gaussian、`sqrt(2/n)` 與 conv bias=0 有主要來源；確切 fan mode、FC 與是否存在 bias parameter 未確認。
- `Q-CONV-001`：padding 1 有 shape 推導與官方 ImageNet deploy 旁證；CIFAR conv/FC bias 未確認。
- `Q-AUG-001`：zero padding與高階順序已確認；精確 RNG/操作程序未確認。
- `Q-BNHP-001`：Caffe b590 defaults 已確認；作者 CIFAR 實際值與 eval recalibration 未確認。
- `Q-WD-001`：Caffe decay multiplier 語意已確認；作者 CIFAR 各 parameter scope 未確認。
- `Q-MEAN-001`：Caffe官方 CIFAR example 支持 full train mean file 與 train/test共用；作者 ResNet CIFAR pipeline 未確認。

### 仍未解除

- `Q-SHORT-001`：作者具體tensor semantics仍未知；專案assumption已批准。
- `Q-FCINIT-001`：作者CIFAR FC initialization仍未知；專案std`0.01`已批准為LOW_CONFIDENCE_ASSUMPTION。
- `Q-BN-001`（HIGH）：原作者 two-GPU CIFAR BN 是 local 或 synchronized。
- `Q-FULLTRAIN-001`（HIGH）：schedule決定後是否以完整 50k 重新訓練。
- `Q-EVAL-001`（HIGH）：8.75% checkpoint selection form。

## 4. 本階段界線

本文件沒有建立或修改模型程式、資料程式、訓練程式，沒有下載 CIFAR-10、執行訓練或修改 Python 套件。`docs/reproduction_spec.md` 未修改。
