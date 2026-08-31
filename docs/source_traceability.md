# Source Traceability

初始規格來源為 `references/1512.03385v1.pdf`（arXiv:1512.03385v1）。2026-08-03 的補充查證另使用論文引用 [13]/[24]、作者官方 repository 與其指定的 BVLC Caffe commit；完整清單與適用限制見 `docs/primary_source_review.md`。Caffe default 只證明框架語意，不自動等同作者 CIFAR 實驗設定。

Status 定義：

- `EXPLICIT`：原文明確寫出。
- `DERIVED`：可由原文數值、公式或結構直接推導。
- `ASSUMPTION`：論文不足，但未來實作必須決定。
- `UNKNOWN`：現階段無法可靠確定。

| Requirement ID | 設定或主張 | Status | Source section/page/table/figure | Evidence summary | Target implementation/test |
| --- | --- | --- | --- | --- | --- |
| GOAL-001 | Plain network 隨深度增加出現 degradation problem，且不是單純 overfitting | EXPLICIT | Introduction，PDF p.1；Figure 1 | 更深 plain net 有更高 training error 與 test error | 比較 Plain-20/更深模型時的 training curves；本階段至少保留 Plain/Residual 對照 |
| GOAL-002 | Residual learning 以 `F(x)=H(x)-x`、輸出 `F(x)+x` 改寫目標 | EXPLICIT | Section 3.1，PDF p.3 | 原文公式與最佳化動機 | Residual-block architecture test |
| DATA-001 | CIFAR-10 有 50k training images、10k test images、10 classes | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Dataset metadata assertion |
| DATA-002 | Network input 為 `32x32` image | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Input-shape test |
| DATA-003 | 使用 per-pixel mean subtraction | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Transform/config inspection |
| DATA-004 | Per-pixel mean 由 training set 計算 | ASSUMPTION | Section 4.2，PDF p.7；`AGENTS.md` | PDF 未寫統計集合；專案規則指定 training set | Mean artifact provenance test |
| DATA-005 | 不使用 standard-deviation normalization | UNKNOWN | Section 4.2，PDF p.7 | CIFAR preprocessing 未提 std normalization | Frozen-config review；不得標成論文明示 |
| DATA-006 | Tensor pixel-value range | UNKNOWN | ResNet Section 4.2，PDF p.7；Caffe b590 `data_transformer.cpp` L42-L123 | ResNet未寫；Caffe default可確認uint8 `0..255`、scale1，但作者CIFAR是否覆寫未知 | Transform unit test after decision |
| DATA-007 | 45k/5k train/val split 用來決定 training schedule | EXPLICIT | Section 4.2，PDF p.7 | 原文說 64k schedule determined on 45k/5k split | Protocol documentation |
| DATA-008 | Table 6 final models 重新使用完整 50k 訓練 | UNKNOWN | Section 4.2，PDF p.7 | 沒有明寫 schedule selection 後的 retraining procedure | Open-question resolution |
| AUG-001 | Training 每側 padding 4 pixels | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Augmentation geometry test |
| AUG-002 | 從 padded image 隨機取 `32x32` crop | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Crop output-shape/distribution test |
| AUG-003 | Random horizontal flip | EXPLICIT | Section 4.2，PDF p.7 | 原文寫 padded image or its horizontal flip | Transform test |
| AUG-004 | Padding mode為constant zero | EXPLICIT | [24] Section 4.1，PDF p.6；ResNet Section 4.2 follow [24] | [24] 明寫「zero padding 4 pixels on each side」 | Edge-pixel test |
| AUG-005 | 幾何子序列採zero-pad→random crop→random flip | ASSUMPTION | [24] Section 4.1；ResNet Section 4.2；`DEC-AUG-001` | 主要來源確認操作但無作者script；project decision已批准 | Transform-order test |
| DATA-009 | Mean subtraction相對於augmentation的位置 | UNKNOWN | ResNet Section 4.2；作者官方repo無CIFAR data script；`DEC-MEANORDER-001` | Paper fact未知；project已批准mean-first，padding發生於centered image | Frozen-config/preprocessing test |
| DATA-010 | Training/test DataLoader sampling policy | ASSUMPTION | ResNet CIFAR論文未指定shuffle、sampler或epoch boundary；`DEC-SHUFFLE-001` | Project批准training shuffle、每次完整traversal重排、non-replacement與stateful sampler；test sequential official order | `DATA-008`至`DATA-012` |
| DATA-011 | Mid-epoch exact resume與consumed cursor | ASSUMPTION | Paper未指定DataLoader/prefetch/checkpoint runtime；`DEC-RNG-002` | Checkpoint保存permutation、consumed cursor、epoch與generator state；issued/prefetched未消耗batch在resume後重發，optimizer update成功後才acknowledge | `DATA-011`、`DATA-013`、`DATA-014`、`CKPT-003` |
| AUG-006 | Per-sample stochastic augmentation RNG | ASSUMPTION | Paper只要求random crop/flip，未指定RNG mapping；`DEC-RNG-002` | Seed由base seed、epoch、official sample index派生；Dataset顯式持有epoch且batch攜帶indices；與worker assignment/prefetch無關 | `DATA-015`至`DATA-018` |
| EVAL-001 | Test 使用原始 `32x32` image single view | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Test-transform inspection |
| EVAL-002 | 不使用 multi-crop test | DERIVED | Section 4.2，PDF p.7 | single view 排除 multi-crop | Test-transform count assertion |
| EVAL-003 | 指標為 CIFAR-10 test error | EXPLICIT | Table 6，PDF p.7 | Caption 為 classification error on test set | Metric unit test |
| EVAL-004 | ResNet-20 8.75% 的 checkpoint selection form | UNKNOWN | Table 6，PDF p.7 | 未說 best/last/mean | 報告中保留 UNKNOWN；專案另固定 final |
| ARCH-001 | CIFAR depth formula 為 `6n+2` weighted layers | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Weighted-layer count test |
| ARCH-002 | Target `n=3` 對應 depth 20 | EXPLICIT | Section 4.2，PDF p.7 | 原文說 `n={3,5,7,9}` leads to 20/32/44/56 | Config assertion |
| ARCH-003 | 20 weighted layers = 19 convolution + 1 FC | DERIVED | Section 4.2，PDF p.7 | `1 + 6n + 1`，代入 `n=3` | Module-type count test |
| ARCH-004 | 三個 stage 的 convolution layers 為 `[6,6,6]` | DERIVED | Section 4.2，PDF p.7 | 每個 feature-map size 有 `2n` layers | Stage-depth test |
| ARCH-005 | 三個 stage 的 two-convolution groups/blocks 為 `[3,3,3]` | DERIVED | Section 4.2，PDF p.7 | 每兩個 `3x3` layers 一組，`n=3` | Block-count test |
| ARCH-006 | Feature-map sizes 為 `[32,16,8]` | EXPLICIT | Section 4.2，PDF p.7 | 原文集合 `{32,16,8}` | Forward shape hooks |
| ARCH-007 | Channels/filters 為 `[16,32,64]` | EXPLICIT | Section 4.2，PDF p.7 | 原文集合 `{16,32,64}` | Forward/module inspection |
| ARCH-008 | Initial layer 為 `3x3` convolution，輸出 16 channels | EXPLICIT | Section 4.2 architecture text/table，PDF p.7 | First layer + table `1+2n`, 16 filters | Stem inspection |
| ARCH-009 | 下採樣由 stride-2 convolution 執行 | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Stage-transition shape test |
| ARCH-010 | 結尾為 global average pooling、10-way FC、softmax | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Head structure/forward test |
| ARCH-011 | Pooling、BN、ReLU、addition 不計入 weighted layers | DERIVED | `6n+2` 定義與架構列舉，PDF p.7 | 公式只計 initial/stack convolution 與 FC | Layer-count test documentation |
| ARCH-012 | `3x3` convolution padding 1 | ASSUMPTION | ResNet Section 4.2；作者ImageNet `ResNet-50-deploy.prototxt` 各3x3 layer | 輸出尺寸約束與官方deploy `pad:1`支持，但無CIFAR training prototxt | Padding inspection + shape test |
| ARCH-013 | Convolution no-bias、FC with-bias | ASSUMPTION | 作者ImageNet deploy `bias_term:false`；`fc1000` + Caffe InnerProduct default | 只有官方ImageNet deploy旁證，CIFAR欄位仍未知 | Parameter-count test after approval |
| ARCH-PLAIN-001 | Plain-20 沒有 shortcut | EXPLICIT | Sections 3.3/4.2；Figure 3 middle vs right，PDF pp.4,7 | Shortcut 是 residual counterpart 的新增部分 | Assert no shortcut/addition modules |
| ARCH-PLAIN-002 | Plain-20 與 ResNet-20 depth、width、parameter count 相同 | EXPLICIT | Section 4.2，PDF p.7 | Option A 使 residual/plain counterparts 完全同參數 | Cross-model parameter-count equality |
| ARCH-RES-001 | 每個 residual block 為兩個 `3x3` convolution | EXPLICIT | Section 4.2，PDF p.7 | Shortcuts connected to pairs of `3x3` layers | Block structure test |
| ARCH-RES-002 | ResNet-20 shortcut 總數為 `3n=9` | EXPLICIT | Section 4.2，PDF p.7 | 原文明列 totally `3n` shortcuts | Shortcut count test |
| ARCH-RES-003 | CIFAR 所有 shortcut 使用 identity option A | EXPLICIT | Section 4.2，PDF p.7 | 原文明列 option A in all cases | Assert no projection parameters |
| ARCH-RES-004 | 維度增加時 option A 以 zero entries padding，無額外參數 | EXPLICIT | Section 3.3，PDF p.4 | Option A 定義 | Shortcut parameter test |
| ARCH-RES-005 | Shortcut 跨 feature-map sizes 時 stride 2 | EXPLICIT | Section 3.3，PDF p.4 | 對 options A/B 皆明列 stride 2 | Shortcut output-shape test |
| ARCH-RES-006 | Stage 2/3 的第一個 block 執行下採樣 | DERIVED | Section 4.2 architecture + block pairing，PDF p.7 | 需在進入新 spatial size 時下採樣 | Stage-boundary test |
| ARCH-RES-007 | Post-activation：addition 後有第二個 ReLU | EXPLICIT | Figure 2，PDF p.2；Section 3.2，PDF p.3 | 原文明寫 second nonlinearity after addition | Operation-order hook test |
| ARCH-RES-008 | 完整 block 順序 `conv-BN-ReLU-conv-BN-add-ReLU` | DERIVED | Sections 3.2/3.4，PDF pp.3-4；Figure 7，PDF p.8 | 合併 BN placement、Figure 2 與 response 定義 | Forward hooks/graph inspection |
| ARCH-RES-009 | Option A 具體 `::2` 索引與 symmetric channel padding | ASSUMPTION | Sections 3.3/4.2，PDF pp.4,7 | Tensor 細節未提供 | Dedicated shortcut value test |
| TRAIN-001 | Optimizer 為 SGD | EXPLICIT | Introduction/Section 3.4，PDF pp.1,4 | 原文直接寫 SGD | Optimizer type assertion |
| TRAIN-002 | Mini-batch size 128 | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Frozen config/global-batch test |
| TRAIN-003 | Paper 使用 two GPUs | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Deviation report；專案 single GPU |
| TRAIN-004 | Two-GPU per-device batch/BN synchronization | UNKNOWN | Section 4.2，PDF p.7 | 未說 batch split 或 synchronized BN | Open-question resolution |
| TRAIN-005 | Initial learning rate 0.1 | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Scheduler trace test |
| TRAIN-006 | 32k、48k 各將 LR 除以 10 | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Boundary scheduler test |
| FRAMEWORK-LR-001 | Caffe b590 multistep boundary：#32,001/#48,001首次使用新LR | EXPLICIT | Caffe b590 `solver.cpp` L194-L258；`sgd_solver.cpp` L27-L50、L102-L115 | 這是明確framework semantics：`iter_`為已完成updates，下一次ApplyUpdate前於`iter_>=stepvalue`切換 | Framework-semantics reference test |
| TRAIN-007 | 本專案採用Caffe b590相同LR boundary | ASSUMPTION | `FRAMEWORK-LR-001`；`DEC-LR-001` | 已批准#32,001/#48,001 boundary；不是已直接證實的作者設定 | Boundary test |
| TRAIN-008 | 64k iterations terminate | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Exact-update termination test |
| TRAIN-009 | Momentum 0.9 | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Optimizer config test |
| TRAIN-010 | Weight decay 0.0001 | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Optimizer config test |
| TRAIN-011 | 原作者 CIFAR weight decay parameter scope | UNKNOWN | ResNet Section 4.2；Caffe b590 `sgd_solver.cpp` L145-L160、`caffe.proto` L281-L303 | Caffe default decay_mult=1已知，但作者CIFAR overrides不可得 | Parameter-group test after approval |
| TRAIN-012 | ResNet-20 不使用 ResNet-110 warm-up | EXPLICIT | Section 4.2，PDF p.7 | Warm-up 明確由「In this case」限定 n=18/ResNet-110 | LR trace asserts first update uses 0.1 |
| TRAIN-013 | 64k x 128 = 8,192,000 presentations，約 163.84 個 50k-equivalent epochs | DERIVED | Section 4.2 values，PDF p.7 | 算術換算 | Report-only consistency calculation |
| INIT-001 | 採用文獻 `[13]` 的 weight initialization | EXPLICIT | Sections 3.4/4.2，PDF pp.4,7 | 原文直接引用 | Initialization provenance/config |
| INIT-002 | [13] convolution使用zero-mean Gaussian、std `sqrt(2/n)` | EXPLICIT | [13] Section 2.2，Eqs. (10)/(14)，PDF pp.2-3 | 明確排除uniform；forward式以fan-in，backward式以fan-out | Initialization distribution test |
| INIT-003 | 原作者ResNet實際使用fan-in或fan-out | UNKNOWN | [13] Section 2.2；Caffe`MSRAFiller`；`DEC-INIT-002` | Paper fact未知；project已批准fan_in | Initialization test |
| INIT-004 | CIFAR single FC initialization | UNKNOWN | [13] Section 3；Caffe`filler.hpp`；`DEC-FC-001` | Paper fact未知；project已批准normal std0.01為LOW_CONFIDENCE_ASSUMPTION | Initialization test + evidence audit |
| INIT-005 | Convolution bias zero/no parameter | ASSUMPTION | [13] Section 2.2設b=0；作者ImageNet deploy conv `bias_term:false` | 零初始化可確認，CIFAR是否建立bias parameter不可確認 | Parameter inspection after decision |
| BN-001 | BN 放在每個 convolution 後、activation 前 | EXPLICIT | Section 3.4，PDF p.4 | 原文直接列出 | BN placement inspection |
| BN-002 | CIFAR models 採 BN | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Assert BN enabled |
| BN-003 | CIFAR 每個 convolution 都有 BN | DERIVED | Sections 3.4/4.2，PDF pp.4,7 | 將通用 placement 套到 CIFAR adoption | Count BN == count convolution |
| BN-004 | 原作者CIFAR BN epsilon/momentum/affine/eval behavior | UNKNOWN | ResNet Sections 3.4/4.2；Caffe b590 sources；`DEC-BN-001A/B/C` | Paper fact未知；project values與scaled-accumulator implementation已批准，mandatory compatibility gate仍適用 | BN config/compatibility tests |
| BN-005 | Caffe-compatible scaled-accumulator implementation | ASSUMPTION | Caffe b590 accumulator/scale-factor framework semantics；`DEC-BN-001C` | Project保存scaled mean、scaled unbiased variance、running scale與counter；eval採accumulator/scale。Framework compatibility PASS不得改寫為paper EXPLICIT | `BN-003`至`BN-011` |
| REG-001 | 不使用 dropout | EXPLICIT | Sections 3.4/4.2，PDF pp.4,7 | 原文直接列出 | Assert no dropout module |
| LOSS-001 | 網路末端使用 softmax | EXPLICIT | Section 4.2，PDF p.7 | 原文直接列出 | Head semantics review |
| LOSS-002 | Exact loss implementation/reduction | UNKNOWN | Section 4.2，PDF p.7 | 只寫 softmax，未寫 loss layer | Loss unit test after review |
| RESULT-001 | ResNet-20 parameters 為 0.27M | EXPLICIT | Table 6，PDF p.7 | 表中直接列出 | Parameter-count comparison |
| RESULT-002 | ResNet-20 CIFAR-10 test error 為 8.75% | EXPLICIT | Table 6，PDF p.7 | 表中直接列出 | Final comparison report；不得承諾逐位相同 |
| RESULT-003 | Plain-20 精確最終 test error | UNKNOWN | Figure 6，PDF p.8；Table 6，PDF p.7 | Table 6 無此列，Figure 6 只有曲線 | 禁止建立精確 paper target |
| RESULT-004 | ResNet-110 為 5 runs，報 best 6.43% 與 mean +/- std 6.61 +/- 0.16% | EXPLICIT | Table 6/caption，PDF p.7 | Caption 明確定義 reporting form | 僅供文獻說明，不套用 ResNet-20 |
| ANALYSIS-001 | Figure 6 dashed 是 training error、bold 是 testing error | EXPLICIT | Figure 6 caption，PDF p.8 | Caption 直接定義 | Plot styling/legend documentation |
| ANALYSIS-002 | Layer response 是每個 `3x3` layer 經 BN 後、ReLU/addition 前的輸出 | EXPLICIT | Figure 7/corresponding text，PDF p.8 | 原文直接定義 | Optional analysis hook，非第一階段 blocker |
| PROJ-001 | 主要結果使用第 64,000 update final checkpoint | ASSUMPTION | `AGENTS.md`；論文 Section 4.2 只寫 terminate at 64k | 防止 best-test checkpoint selection 的專案規則 | Final-checkpoint/global-step assertion |
| PROJ-002 | 單 RTX 3070 Ti、FP32、AMP/TF32/compile disabled | ASSUMPTION | `AGENTS.md` | 本專案固定執行環境，不是論文設定 | Environment/config audit |
| PROJ-003 | 正式訓練前通過 architecture/shape/parameter/shortcut/forward/backward/smoke tests | ASSUMPTION | `AGENTS.md` | 專案品質閘門 | CI/preflight test suite |

## 補充主要來源定位

| Trace ID | 主張 | Status | 官方來源與精確定位 | Evidence summary | 對應問題/決定 |
| --- | --- | --- | --- | --- | --- |
| PSR-INIT-001 | He initialization為Gaussian，std `sqrt(2/fan_in)` | EXPLICIT | [13] <https://arxiv.org/pdf/1502.01852>，Section 2.2 Eq. (10)，PDF pp.2-3 | `n=k^2c`，zero-mean Gaussian | Q-INIT-001 / DEC-INIT-001 |
| PSR-INIT-002 | fan-out式也被[13]認可 | EXPLICIT | [13] Section 2.2 Eq. (14)及其後文字，PDF p.3 | `n_hat=k^2d`；任一式均可 | Q-INIT-001 / DEC-INIT-002 |
| PSR-INIT-003 | Caffe MSRA filler預設fan-in且為Gaussian | EXPLICIT | Caffe b590 `include/caffe/filler.hpp` L181-L207 | 支援FAN_IN/FAN_OUT/AVERAGE；InnerProduct有shape caveat | Q-INIT-001、Q-FCINIT-001 |
| PSR-AUG-001 | 每側zero padding 4 pixels | EXPLICIT | [24] <https://arxiv.org/pdf/1409.5185>，Section 4.1，PDF p.6 | zero pad後crop與random flip | Q-PAD-001 / DEC-AUG-001 |
| PSR-REPO-001 | 作者repo只提供ImageNet deploy且模型非用Caffe b590訓練 | EXPLICIT | <https://github.com/KaimingHe/deep-residual-networks>，README「Introduction」與「Notes」 | 無CIFAR train prototxt/solver；Caffe語意不得倒推 | Q-CAFFE-001 / DEC-CAFFE-001 |
| PSR-CONV-001 | 作者ImageNet deploy中3x3 conv pad1、conv no-bias、conv後BN+Scale | EXPLICIT | 作者repo `prototxt/ResNet-50-deploy.prototxt`，例如layers `res2a_branch2b`、後續3x3 layers | 只適用官方deploy旁證，不等同CIFAR training | Q-CONV-001 |
| PSR-WD-001 | Caffe local decay = global decay × decay_mult；default decay_mult=1 | EXPLICIT | Caffe b590 `src/caffe/solvers/sgd_solver.cpp` L145-L160；`src/caffe/proto/caffe.proto` L281-L303 | 每個learnable param可由ParamSpec覆寫 | Q-WD-001 / DEC-WD-001 |
| PSR-PRE-001 | Caffe default以uint8 0..255、先減mean再乘scale1 | EXPLICIT | Caffe b590 `data_transformer.cpp` L42-L123；`caffe.proto` L401-L415 | mean-file shape須與datum C/H/W相同 | Q-MEAN-001 / DEC-PRE-001 |
| PSR-PRE-002 | Caffe官方CIFAR example由train LMDB建mean並供train/test共用 | EXPLICIT | Caffe b590 `examples/cifar10/create_cifar10.sh` L14-L17；`cifar10_quick_train_test.prototxt` L2-L34 | 是框架官方example，不是作者ResNet config | Q-MEAN-001 / DEC-PRE-001 |
| PSR-LR-001 | Caffe b590 iter從0，代表已完成weight updates | EXPLICIT | Caffe b590 `src/caffe/solver.cpp` L45-L64、L194-L258 | ApplyUpdate後才遞增iter；只描述framework semantics | FRAMEWORK-LR-001 |
| PSR-LR-002 | Caffe b590 multistep在iter>=stepvalue時於當次update前切換LR | EXPLICIT | Caffe b590 `src/caffe/solvers/sgd_solver.cpp` L27-L50、L102-L115 | Caffe語意下#32,001首次使用0.01；不證明作者CIFAR solver採用同一設定 | FRAMEWORK-LR-001 |
| PSR-LR-003 | Caffe b590 max_iter與final snapshot語意 | EXPLICIT | Caffe b590 `src/caffe/solver.cpp` L278-L300；`caffe.proto` `snapshot_after_train` | Caffe語意下max_iter=64000完成64k更新；snapshot在iter遞增後 | FRAMEWORK-LR-001 / DEC-LR-001 |
| PSR-BN-001A | Caffe BN default eps1e-5、TEST global stats；專案single GPU/non-SyncBN及evaluation policy另為project decision | EXPLICIT | Caffe b590 `caffe.proto` L491-L500；`batch_norm_layer.cpp` L10-L139 | framework semantics只支持部分欄位，不證明作者two-GPU CIFAR設定 | Q-BN-001、Q-BNHP-001 / DEC-BN-001A |
| PSR-BN-001B | Caffe moving fraction .999、unbiased variance accumulator與TEST scale-factor語意 | EXPLICIT | Caffe b590`caffe.proto`與`batch_norm_layer.cpp`；`DEC-BN-001B/C` | Framework semantics支持專案scaled-accumulator reference；不證明作者CIFAR實際BN設定 | Q-BNHP-001 / DEC-BN-001B / DEC-BN-001C |
| PSR-BN-002 | Caffe P2PSync未同步forward BN activation stats | EXPLICIT | Caffe b590 `src/caffe/parallel.cpp` L268-L283、L325-L378 | 每GPU replica forward；只匯總parameter gradients | Q-BN-001 |
| PSR-BN-003 | 作者released ImageNet BN stats為訓練後重新平均，不是moving average | EXPLICIT | 作者repo README「Notes」BN段落 | 不能外推CIFAR | Q-BNHP-001 |
