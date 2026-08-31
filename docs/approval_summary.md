# Decision Approval Summary

## 1. Approval record

- Approval date：`2026-08-03`
- Baseline source commit：`f3e20eb`（由使用者指定；本Codex工作目錄未掛載Git metadata，未在本階段自行驗證commit object）
- Frozen config：`configs/cifar10_plain20_resnet20_frozen.yaml`
- Approval meaning：`APPROVED_BY_USER`只批准本專案的實作選擇，不代表原作者CIFAR設定已被完整恢復或主要來源已直接證實。
- Test-set discipline：所有批准均在未使用CIFAR-10 test表現選值的前提下完成；後續不得因test accuracy更換。

## 2. APPROVED_BY_USER decisions

| Decision ID | 批准內容摘要 | Risk | 證據邊界／execution gate |
| --- | --- | --- | --- |
| DEC-INIT-001 | Conv zero-mean normal | LOW | Distribution由[13]直接支持。 |
| DEC-INIT-002 | Conv fan-in、gain sqrt(2) | MEDIUM | [13]亦允許fan-out；作者實際選擇未知。 |
| DEC-INIT-003 | Conv no-bias | MEDIUM | 只有[13] bias=0與作者ImageNet deploy旁證。 |
| DEC-FC-001 | FC normal std0.01、bias0 | HIGH | `LOW_CONFIDENCE_ASSUMPTION`；作者CIFAR FC設定未知；不得因test結果更換。 |
| DEC-FCBIAS-001 | FC bias存在且為0 | MEDIUM | Caffe/ImageNet deploy旁證，不是CIFAR直接證據。 |
| DEC-SHORT-001 | Option A `::2`、symmetric zero padding、odd-size error | HIGH | Tensor semantics無作者正式CIFAR實作；須逐值測試。 |
| DEC-CONV-001 | 所有3x3 conv padding1 | LOW | Shape constraint與作者deploy強旁證。 |
| DEC-BN-001A | Single GPU/non-SyncBN、affine、gamma1/beta0、eps1e-5、running-stat eval、no recalibration | HIGH | Single GPU/non-SyncBN是project decision，不是作者two-GPU設定。 |
| DEC-BN-001B | BN update coefficient 0.001對應Caffe .999 | HIGH | Compatibility implementation與execution gate由`DEC-BN-001C`固定。 |
| DEC-BN-001C | Caffe-compatible scaled accumulators、running scale及de-biased eval | HIGH | Project framework-semantics decision；全部BN buffers須checkpoint；mandatory compatibility test未通過禁止正式訓練。 |
| DEC-WD-001 | Weight decay套用所有learnable parameters | HIGH | `caffe-default-derived assumption`，不是paper-faithful fact。 |
| DEC-PRE-001 | `[0,255]`、完整50k`3x32x32`mean、train/test共用、不做std normalization | HIGH | Caffe官方example旁證；作者ResNet CIFAR pipeline未知。 |
| DEC-AUG-001 | zero-pad→crop→flip幾何子序列 | MEDIUM | 操作受主要來源支持，精確程序為project decision。 |
| DEC-MEANORDER-001 | Mean-first完整train/test transform | HIGH | 作者順序未知；為保留per-pixel座標語意的project assumption。 |
| DEC-LR-001 | #32001/#48001切LR、#64000後final | MEDIUM | Caffe framework semantics已知；作者CIFAR solver設定未知。 |
| DEC-CAFFE-001 | Caffe只作framework semantics/assumption依據 | LOW | 防止把Caffe defaults冒充paper facts。 |
| DEC-SHUFFLE-001 | Training/test DataLoader sampling policy | MEDIUM | Training shuffle、non-replacement、base-seed-derived可恢復generator；test固定official order且無random sampler。Paper未指定。 |
| DEC-RNG-002 | Per-sample augmentation RNG與consumed-cursor exact resume | HIGH | Seed identity、Dataset epoch、batch indices、post-update acknowledgement與sampler checkpoint state皆為project decision；paper未指定。 |

## 3. Approved assumptions risk register

下表保留`assumptions.md`的paper-status；Risk只評估project assumption若錯誤時對重現的影響。

| Assumption ID(s) | Risk | 已批准內容 |
| --- | --- | --- |
| ASSUMP-RNG-001 | MEDIUM | Seed1及完整RNG state保存。 |
| ASSUMP-FW-001 | HIGH | PyTorch自建模型，跨framework semantics須測試。 |
| ASSUMP-GPU-001、ASSUMP-BATCH-001、ASSUMP-BN-001 | HIGH | Single GPU、per-device/global batch128、non-SyncBN。 |
| ASSUMP-SCALE-001、ASSUMP-MEAN-001 | HIGH | `[0,255]`與完整50k per-pixel mean。 |
| ASSUMP-AUGPAD-001、ASSUMP-AUGORDER-001 | MEDIUM | Constant-zero padding與幾何順序。 |
| ASSUMP-MEANORDER-001 | HIGH | Mean-first，padding 0為centered zero。 |
| ASSUMP-CONVPAD-001 | LOW | Conv padding1。 |
| ASSUMP-CONVBIAS-001、ASSUMP-FCBIAS-001 | MEDIUM | Conv no-bias、FC bias true/zero。 |
| ASSUMP-INITDIST-001、ASSUMP-FAN-001 | MEDIUM | Normal fan-in、gain sqrt(2)。 |
| ASSUMP-FCINIT-001 | HIGH | Normal std0.01，low-confidence。 |
| ASSUMP-BNEPS-001、ASSUMP-BNAFF-001 | MEDIUM | eps1e-5、gamma1/beta0。 |
| ASSUMP-BNMOM-001、ASSUMP-BNCOMP-001 | HIGH | Momentum0.001、scaled accumulators/running scale、de-biased eval及mandatory compatibility gate。 |
| ASSUMP-WD-BIAS-001、ASSUMP-WD-BN-001 | HIGH | All-learnable weight decay。 |
| ASSUMP-LOSS-001 | MEDIUM | Cross-entropy from logits、mean reduction。 |
| ASSUMP-SHUFFLE-001 | MEDIUM | Training每次資料遍歷不放回重排、base-seed-derived可恢復generator；test sequential official order。 |
| ASSUMP-WORKERS-001、ASSUMP-WSEED-001、ASSUMP-RNG-002 | HIGH | 4 workers、DataLoader-derived worker seeds、per-sample identity RNG與consumed-cursor resume。 |
| ASSUMP-DET-001、ASSUMP-CUDNN-001、ASSUMP-TF32-001 | MEDIUM | Deterministic algorithms、cuDNN benchmark false、TF32 false。 |
| ASSUMP-LOG-001、ASSUMP-EVAL-001、ASSUMP-CKPTINT-001 | MEDIUM | 100-update logging、1000-update eval、32k/48k/64k checkpoints。 |
| ASSUMP-FINAL-001、ASSUMP-LRSTEP-001 | HIGH | 64k final與exact LR boundaries。 |
| ASSUMP-SHORT-001、ASSUMP-SHORTPAD-001、ASSUMP-ODD-001 | HIGH | `::2`、symmetric padding、odd-size error。 |

## 4. Still lacking direct author CIFAR evidence

下列paper facts仍是`UNKNOWN`或只部分確認，即使project decision已批准也不得改寫證據狀態：

- 原作者實際fan-in或fan-out，以及CIFAR FC initialization。
- CIFAR conv/FC bias欄位與完整training prototxt。
- Option A的具體spatial indexing、channel-padding位置與odd-size行為。
- 原作者two-GPU per-device batch、local/synchronized BN及BN hyperparameters。
- 原作者CIFAR running-statistics實作仍未知；本專案scaled-accumulator與scale-factor PASS只證明預註冊Caffe b590 framework reference，不是paper fact。
- Weight decay對conv/FC bias、BN gamma/beta的原作者scope。
- 原始pixel range、mean統計集合/shape、mean與augmentation順序及test mean流程。
- 原作者CIFAR LR boundary的step-before/after語意。
- Table 6模型是否在schedule決定後以完整50k重新訓練。
- ResNet-20 8.75%的checkpoint/reporting form。
- Plain-20精確原論文final test error。
- Exact loss API/reduction、原作者shuffle/sampler、validation indices、evaluation interval及curve smoothing。

## 5. Remaining unapproved Phase 0 implementation item

- No remaining unapproved Phase 0 implementation decisions. This does not authorize Phase 1 or formal training.

## 6. Change-control rule

任何未來變更必須：

1. 建立或更新decision並保留paper evidence status。
2. 取得使用者明確批准；不得以test結果作理由。
3. 產生新的frozen config與SHA-256，保留舊config供稽核。
4. 更新implementation/test specs及approval summary。
5. 以新的Git commit記錄；不得覆寫既有批准紀錄。

本輪批准不代表聲稱原作者設定已被完整恢復。

## 7. Frozen-config validation snapshot

- Static structure checks：PASS（18/18 decision IDs一致；新增BN implementation與exact-resume欄位為required；無unknown implementation）。
- Previous SHA-256：`04BD17962E9D57AF5FE092AE810D896F4179AE357840F42E3105991C1D9C8EFC`（15 decisions；批准shuffle前版本）。
- Superseded intermediate SHA-256：`4830A13819A2B32F6876D102A9346BD555F143E6741ED44C1960A7A8E1BD363C`（16 decisions；final wording收斂前版本）。
- Immediately previous SHA-256：`9BF0A97F8B19D9B30EFF0E833F01FEDD609F95F2DD1F26991226090BCEA9C078`（16 decisions；Phase 0 closeout前版本）。
- Current SHA-256：`B6E9AA16D049FF5F5C089FB2FBAEDFC5608A220AC56E713FCF8DE2850653F84E`（18 decisions；加入`DEC-BN-001C`與`DEC-RNG-002`及其required fields）。
- Standard YAML/schema validation：既有專案`.venv`以PyYAML safe-load及strict schema loader驗證PASS；舊16-decision config、missing fields、unknown implementation與runtime override均明確拒絕。
