# Test Specification

本文件只定義測試，不建立或執行Python測試。除非另有明示，所有測試均阻擋正式訓練；PASS/FAIL紀錄必須包含source commit、config SHA-256、environment fingerprint及執行時間。任何容差必須在第一次看到測試結果前固定，且不得依test accuracy調整。

## 1. Config tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CONFIG-001 | 驗證YAML schema與required fields | Frozen YAML存在；schema規格已版本化 | `configs/cifar10_plain20_resnet20_frozen.yaml` | 用safe YAML parser載入；檢查型別、required keys、allowed enums、數值範圍及跨欄位一致性 | Parse成功；schema v1完整；無未知/缺漏required field；Plain error為null | 隱性default、錯型別或遺漏決策造成不可稽核run | 是 |
| CONFIG-002 | 防止CLI靜默override frozen config | Config loader與CLI介面完成 | Frozen YAML及任一不同override | 啟動dry-run config resolution；嘗試改seed/LR/model等欄位；檢查hash與exit behavior | 原config不可被原地覆寫；override被拒絕，或明確產生新config/hash且要求新approval | 實際run與批准設定不同且無紀錄 | 是 |
| CONFIG-003 | 驗證Phase 0 closeout required fields與decisions | `DEC-BN-001C`、`DEC-RNG-002`已批准 | Frozen YAML | 檢查BN implementation/accumulators/scale/checkpoint buffers、per-sample RNG、consumed-cursor resume欄位及18個decision IDs | 全部欄位精確存在；兩個新decision不可移除；allowed values與cross-field invariants一致 | 執行語意未被frozen config約束 | 是 |
| CONFIG-004 | 拒絕舊config與unknown implementation | Strict schema loader完成 | 移除新欄位/decisions的16-decision舊payload；未知BN implementation | 分別safe-load與schema validation | 舊payload、missing decision、unknown implementation全部明確FAIL，不得fallback | 舊config或未批准implementation被靜默接受 | 是 |

## 2. Architecture tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARCH-001 | 驗證20 weighted layers | Plain/ResNet constructors完成 | 兩個model instances | 依conv/linear module與weighted-layer定義計數 | 兩模型均為20 | Depth不符paper target | 是 |
| ARCH-002 | 驗證19 Conv2d+1 Linear | 同上 | Module trees | 精確計數Conv2d與Linear；排除BN/ReLU/pooling | 各19 Conv2d、1 Linear | Layer type/count錯誤 | 是 |
| ARCH-003 | 驗證stage blocks`[3,3,3]` | Model metadata可檢查 | 兩模型 | 逐stage列出two-conv groups/blocks | 每stage恰3，總計9 | Stage depth不符`n=3` | 是 |
| ARCH-004 | 驗證forward shapes 32/16/8 | Forward hooks可用；不執行正式資料 | 固定synthetic tensor`[2,3,32,32]` | 以CPU forward抓stem及每stage output | Spatial依序32、32、16、8；channels 16、16、32、64 | Padding/stride/stage transition錯誤 | 是 |
| ARCH-005 | 驗證Plain/ResNet parameter count相同 | 兩模型設定完全相同 | Named parameters | 分別加總learnable parameter numel並比較 | Exact equality；shortcut增加0 parameters | 對照不再只差residual addition | 是 |
| ARCH-006 | 比較ResNet-20 parameter count與paper 0.27M | ARCH-005通過；bias/BN scope已凍結 | ResNet named parameters與Table 6 target | 計算exact count；報exact及換算M；列各類參數明細 | Round-to-2-decimal M應與0.27M合理一致；不得以rounding掩蓋結構性差異 | 遺漏/多出layer、bias或BN parameters | 是 |

## 3. Option A shortcut tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SHORT-001 | 確認shortcut無trainable parameters | Option A完成 | 所有9個shortcuts | 檢查parameters/buffers/state及model count差 | 0 trainable parameters；無projection conv/linear | 違反option A與Plain/ResNet等參數 | 是 |
| SHORT-002 | Identity shortcut逐值相等 | Identity case可直接呼叫 | 唯一值synthetic tensor | 呼叫identity path並做exact elementwise比較 | Output與input shape/value/dtype/device完全相同 | Identity path意外轉換或複製錯誤 | 是 |
| SHORT-003 | 驗證16→32 tensor semantics | Downsample shortcut完成 | 唯一編碼tensor`[2,16,32,32]` | 計算expected=`input[:,:,::2,::2]`；比較32-channel output | Spatial為16；前8/後8 channels全0；中間16 exact等於expected | Index或padding位置錯誤 | 是 |
| SHORT-004 | 驗證32→64 tensor semantics | 同上 | 唯一編碼tensor`[2,32,16,16]` | 計算expected=`input[:,:,::2,::2]`；比較64-channel output | Spatial為8；前16/後16全0；中間32 exact等於expected | Stage 3 shortcut錯誤 | 是 |
| SHORT-005 | Odd-size明確報錯 | Odd-size guard完成 | `[2,16,31,31]`等odd tensors | 呼叫downsample shortcut並擷取exception | 在計算前以明確、可辨識錯誤拒絕；不得silent floor/ceil | 未批准的shape semantics被靜默採用 | 是 |

## 4. Initialization tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INIT-001 | 驗證conv為normal非uniform | Initializer完成；固定seed與預先註冊distribution test方法 | 多個代表性conv tensors | 初始化大樣本；檢查config path與distribution diagnostics | 使用normal RNG path；分布diagnostic不支持uniform實作 | 初始化recipe錯誤 | 是 |
| INIT-002 | 驗證conv std接近`sqrt(2/fan_in)` | 在看結果前固定sample size與abs/rel tolerance | 各kernel/channel shape conv weights | 對每層計算target與empirical std；輸出差值 | 每層在預先固定容差內；不得事後放寬 | Fan mode/gain或shape計算錯誤 | 是 |
| INIT-003 | 驗證所有conv沒有bias | Models完成 | 19 conv modules | 檢查bias attribute及named parameters | 全部conv bias不存在 | Parameter count、WD與BN前運算偏離 | 是 |
| INIT-004 | 驗證FC weight/bias | FC initializer完成 | 兩模型classifier | 固定seed初始化；量測weight mean/std並檢查bias | Weight normal、mean接近0、std 0.01在預註冊容差內；bias全0 | LOW_CONFIDENCE_ASSUMPTION未被忠實實作 | 是 |

## 5. Batch Normalization tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BN-001 | 驗證每個conv後存在BN | Model graph/hook inspection可用 | Plain/ResNet module graph | 逐conv檢查下一運算與總數 | 每個conv後、activation前有BN；BN count=conv count | 違反paper BN placement | 是 |
| BN-002 | 驗證BN靜態設定與初始化 | BN modules完成 | 所有BN layers | 檢查affine、gamma、beta、eps、sync type及recalibration flag | affine true、gamma1、beta0、eps1e-5、非SyncBN、eval用running stats、無recalibration | BN config與批准決策不符 | 是 |
| BN-003 | 驗證Caffe-compatible running-stat semantics | `DEC-BN-001B/C`已批准；固定4 batches、float64、shape與abs/rel tolerance | 4個固定小型tensors，包含不同mean/variance | 每batch記biased/unbiased variance、Caffe accumulator/scale與implementation effective stats；比較每步mean/variance/scale及final eval output | Compatibility PASS；全部預註冊比較在容差內；否則FAIL並回到decision review | Running variance/scale-factor不等價導致評估系統性偏差 | 是；mandatory execution gate |
| BN-004 | Training output reference comparison | `CaffeCompatibleBatchNorm2d`完成 | 固定float64 NCHW tensor | 以batch biased variance手算training normalization並比較layer output | 在預註冊容差內一致 | Training forward公式偏離BN定義 | 是 |
| BN-005 | Affine initialization | Affine enabled | 新BN layer | 檢查gamma/beta | Gamma全1、beta全0 | 初始化與批准設定不符 | 是 |
| BN-006 | Effective running statistics | 至少完成一個training batch | 固定tensor | 比較`running_mean/running_scale`及`running_var/running_scale`與Caffe reference | Mean與unbiased variance逐值在容差內 | Scaled accumulator或de-bias公式錯誤 | 是 |
| BN-007 | Eval不得修改buffers | BN已有running state | Eval probe | Eval前後比較全部named buffers | Exact不變 | Evaluation污染checkpoint state | 是 |
| BN-008 | BN state_dict roundtrip | BN已有running state | 完整state_dict與eval probe | Save/load新instance後比較全部buffers與eval output | Buffers與output exact一致 | running scale/counter未checkpoint | 是 |
| BN-009 | First-batch前eval明確失敗 | 新BN layer、無training forward | Eval probe | 呼叫eval forward並擷取exception | 明確拒絕undefined effective statistics | Silent使用零scale或隱性default | 是 |
| BN-010 | Dtype/device behavior | CPU float32/float64；CUDA若可用 | 對應dtype/device tensors | Forward並檢查output、parameters及buffers | Dtype/device一致；無隱性CPU copy或cast | 數值路徑或裝置錯誤 | 是 |
| BN-011 | `num_batches_tracked`語意 | BN layer完成 | 兩次train forward加一次eval | 每步檢查counter | 只在成功training forward後遞增；eval不變 | Checkpoint counter與實際updates不一致 | 是 |

## 6. Data and preprocessing tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DATA-001 | 確認mean只由training set計算 | Dataset provenance與mean generator完成 | Official train/test metadata | 追蹤sample IDs與artifact metadata；斷言無test ID | Mean source恰為50k official training images | Test leakage | 是 |
| DATA-002 | 驗證mean shape | Mean artifact存在 | Mean artifact | 載入並檢查shape/dtype/finite | Shape`[3,32,32]`，float且finite | Per-pixel語意錯誤 | 是 |
| DATA-003 | 驗證train/test共用mean hash | Transform configs完成 | Train/test transform與artifact | 解析兩pipeline引用並計算artifact SHA-256 | 路徑/內容hash完全相同 | Train/test preprocessing不一致或test statistics leakage | 是 |
| DATA-004 | 驗證0..255 scale且不除255 | Dataset wrapper完成 | 已知uint8像素樣本 | 轉float後、減mean前擷取tensor | 數值等於原uint8轉float；沒有`/255` | 輸入尺度、初始化與最佳化偏離 | 是 |
| DATA-005 | 驗證mean subtraction後padding為0 | Mean-first transform完成 | 人工構造image=mean artifact | 執行subtract再constant pad，於crop前檢查 | Centered image全0且四周padding全0 | Padding發生順序/mode錯誤 | 是 |
| DATA-006 | 驗證training transform輸出 | Transform完成；固定RNG | 多個synthetic`[3,32,32]`images | 執行完整train pipeline並記crop/flip決策 | Output shape`[3,32,32]`、float、finite；操作順序與config一致 | Geometry或dtype錯誤 | 是 |
| DATA-007 | 驗證test無random operation | Test transform完成 | 同一test image重複多次及不同RNG states | 多次執行並比較output/RNG consumption | Outputs exact相同；沒有crop/flip；不消耗augmentation RNG | Test protocol不穩定或TTA | 是 |
| DATA-008 | 驗證training sampler使用shuffle且不replacement | Training DataLoader完成 | 可追蹤唯一sample ID的完整training dataset | 檢查sampling設定並完成一次完整training-set traversal，統計sample IDs | `shuffle=true`；每個training ID恰出現一次，無重複、無遺漏、無replacement sampling | Sampling policy與批准決策不符 | 是 |
| DATA-009 | 驗證相同base seed的第一輪sample順序 | Training DataLoader與generator derivation完成 | 相同dataset與相同base seed的兩個獨立loader | 各自取得第一個完整training-set traversal的sample ID序列並比較 | 兩個第一輪sample序列exact相同 | Generator未由base seed確定性派生 | 是 |
| DATA-010 | 驗證不同預先註冊seed產生不同排列 | Training DataLoader完成；至少兩個seed在測試前預先註冊 | 相同dataset與不同預先註冊base seeds | 各自取得第一輪sample ID序列並比較 | 每個序列均無replacement且不同seed的排列不相同；不得依test accuracy選seed | Seed未影響排列或測試後挑seed | 是 |
| DATA-011 | 驗證checkpoint resume後第一個未消耗sample/batch | `StatefulBatchSampler`與generator recovery完成 | 固定base seed、4 workers及prefetch | 保存permutation、consumed cursor、epoch與generator state；resume並比較不中斷reference | Prefetched但未consumed batches重發；第一個及後續indices exact一致 | Issued cursor或只存generator造成跳批／重複 | 是 |
| DATA-012 | 驗證test sampler不shuffle且保持official order | Test DataLoader完成 | Official test dataset indices | 檢查sampler與shuffle設定；完整遍歷並記錄indices；在不同RNG state下重複 | `shuffle=false`、無random sampler；indices每次均為official dataset order且exact一致 | Test sampling隨機化或順序改變 | 是 |
| DATA-013 | Sampler epoch/final-batch/drop-last semantics | Stateful sampler完成 | Dataset sizes可整除與不可整除batch size | 完整遍歷兩epoch，分別測`drop_last=false/true` | 每epoch重新排列；final partial只在false保留；true只捨棄尾端不足batch | Epoch transition或batch completeness錯誤 | 是 |
| DATA-014 | Sampler state validation | Sampler state schema完成 | Bad schema、missing field、loader mismatch、bad cursor、非bijection permutation | 逐一load | 全部明確拒絕 | Corrupt/incompatible checkpoint被接受 | 是 |
| DATA-015 | 相同sample identity產生相同augmentation | Per-sample RNG完成 | 相同base seed/epoch/official index，不同外部RNG狀態 | 產生Python/NumPy/torch random draws | 全部exact一致 | Augmentation仍依賴全域stream | 是 |
| DATA-016 | Epoch/index改變augmentation stream | 同上 | 只改epoch或official index | 比較derived seed與random draws | Stream不同 | 每epoch或每sample augmentation重複 | 是 |
| DATA-017 | Worker assignment/prefetch independence | Worker seeder與per-sample RNG完成 | 同identity分派到不同worker seed/順序 | 在不同worker-local state產生augmentation | Exact一致 | Worker排程改變資料內容 | 是 |
| DATA-018 | Per-sample context恢復外層RNG | RNG context完成 | 已知Python/NumPy/torch CPU state | 進入context消耗randomness後離開，再與未進入reference比較 | 三套外層state與後續draw exact一致 | Dataset transform污染worker其他RNG流程 | 是 |

## 7. Optimizer tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-001 | 驗證SGD基本設定 | Optimizer factory完成 | Frozen config與model | 建立optimizer後檢查class/defaults/groups | SGD、momentum0.9、weight decay1e-4、Nesterov false | Training recipe錯誤 | 是 |
| OPT-002 | 驗證parameter group完整且唯一 | Model與optimizer完成 | Named learnable params及groups | 以object identity比對集合與重複次數 | 每個learnable parameter恰出現一次；無遺漏/重複 | 未更新、重複更新或不同regularization | 是 |
| OPT-003 | 驗證BN affine與FC bias亦decay | OPT-002通過 | BN gamma/beta、FC bias、groups | 查每個目標parameter所屬group的effective decay | 全部effective weight decay=1e-4；conv bias不存在 | 未實作批准的all-learnable scope | 是 |

## 8. Learning-rate tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR-001 | 驗證首update | LR controller完成 | completed_updates=0 | 查下一次update的LR | Update #1使用0.1 | 初始LR錯誤 | 是 |
| LR-002 | 驗證32k舊LR末端 | 同上 | completed_updates=31999 | 執行/查update #32,000 | LR=0.1 | Milestone早切 | 是 |
| LR-003 | 驗證32k後切換 | 同上 | completed_updates=32000 | 查update #32,001 | LR=0.01 | Milestone晚切 | 是 |
| LR-004 | 驗證48k第二段末端 | 同上 | completed_updates=47999 | 查update #48,000 | LR=0.01 | 第二boundary早切 | 是 |
| LR-005 | 驗證48k後切換 | 同上 | completed_updates=48000 | 查update #48,001 | LR=0.001 | 第二boundary晚切 | 是 |
| LR-006 | 驗證64k最後update與終止 | Engine dry-run/controller完成 | completed_updates=63999 | 執行/模擬update #64,000後再請求下一update | #64,000用0.001；完成後completed_updates=64000並拒絕#64,001 | Off-by-one或超訓練 | 是 |

## 9. Checkpoint and recovery tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CKPT-001 | Resume不重複/跳過update | Checkpoint/engine完成；固定synthetic batches | 連續N updates與K處中斷版本 | 比較update IDs、batch IDs、model/optimizer state progression | Resume第一個update=K+1；序列無重複或缺口 | 訓練長度與資料序列失真 | 是 |
| CKPT-002 | Resume前後LR一致 | Scheduler state可保存 | 31,999/32,000/47,999/48,000等boundary checkpoints | Save/load後查下一update LR並與連續run比較 | 每個boundary exact一致 | Resume改變schedule | 是 |
| CKPT-003 | Resume前後RNG、sampler與augmentation可精確恢復 | 全部RNG與sampler state schema完成 | Python/NumPy/torch CPU/CUDA、worker/sampler generators、permutation/cursor/epoch | 生成序列並在有worker prefetch時checkpoint；不中斷與reload後比較第一未消耗batch、後續indices及augmentation | 同一批准環境下global RNG序列、sample order與per-sample augmentation exact一致 | Resume後資料、初始化或augmentation路徑改變 | 是 |

## 10. Evaluation tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EVAL-001 | 只使用single original view | Evaluation pipeline完成 | Official test samples | 檢查transform graph及每sample view count | 每sample恰一個原始32x32 view，只做float與mean subtraction | TTA或crop造成不可比 | 是 |
| EVAL-002 | 驗證test error計算 | Metric完成 | 人工logits/labels含已知錯誤數 | 計算predictions、錯誤數與百分比 | `100*incorrect/total`與reported percent一致 | 指標/分母/accuracy-error混淆 | 是 |
| EVAL-003 | 防止test-based checkpoint selection | Reporting/selection policy完成 | 多個checkpoint及刻意不同test metrics | 嘗試觸發best selection/early stop/config update | 系統拒絕或忽略test-based selection；primary固定64k final | Test leakage與結果挑選 | 是 |

## 11. Smoke tests

| Test ID | 目的 | 前置條件 | 輸入 | 步驟 | 預期結果 | 失敗代表的風險 | 阻擋正式訓練 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SMOKE-001 | 驗證小規模forward/backward/update | Phase 0–3 blocking unit tests先通過；只用synthetic或approved training subset | Plain/ResNet各一小batch | 執行forward、loss、backward及一個optimizer update；不執行正式訓練 | 兩模型均成功；completed_updates恰加1；state可save | 整合路徑不可用 | 是 |
| SMOKE-002 | 驗證loss/gradient finite | SMOKE-001環境 | Smoke batch的loss/gradients | 在forward/backward後檢查全部loss、activation摘要及gradients | 全部finite；無NaN/Inf；異常立即FAIL | 數值不穩定或實作錯誤 | 是 |

> 注意：Smoke table的每個test仍遵循欄位語意；不使用CIFAR-10 test set，也不得把smoke loss當作調參依據。
