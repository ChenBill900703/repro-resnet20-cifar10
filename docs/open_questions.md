# Open Questions

本文件只列需要人工確認或下一階段查閱額外主要來源的問題。2026-08-03 已完成第一輪「主要來源補充查證」；詳見 `docs/primary_source_review.md`。查證只使用核准的論文、作者官方 repository 與 BVLC Caffe 官方原始碼，未以第三方 repository 補足設定。

優先級定義：`BLOCKER` = 不決定就不能正確實作；`HIGH` = 可能顯著影響結果；`MEDIUM` = 影響可重現性或精確度；`LOW` = 主要影響紀錄與工程便利性。

| ID | 優先級 | 問題 | 為何論文不足 | 可查證來源 | 實作前是否必須回答 |
| --- | --- | --- | --- | --- | --- |
| Q-INIT-001 | BLOCKER | 初始化文獻 `[13]` 的 exact distribution、fan mode、gain，以及 convolution/FC/bias 各自規則為何？ | 本論文只寫「weight initialization in [13]」，未重列公式或逐層規則 | `[13]` 原始論文及其官方補充材料 | 是 |
| Q-SHORT-001 | BLOCKER | Option A stride-2 shortcut 的具體空間索引、channel zero-padding 分配與操作順序為何？ | Sections 3.3/4.2 只明寫 stride 2、identity、extra zero entries；沒有 tensor-level 定義 | 原作者官方 Caffe prototxt/source（若可取得）、作者補充材料 | 是 |
| Q-LR-001 | BLOCKER | 「at 32k and 48k iterations」代表第 32,000/48,000 個 update 使用新 LR，還是完成該 update 後才切換？ | 論文沒有 step-before/step-after 或 global-step 初值語意 | 原作者官方 training solver/config/logs | 是 |
| Q-CONV-001 | BLOCKER | 所有 `3x3` convolution 的 padding 與 bias 設定為何？ | 輸出尺寸約束暗示保持尺寸，但 CIFAR 段落未明列 padding/bias；Figure 2 說 bias 只是在 notation 中省略，不能證明實作無 bias | 原作者官方 Caffe prototxt；框架版本文件 | 是 |
| Q-AUG-001 | HIGH | Augmentation 文獻 `[24]` 是否需進一步閱讀；其 padding mode 或其他細節是否為本論文「follow」的一部分？ | 本論文列出 4-pixel padding、crop、flip，但未寫 padding mode；僅引用 `[24]` | `[24]` 原始論文與官方補充材料 | 是，至少需決定 padding mode |
| Q-CAFFE-001 | HIGH | 原作者 Caffe 實作是否可取得，且可否視為補充主要來源？應採何種證據優先順序？ | 論文提到 Caffe，但本 PDF 沒附 CIFAR prototxt/solver；專案禁止第三方 GitHub 實作 | 作者／Microsoft Research 官方發布、論文官方 supplementary | 是；需先由人工核准來源資格 |
| Q-BN-001 | HIGH | 論文 two-GPU 訓練時 BN 是 per-device local statistics、跨 GPU 同步，還是其他聚合方式？ | 只寫 BN `[16]` 與 two GPUs，未寫同步行為 | 原作者官方 Caffe 實作與當時 Caffe BN layer 行為；`[16]` | 是，或明確接受單 GPU 偏離 |
| Q-BNHP-001 | HIGH | BN epsilon、running-statistics momentum、affine 與 evaluation statistics 行為為何？ | 本論文未列這些 hyperparameters | 原作者官方 Caffe prototxt/source、當時 Caffe BN/Scale layer 文件、`[16]` | 是 |
| Q-WD-001 | HIGH | Weight decay `0.0001` 是否套用 convolution bias、FC bias、BN gamma/beta？ | 只提供整體 weight decay 數值，未列 parameter groups | 原作者官方 solver/prototxt；當時 Caffe parameter decay multipliers | 是 |
| Q-PAD-001 | HIGH | 4-pixel augmentation padding 的 mode 是 zero、reflect 或其他？ | Section 4.2 只寫 padded，未定義 mode | `[24]`、原作者官方 data pipeline | 是 |
| Q-MEAN-001 | HIGH | Per-pixel mean 是由完整 50k training set、45k subset，或其他集合／流程計算？ | Section 4.2 只寫「per-pixel mean subtracted」；沒有統計集合或 mean shape | 原作者官方 preprocessing/config；CIFAR experiment supplementary | 是 |
| Q-FULLTRAIN-001 | HIGH | 45k/5k 決定 schedule 後，Table 6 模型是否重新用完整 50k training set 訓練？ | 論文同時說 experiments trained on the training set，及 schedule determined on 45k/5k，未交代最終 retraining procedure | 原作者 logs、solver configs、supplementary | 是，否則無法精確對齊資料協定 |
| Q-EVAL-001 | HIGH | ResNet-20 的 8.75% 是第 64k final checkpoint、訓練過程 best checkpoint，還是其他選擇？ | Table 6 只給單值；只有 ResNet-110 caption 明確說 `best (mean+/-std)` | 原作者 logs、官方 experiment notes | 否；本專案已預先固定用 64k final，但須列為偏離／未知 |
| Q-FCINIT-001 | HIGH | FC weight 與 bias 是否沿用 `[13]`、使用 Caffe default，或另有初始化？ | CIFAR 段落只籠統說 weight initialization in `[13]` | `[13]`、原作者 Caffe prototxt | 是 |
| Q-LOSS-001 | MEDIUM | Softmax 對應的 loss layer、reduction 與 label handling 精確為何？ | 論文只寫 network ends with softmax | 原作者 Caffe prototxt、當時 Caffe SoftmaxWithLoss 文件 | 是 |
| Q-PLAIN-001 | MEDIUM | Plain-20 精確最終 test error 是否存在於作者其他正式材料？ | Table 6 不列 plain models；Figure 6 解析度不足，只能看曲線 | 作者官方 supplementary、正式 experiment logs；不得使用第三方估值 | 否；本專案可只比較相對行為 |
| Q-REPORT-001 | MEDIUM | 除 ResNet-110 外，各 Table 6 數字來自單 run、best run、last run 或其他 aggregation？ | Caption 只特別說明 ResNet-110 的 5-run form | 作者官方 logs／supplementary | 否，但報告須保留 UNKNOWN |
| Q-EVALINT-001 | MEDIUM | Figure 6 的 training/test error 評估頻率、train error 計算集合與平滑方式為何？ | 圖與 caption 未說 sampling、aggregation 或 smoothing | 原作者 logs／plotting scripts | 否；專案可預先固定自身規則 |
| Q-PLAINBN-001 | MEDIUM | CIFAR plain/residual 是否確實在每一個 convolution 後都放 BN，包括 residual branch addition 前一層？ | Section 3.4 對整體 implementation 說 every convolution；Section 4.2 只說 adopt BN；Figure 7 可間接支持 | 原作者 official prototxt；`[16]` | 是，現階段可先列 DERIVED |
| Q-VAL-001 | LOW | 45k/5k validation subset 的固定 indices 與 seed 為何？ | 論文只給 split 大小 | 原作者 data split/list files | 否；本專案不重做 schedule search |

## 主要來源補充查證結果（2026-08-03）

下表分開記錄paper fact與project decision，不會把Caffe b590語意或已批准assumption冒充為原作者CIFAR設定。`RESOLVED`只表示主要來源問題已有足夠答案；project approval以`decision_log.md`與Frozen YAML為準。

| Question ID | 查證結論 | 問題狀態 | 更新摘要 | 後續動作 |
| --- | --- | --- | --- | --- |
| Q-INIT-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ PROJECT DECIDED | [13]確認zero-mean Gaussian與`sqrt(2/n)`；作者實際fan mode及FC設定未確認。 | `DEC-INIT-001/002`與`DEC-FC-001`已批准；FC std`0.01`保留LOW_CONFIDENCE_ASSUMPTION。 |
| Q-SHORT-001 | STILL_UNKNOWN | OPEN（paper fact）/ PROJECT DECIDED | 只確認stride 2、extra zero entries、無參數；官方來源無`::2`/padding位置。 | `DEC-SHORT-001`已批准`::2`、symmetric padding及odd-size error；須preflight value tests。 |
| Q-LR-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ PROJECT DECIDED | Caffe b590 framework semantics確認#32,001與#48,001首次使用新LR；原作者CIFAR是否相同仍未知。 | `DEC-LR-001`已批准相同boundary作project assumption；須boundary tests。 |
| Q-CONV-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ PROJECT DECIDED | padding 1有shape推導及作者ImageNet deploy旁證；CIFAR conv/FC bias仍無直接證據。 | `DEC-CONV-001`、`DEC-INIT-003`、`DEC-FCBIAS-001`已批准；須架構與parameter tests。 |
| Q-AUG-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ PROJECT DECIDED | 主要來源支持操作，但mean相對位置仍未由作者確認。 | `DEC-AUG-001`與mean-first的`DEC-MEANORDER-001`已批准為project assumptions。 |
| Q-CAFFE-001 | CONFIRMED_PRIMARY_SOURCE | RESOLVED | 作者官方repo可用且指向Caffe b590；但只有ImageNet deploy，README明說不是用該版Caffe訓練。 | `DEC-CAFFE-001`已批准：Caffe只作framework semantics與assumption依據。 |
| Q-BN-001 | STILL_UNKNOWN | OPEN（paper fact）/ PROJECT DECIDED | Caffe b590 P2PSync為local forward BN stats；無法證明作者自有two-GPU CIFAR實作相同。 | `DEC-BN-001A`已批准single GPU/non-SyncBN並要求揭露偏離。 |
| Q-BNHP-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ APPROVED WITH GATE | Caffe b590確認eps`1e-5`、moving fraction`.999`、TEST global stats；作者CIFAR值及Caffe/PyTorch等價性未知。 | `DEC-BN-001A/B`已批准；`momentum=0.001`在mandatory compatibility test通過前禁止正式訓練。 |
| Q-WD-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ PROJECT DECIDED | Caffe確認global decay乘`decay_mult`且預設1；作者CIFAR param overrides不可得。 | `DEC-WD-001`已批准all-learnable scope為caffe-default-derived assumption；非paper-faithful fact。 |
| Q-PAD-001 | CONFIRMED_PRIMARY_SOURCE | RESOLVED | [24]明確寫每側zero padding 4 pixels。 | constant-zero padding已由`DEC-AUG-001`批准。 |
| Q-MEAN-001 | PARTIALLY_CONFIRMED | OPEN（paper fact）/ PROJECT DECIDED | Caffe官方CIFAR example支持full-shape training mean；作者ResNet流程與順序未確認。 | `DEC-PRE-001`與`DEC-MEANORDER-001`已批准完整50k mean及mean-first順序為assumptions。 |
| Q-FULLTRAIN-001 | STILL_UNKNOWN | OPEN / HIGH | 本輪允許來源仍無 final 50k retraining procedure。 | 等待作者正式log/config/supplementary；不得自行聲稱已知。 |
| Q-EVAL-001 | STILL_UNKNOWN | OPEN / HIGH | 本輪允許來源仍無8.75% checkpoint selection form。 | 保持專案64k final規則，並明列與paper reporting form不同。 |
| Q-FCINIT-001 | STILL_UNKNOWN | OPEN（paper fact）/ PROJECT DECIDED | [13]的FC數值是其ImageNet特定例外；Caffe MSRA filler也註記不適用InnerProduct shape假設。 | `DEC-FC-001`已批准normal std`0.01`為LOW_CONFIDENCE_ASSUMPTION；不得因test結果更換。 |
| Q-PLAINBN-001 | CONFIRMED_PRIMARY_SOURCE | RESOLVED | ResNet Section 3.4明說每個conv後BN，CIFAR Section 4.2明說採BN。 | 後續架構測試固定每conv後BN。 |

## 與 `AGENTS.md` 的證據狀態對齊

`AGENTS.md` 已於本輪修正，現在固定記錄：

- Per-pixel mean subtraction是**EXPLICIT**；由完整training set計算mean是**ASSUMPTION**，對應`Q-MEAN-001`。
- Option A的stride 2、zero entries與無參數是**EXPLICIT**；`::2`及symmetric zero padding是**ASSUMPTION**，對應`Q-SHORT-001`。
- 第64,000 update final checkpoint是**ASSUMPTION／專案規則**，不是論文reporting form，對應`Q-EVAL-001`。
