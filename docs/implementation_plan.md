# Implementation Plan

本文件記錄已完成的Phase 0基礎與後續實作順序。Phase 0只建立config、reproducibility、sampling與BN compatibility基礎及其synthetic tests；不建立Plain/ResNet模型、CIFAR dataset/transform或training engine。所有後續實作必須以`configs/cifar10_plain20_resnet20_frozen.yaml`為唯一設定來源，並遵守`AGENTS.md`與`docs/decision_log.md`。

## 1. 實作順序

### Phase 0：設定與可重現性基礎

- Config loader：只接受標準YAML資料，驗證`schema_version`、required fields、型別、範圍與交叉欄位不變量。
- Evidence-aware config validation：保留FC initialization、Option A、BN momentum、weight-decay scope、preprocessing order及final-checkpoint form的evidence status。
- Frozen protection：CLI不得靜默覆寫frozen config；任何override都必須產生新config、new SHA-256及新的decision approval。
- Seed/environment utility：固定base seed 1，設定Python、NumPy、torch CPU/CUDA及獨立sampler/worker generators，保存/恢復全部RNG state及environment fingerprint。
- Exact-resume sampling：依`DEC-RNG-002`，`StatefulBatchSampler`保存permutation、consumed cursor、epoch與generator state；issued/prefetched batch不推進checkpoint cursor，只有optimizer update成功後才`mark_batch_consumed`。
- Per-sample augmentation RNG contract：stochastic transform seed由base seed、epoch與official sample index派生；Dataset必須顯式持有epoch，training batch必須攜帶indices，結果不依賴worker assignment或prefetch順序。
- BN compatibility harness規格落地：使用`CaffeCompatibleBatchNorm2d`的scaled mean/unbiased-variance accumulators與running scale；eval採accumulator/scale，checkpoint保存全部buffers；小型tensor compatibility test是`DEC-BN-001B/C`的execution gate。

### Phase 1：架構元件

- Initialization helpers：conv zero-mean normal、fan-in、gain`sqrt(2)`；FC zero-mean normal std`0.01`；FC bias 0；BN gamma 1/beta 0。
- Option A shortcut：identity或`::2`偶數索引下採樣加symmetric channel zero-padding；不得有parameter；odd spatial size報錯。
- Post-activation block：`conv-BN-ReLU-conv-BN-add-ReLU`。
- Plain-20：與ResNet-20共享stem、stage、head與初始化邏輯，但沒有shortcut/addition。
- ResNet-20：`n=3`、blocks`[3,3,3]`、channels`[16,32,64]`、9個Option A shortcuts。
- Phase 1完成條件：架構、shape、weighted-layer count、parameter equality、shortcut及initialization tests通過。

### Phase 2：資料與前處理

- CIFAR dataset wrapper：只使用official 50k train/10k test split；本階段規劃不下載資料。
- Mean artifact generation：只從完整50k training set的`[0,255]`float images計算`3x32x32`mean，保存來源metadata與SHA-256。
- Training transform：float`[0,255]`→subtract mean→constant-zero pad 4→random crop 32→random horizontal flip。
- Test transform：float`[0,255]`→subtract同一mean；無random operation。
- Training DataLoader：`shuffle=true`且不使用replacement sampling；每完成一次完整training-set traversal後重新產生排列。
- Sampler generator：使用由base seed確定性派生的`torch.Generator`；checkpoint同時保存完整permutation、consumed cursor、epoch與generator state。只保存generator state不符合exact resume。
- Test DataLoader：`shuffle=false`，保持official dataset order；不得使用random sampler。
- DataLoader worker seeding：由獨立base-seed-derived generator產生PyTorch worker seed，再同步Python/NumPy/torch CPU；augmentation本身必須使用per-sample identity RNG，不得依賴worker-local stream；不得使用test statistics。
- Phase 2完成條件：mean provenance/hash、padding centered-zero、scale、shape、determinism及test non-random tests通過。

### Phase 3：最佳化、精確更新與評估

- SGD optimizer parameter groups：所有learnable parameters恰好出現一次，weight decay`0.0001`涵蓋conv/FC weights、FC bias與BN gamma/beta；momentum`0.9`、Nesterov false。
- Exact-update LR controller：以completed optimizer updates計數，精確實作1–32000、32001–48000、48001–64000三段LR。
- Checkpoint state：保存完整恢復狀態、config payload/hash、commit與environment fingerprint。
- Evaluation：single original`32x32`view，test error百分比；evaluation結果不得影響config、seed、checkpoint或停止條件。
- Phase 3完成條件：optimizer coverage、LR boundaries、checkpoint resume及evaluation tests通過。

### Phase 4：執行控制、紀錄與報告

- Training engine：只有全部preflight結果為PASS才允許formal run；精確執行64,000 updates。
- Logging：每update記LR/completed updates；每100 updates記train metrics；每1,000 updates依預先批准流程評估。
- Checkpoints：32k、48k、64k保存；只有update 64,000 final是主要結果。
- Plots：清楚區分train/test metrics，不使用test curve選擇模型。
- Comparison report：ResNet-20與Table 6的8.75%透明比較；Plain-20 paper exact error保持UNKNOWN；列出所有assumptions與偏離。

## 2. 預計檔案

Phase 0已建立：

- `src/resnet_repro/config.py`
- `src/resnet_repro/reproducibility.py`
- `src/resnet_repro/sampling.py`
- `src/resnet_repro/batch_norm.py`
- `src/resnet_repro/bn_compatibility.py`
- `tests/phase0/*.py`

以下仍只列後續規劃，Phase 0收尾不得建立：

- `src/resnet_repro/models/initialization.py`
- `src/resnet_repro/models/blocks.py`
- `src/resnet_repro/models/plain.py`
- `src/resnet_repro/models/resnet.py`
- `src/resnet_repro/data/cifar10.py`
- `src/resnet_repro/data/transforms.py`
- `src/resnet_repro/training/optimizer.py`
- `src/resnet_repro/training/schedule.py`
- `src/resnet_repro/training/checkpoint.py`
- `src/resnet_repro/training/engine.py`
- `src/resnet_repro/evaluation.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- 對應的`tests/*.py`

## 3. 依賴關係

```text
Frozen YAML + decisions
        |
        v
Phase 0 config/evidence/RNG ----------------------+
        |                                         |
        +--> BN compatibility harness (gate)      |
        |                                         |
        +--> Phase 1 models ----+                  |
        |                       |                  |
        +--> Phase 2 data ------+--> Phase 3 optimizer/schedule/checkpoint/eval
                                                   |
                                                   v
                                      Phase 4 engine/logging/report
```

- Phase 0必須最先完成；其他模組不得各自建立隱性defaults。
- Phase 1的initialization、Option A與block可在Phase 0之後平行；Plain/ResNet需等待共同元件。
- Phase 2的mean artifact、transforms與seeding可在Phase 0之後平行規劃，但整合測試需dataset wrapper。
- Phase 3的LR controller與checkpoint schema可在Phase 1/2之外先行實作；optimizer coverage需模型，evaluation需模型與test transform。
- Phase 4必須等待Phase 1–3及全部preflight。

## 4. 實作不變量

- Plain-20與ResNet-20的learnable parameter count完全相同。
- 所有Option A shortcuts無trainable parameters；禁止projection shortcut。
- 20 weighted layers恰為19 convolution加1 FC。
- Stage blocks為`[3,3,3]`，channels為`[16,32,64]`，spatial sizes為`[32,16,8]`。
- 每個convolution後有BN；block為post-activation。
- Conv無bias；FC有bias；BN affine enabled。
- 所有learnable parameters恰好屬於一個optimizer group並套用相同weight decay。
- Update #32,000仍用0.1；#32,001用0.01；#48,000用0.01；#48,001用0.001；#64,000用0.001後終止。
- Test result不得影響設定、seed、early stopping或checkpoint selection。
- Frozen config不得被CLI靜默覆寫；任何override必須產生新config SHA-256、decision及approval。
- BN必須使用approved scaled-accumulator implementation；standard PyTorch running-stat initialization/eval不可替代。Compatibility test未通過時不得正式訓練。
- Stochastic augmentation只能由`(base_seed, epoch, official_sample_index)`決定；worker assignment與prefetch不得改變結果。
- Sampler checkpoint cursor只代表optimizer已成功消耗的samples；issued/prefetched未消耗batch在resume後重發。
- Plain-20 exact paper test error永遠保持null/UNKNOWN，除非取得新的作者正式主要來源並重審。

## 5. Checkpoint schema

每個checkpoint至少包含：

| 欄位 | 內容與不變量 |
| --- | --- |
| `schema_version` | Checkpoint schema版本；不相容版本拒絕載入。 |
| `model_name` / `model_state` | `plain20`或`resnet20`與完整model state。 |
| `optimizer_state` | SGD momentum buffers及所有parameter-group state。 |
| `completed_updates` | 已完成optimizer updates；恢復後下一次執行`completed_updates+1`。 |
| `current_lr` | 與scheduler由`completed_updates`推導的LR完全一致。 |
| `scheduler_state` | Boundary/controller state；不得只依epoch重建。 |
| `bn_running_statistics` | 所有BN scaled running mean、scaled unbiased running variance、running scale與`num_batches_tracked`。 |
| `python_rng_state` | Python RNG。 |
| `numpy_rng_state` | NumPy RNG。 |
| `torch_cpu_rng_state` | torch CPU RNG。 |
| `torch_cuda_rng_states` | 所有使用中CUDA devices；主要run為一張GPU。 |
| `dataloader_generator_state` | Sampler/worker generator states與worker derivation metadata。 |
| `sampler_state` | Current permutation、consumed cursor、epoch及sampler generator state；不得以issued/prefetch cursor替代。 |
| `augmentation_rng_policy` | Base seed與per-sample seed schema version；Dataset epoch與official indices contract。 |
| `frozen_config` | 完整YAML資料快照，不只路徑。 |
| `config_sha256` | Frozen config bytes的SHA-256。 |
| `git_commit` | Source commit，基準為`f3e20eb`；另記dirty狀態。 |
| `environment_fingerprint` | Python/PyTorch/CUDA/cuDNN/GPU與有效package-lock fingerprint。 |
| `mean_artifact_sha256` | Training-derived mean artifact hash；train/test必須一致。 |

## 6. Failure recovery

1. 只從成功完成原子寫入且schema/hash驗證通過的checkpoint恢復。
2. 驗證model name、config SHA-256、mean hash、source commit/environment差異；任何不符均停止，不靜默繼續。
3. 載入model、optimizer、scheduler、BN全部buffers及全部RNG/DataLoader/sampler states；sampler issued cursor從consumed cursor重建。
4. 驗證checkpoint的`completed_updates=N`時，下一次optimizer update必須編號`N+1`；不得先額外step scheduler或重新消耗一個batch。
5. 在不更新parameter的情況下驗證目前LR等於frozen schedule對`N+1`的期望值。
6. 以專用resume test比較連續run與中斷/恢復run的第一個未消耗batch、後續indices、per-sample augmentation、LR、loss及state；同一批准環境的sample order與augmentation必須exact一致。
7. 恢復失敗不得改用較舊checkpoint而不揭露；需記錄故障、選定恢復點與重複/遺失的風險，重新取得執行批准。

## 7. 正式訓練啟動條件

只有同時滿足下列條件才可開始正式64k training：

- Frozen YAML parse/schema/hash驗證通過，且與checkpoint/command line一致。
- `docs/test_specification.md`所有blocking tests均有可檢查PASS紀錄。
- BN compatibility test已完成並確認`DEC-BN-001C`的scaled-accumulator/scale-factor語意；任何buffer、公式或容差變更都須重新批准與測試。
- `DATA-011`／`CKPT-003`已驗證四worker prefetch下的consumed-cursor resume與per-sample augmentation exact equality。
- Architecture、shape、layer/parameter counts、Plain/ResNet equality、shortcut、initialization、optimizer、LR、preprocessing、forward、backward及smoke tests全部通過。
- 本機environment verification與有效的`environment/requirements-lock.txt`可稽核；不得以Codex sandbox冒充。
- CIFAR-10 official split與training-only mean provenance已驗證。
- Source commit、dirty status、config SHA-256、seed與output paths已記錄。
- 沒有任何test-based configuration、seed或checkpoint selection。

任何一項缺失、失敗或狀態不一致都必須停止；不得用warning繞過。
