# 結果來源與可追蹤性

## 正式成果

- Plain-20：`plain20_formal_64000_20260804_125449`
- ResNet-20：`resnet20_formal_64000_20260804_141544`
- 正式結果：第 64,000 update 的 final checkpoint
- Seed：1
- GPU：NVIDIA GeForce RTX 3070 Ti
- Precision：FP32
- Test-set selection：禁止，最終 checkpoint 預先固定

## 需要誠實揭露的來源限制

原始 formal run manifest 記錄的基準 commit 是 `e32662f0e528ac00b0304e96d1dea6e12de25da2`，同時記錄 `source_dirty: true`。這表示訓練執行當下存在尚未提交的 Phase 4 變更。

本 Repository 是後續整理的公開快照，保存程式、固定設定、測試、最終數值、圖表與報告，但不宣稱僅憑某個公開 commit 就能逐位元重建既有權重。若研究需要最高等級的 commit-to-checkpoint 可追蹤性，應由乾淨且已標記的 release commit 重新執行正式訓練。

## 已保存的驗證資訊

- Frozen config SHA-256：`B6E9AA16D049FF5F5C089FB2FBAEDFC5608A220AC56E713FCF8DE2850653F84E`
- Mean artifact SHA-256：`6DAFA62D5751FB9EAA9537BF61D9485DF692926CEC1B16B4BFF53927C40AA0F1`
- 正式 preflight 曾記錄 150 tests passed
- ResNet-20 final predictions SHA-256：`E24C33C1426F4207B3844C7DBF74F4E2455B02E7FF34862F731ACA40B4B667F8`
