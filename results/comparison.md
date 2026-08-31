# 正式結果比較

## 結論

在相同資料、主要超參數與 64,000 updates 的設定下，ResNet-20 的 test error 為 **8.51%**，Plain-20 為 **9.57%**。本次實驗中，加入 residual shortcut 後：

- 多辨識正確 106 張 CIFAR-10 測試影像。
- Accuracy 從 90.43% 增加至 91.49%。
- Test error 下降 1.06 個百分點，相對下降約 11.08%。

| 模型 | Correct | Accuracy | Test error | Mean loss |
| --- | ---: | ---: | ---: | ---: |
| Plain-20 | 9,043／10,000 | 90.43% | 9.57% | 0.391285 |
| ResNet-20 | 9,149／10,000 | 91.49% | 8.51% | 0.361592 |

## 與論文比較

主要來源可確認 ResNet-20 的表列 test error 為 8.75%。本專案得到 8.51%，低 0.24 個百分點。

Plain-20 的精確原論文 final test error 在本專案採用的證據規則下標記為 UNKNOWN，因此不捏造或反推數值。

## 解讀限制

這些數值只代表固定 seed 1 的一次正式訓練。PyTorch 與原論文框架的 Batch Normalization、初始化、資料增強及其他未完整公開細節可能不同，因此不能只依數字宣稱全面優於原論文。

兩個 JSON 原始摘要保存在 `results/raw/`；訓練圖表保存在 `results/figures/`。
