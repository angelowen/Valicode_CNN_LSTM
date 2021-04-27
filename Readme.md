# 使用CNN+LSTM預測驗證碼
* 結論: Fail
* 原因: 每個驗證碼的數字前後並無關係，因此使用LSTM時無法降低training loss，另外訓練集數量也過少，導致資訊量不足