# python_executor
安全執行 Python 程式碼。預先注入 pandas (pd)、numpy (np)、Biopython (Bio)、matplotlib (plt)。
## 參數
- code (str): Python 程式碼字串
## 使用時機
- 需要數值計算、資料分析、資料轉換等複雜運算時使用。
- 處理 download_file 下載的 CSV 檔案：pd.read_csv("/path/to/file.csv")
- 資料篩選、彙總、格式轉換等操作。
## 輸出方式
- print() 的輸出會被回傳。
- 命名為 result、answer、output、value 的變數會自動回傳。
- matplotlib 圖表會自動儲存為 PNG。
## 注意
- 檔案路徑使用 download_file 或其他工具回傳的實際路徑。
- 不要假設檔案存在，先用 download_file 下載。
