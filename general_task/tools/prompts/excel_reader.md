# excel_reader
使用 pandas 讀取 Excel (xlsx/xls) 或 CSV 檔案，回傳欄位資訊與前 5 筆資料預覽。
## 參數
- file_path (str): 本地檔案路徑（必須先用 download_file 下載到本地）
- sheet (str, 選填): 工作表名稱
## 使用時機
- 快速預覽 CSV/Excel 檔案的結構和前幾筆資料。
- 如需更複雜的篩選和分析，改用 python_executor 搭配 pandas。
