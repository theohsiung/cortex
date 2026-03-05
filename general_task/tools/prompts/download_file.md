# download_file
從 URL 下載檔案（CSV、Excel、PDF、JSON、ZIP 等）到本地端，回傳儲存路徑供後續工具處理。
## 參數
- url (str): 檔案的直接下載 URL
- save_path (str, 選填): 儲存路徑（不填會自動從 URL 推斷檔名）
## 使用時機
- 當你有一個指向資料檔案的 URL 時（CSV、Excel、PDF、JSON、ZIP），用此工具下載。
- 從 web_search 找到下載連結後，下一步就是 download_file。
- 從資料庫或 API 端點取得資料檔（如 USGS、政府開放資料的 CSV export）。
## 不要搞混
- 要下載檔案 → download_file（本工具）
- 要讀取網頁 HTML 內容 → web_browser
- 要搜尋資訊 → web_search
## 下載後的下一步
- CSV/Excel → excel_reader 或 python_executor (pd.read_csv)
- PDF → pdf_reader
- JSON/XML/TXT → file_reader
- ZIP → zip_extractor
