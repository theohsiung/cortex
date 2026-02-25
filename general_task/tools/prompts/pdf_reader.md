# pdf_reader
逐頁擷取 PDF 文字內容。
## 參數
- file_path (str): 本地 PDF 檔案路徑（必須先用 download_file 下載到本地）
- page (int, 選填): 指定頁碼（1-based），不填則讀取全部
## 使用時機
- 需要讀取 PDF 文件內容時使用。
- 檔案必須已在本地端，如果是網路上的 PDF，先用 download_file 下載。
