# docx_writer
建立 Word (.docx) 文件。支援標題、段落、項目清單、編號清單、表格、分頁。
## 參數
- file_name (str): 輸出檔名（例如 'report.docx'），會存到 output 目錄
- content (str): 文件內容，使用簡易標記格式（# 標題、- 項目、| 表格等）
- title (str, 可選): 文件標題，加在文件最上方
## 使用時機
- 需要產生 Word 文件時使用。
- 讀取現有 DOCX 請用 file_reader。
