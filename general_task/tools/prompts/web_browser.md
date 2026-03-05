# web_browser
使用 Playwright（支援 JS 渲染）或 requests 擷取網頁的文字內容。
## 參數
- url (str): 目標網頁 URL
## 使用時機
- 需要讀取特定網頁的 HTML 文字內容時使用（文章、文件、表格頁面等）。
- 搭配 web_search 取得的 URL 使用，深入閱讀頁面內容。
## 不要搞混
- 讀取網頁 HTML 頁面 → web_browser（本工具）
- 下載資料檔案（CSV、Excel、PDF）→ download_file
- web_browser 適合讀 HTML 頁面，不適合下載二進位檔案。
