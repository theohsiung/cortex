# audio_transcription
使用 OpenAI Whisper 將音訊檔案轉換為文字。
## 參數
- audio_path (str): 本地音訊檔案路徑（必須先用 download_file 下載到本地）
- language (str, 選填): 語言代碼（如 "zh", "en"）
## 使用時機
- 需要轉錄音訊或語音內容時使用。
- 檔案必須已在本地端，如果是網路上的音訊，先用 download_file 下載。
