# Cortex 專案開發規範

## 1. 專案簡介
- **定位**: 基於 Google ADK 的多步驟 AI Agent 框架（自動將複雜任務拆解為 DAG，並行執行、驗證、重新規劃）。
- **啟動方式**:
  - API 服務: `uv run uvicorn api:app --reload --host 0.0.0.0 --port 8999`
  - 腳本執行: `bash scripts/run.sh` 或 `uv run python example.py`

## 2. 程式碼實作規範 (Critical)
在編寫或修改 Python 程式碼時，必須嚴格遵守以下專案限制與風格：
- **依賴管理**: 統一使用 `uv` 執行所有套件管理與腳本指令。
- **Type Hints 型別提示**: 
  - 檔案開頭必須包含 `from __future__ import annotations`。
  - **禁止使用** `typing.Optional[X]`，一律改用 `X | None` 語法。
  - 所有函式簽名都必須標註回傳型別，如果沒有回傳值必須標註 `-> None`。
- **架構擴充**: 若要自訂 Agent (Custom Executor)，請參考 `general_task/` 目錄結構，需要實作工廠函式並在 `config.toml` 中註冊 `intent`。開發工具函式遇模型幻覺時，要妥善透過 `ToolManager` 加上別名機制 (alias)。

## 3. 測試與品質檢驗 (Quality Checks)
每次修改程式碼完成後，必須確保通過以下檢查（AI 應優先確保這些指令跑得過）：
- `uv run ruff check --fix .` (Linting 和自動修復)
- `uv run ruff format .` (程式碼格式化)
- `uv run mypy .` (靜態型別檢查，必須零錯誤)
- `uv run pytest tests/ -v` (單元測試必須全數通過)

## 4. AI 開發核心工作流
在處理任何使用者需求時，必須遵守以下順序：

1. **盤點 Skills**: 開始工作前，必須先檢查專案中有哪些可用的 skills 能夠輔助當前任務。
2. **啟動 Brainstorming (架構與規劃)**: 
   - 針對**新功能開發、架構變更或複雜問題解決**，**必須**優先使用 `@brainstorming` skill。
   - 透過該流程釐清需求、非功能性需求 (NFR) 並產出設計方案後，才能進入實作。*(若是微小的 Bug 修復或簡單文案調整，可視情況跳過此步驟)*。
3. **善用 Skills 實作**: 在規劃與實作的過程中，根據使用者需求，只要有合適的領域專門 skills (例如：前端、資料庫、測試)，就必須調用它們來完成任務，避免完全從零硬寫。
