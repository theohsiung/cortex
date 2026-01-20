# Local Filesystem MCP Tool Design

## Summary

將 filesystem tool 從 Docker 容器內移到本地執行，使用 `@anthropic/mcp-filesystem`，讓檔案可以永久保留。

## Changes

### Before
- Filesystem: Docker 容器內的 `filesystem_server.py`
- Shell: Docker 容器內的 `shell_server.py`
- 容器刪除後檔案消失

### After
- Filesystem: 本地執行 `@anthropic/mcp-filesystem`，限制 `userspace/{user_id}/`
- Shell: 仍在 Docker 容器內，mount `userspace/{user_id}/` 到 `/workspace`
- 檔案永久保留

## Implementation

1. 移除 Docker 內的 filesystem_server.py
2. 新增 user_id 參數（選填，預設自動產生 UUID）
3. 建立 userspace 目錄結構
4. Filesystem tool 改用 `npx @anthropic/mcp-filesystem`
5. Docker volume mount 改為 `userspace/{user_id}/`
6. 更新測試
