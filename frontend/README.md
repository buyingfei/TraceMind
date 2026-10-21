# Diagnose Frontend（对话客户端）

Vue 3 + Vite。左侧历史、右侧对话；粘贴图片会上传到后端 `data/chat_uploads`，消息入库 MySQL。

## 启动

先启后端（仓库根目录）：

```bash
python -m app.main
# 或: uvicorn app.main:app --reload --port 8000
```

再启前端：

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`（Vite 已把 `/api` 代理到 `8000`）。

## 操作

- 新对话 / 历史切换
- `Ctrl+V` 粘贴截图后发送
- 带图：PaddleOCR → 模型综合回答
- 回答使用 SSE 流式：`POST /api/v1/chats/{id}/messages/stream`
