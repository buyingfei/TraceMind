# Diagnose Web（对话客户端）

Vue 3 + Vite。左侧历史、右侧对话；粘贴图片会上传到后端 `data/chat_uploads`，消息入库 MySQL。

## 启动

先启后端（项目根目录）：

```bash
uvicorn app.main:app --reload --port 8000
```

再启前端：

```bash
cd web
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`（Vite 已把 `/api` 代理到 `8000`）。

## 操作

- 新对话 / 历史切换
- `Ctrl+V` 粘贴截图后发送
- 带图：moondream2 → qwen2.5-7b（失败 gpt-sol 兜底）
- 回答使用 SSE 流式打字机：`POST /api/v1/chats/{id}/messages/stream`
