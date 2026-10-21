const BASE = '/api/v1/chats'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...(options.headers || {}),
    },
    ...options,
  })
  const text = await res.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { detail: text }
  }
  if (!res.ok) {
    const detail = data?.detail || res.statusText
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return data
}

export function listConversations() {
  return request('')
}

export function createConversation(title = '新对话') {
  return request('', {
    method: 'POST',
    body: JSON.stringify({ title }),
  })
}

export function getMessages(conversationId) {
  return request(`/${conversationId}/messages`)
}

export function sendMessage(conversationId, payload) {
  return request(`/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * SSE 发送消息。onEvent({ event, data })
 */
export async function sendMessageStream(conversationId, payload, onEvent) {
  const res = await fetch(`${BASE}/${conversationId}/messages/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const text = await res.text()
    let detail = text
    try {
      detail = JSON.parse(text)?.detail || text
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  if (!res.body) {
    throw new Error('浏览器不支持流式响应')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    buffer = buffer.replace(/\r\n/g, '\n')

    let sep
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      if (!chunk.trim()) continue

      let eventName = 'message'
      const dataLines = []
      for (const line of chunk.split('\n')) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trimStart())
        }
      }
      const raw = dataLines.join('\n')
      let data = raw
      try {
        data = raw ? JSON.parse(raw) : null
      } catch {
        /* keep string */
      }
      onEvent?.({ event: eventName, data })
    }
  }
}

export async function uploadDataUrl(dataUrl, conversationId) {
  return request('/assets/upload-data-url', {
    method: 'POST',
    body: JSON.stringify({
      data_url: dataUrl,
      conversation_id: conversationId ?? null,
    }),
  })
}

export function clientMsgId() {
  return `cli_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}
