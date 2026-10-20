import { computed, ref } from 'vue'

function uid(prefix = 'id') {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

function nowLabel() {
  return new Date().toLocaleString('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const seed = [
  {
    id: 'c_demo_1',
    title: '门店自动入库失败排查',
    updatedAt: Date.now() - 3600_000,
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: '连锁店自动入库偶发失败，帮我从代码侧排查方向。',
        images: [],
        createdAt: Date.now() - 3600_000,
      },
      {
        id: 'm2',
        role: 'assistant',
        content:
          '可以从入库触发链路、幂等校验、库存锁定与回滚日志三条线并行看。先确认失败时的 image_tag / 门店编码，我再帮你缩小到具体模块。',
        images: [],
        createdAt: Date.now() - 3500_000,
      },
    ],
  },
  {
    id: 'c_demo_2',
    title: 'Wintogether 启动类说明',
    updatedAt: Date.now() - 86400_000,
    messages: [
      {
        id: 'm3',
        role: 'user',
        content: 'WintogetherWebOpApplication 做了什么？',
        images: [],
        createdAt: Date.now() - 86400_000,
      },
      {
        id: 'm4',
        role: 'assistant',
        content:
          '它是 web-op 模块的 Spring Boot 启动入口，开启了异步、缓存、Feign 与 MapperScan，并指定了多包扫描范围。',
        images: [],
        createdAt: Date.now() - 86300_000,
      },
    ],
  },
]

const conversations = ref(structuredClone(seed))
const activeId = ref(conversations.value[0]?.id ?? null)

export function useChats() {
  const activeChat = computed(
    () => conversations.value.find((c) => c.id === activeId.value) ?? null,
  )

  const sortedConversations = computed(() =>
    [...conversations.value].sort((a, b) => b.updatedAt - a.updatedAt),
  )

  function selectChat(id) {
    activeId.value = id
  }

  function createChat() {
    const chat = {
      id: uid('c'),
      title: '新对话',
      updatedAt: Date.now(),
      messages: [],
    }
    conversations.value.unshift(chat)
    activeId.value = chat.id
    return chat
  }

  function renameFromFirstMessage(chat, text) {
    if (chat.title !== '新对话') return
    const title = text.trim().replace(/\s+/g, ' ').slice(0, 18)
    if (title) chat.title = title
  }

  function sendMessage({ text, images }) {
    let chat = activeChat.value
    if (!chat) {
      chat = createChat()
    }

    const content = (text || '').trim()
    if (!content && (!images || images.length === 0)) return

    chat.messages.push({
      id: uid('m'),
      role: 'user',
      content,
      images: images || [],
      createdAt: Date.now(),
    })
    renameFromFirstMessage(chat, content || '图片消息')
    chat.updatedAt = Date.now()

    // 样式阶段：模拟助手回复
    window.setTimeout(() => {
      const target = conversations.value.find((c) => c.id === chat.id)
      if (!target) return
      const imgHint = images?.length ? `（已收到 ${images.length} 张图片）` : ''
      target.messages.push({
        id: uid('m'),
        role: 'assistant',
        content: content
          ? `已收到：${content}${imgHint}\n\n（当前为界面预览，尚未接入诊断 Agent）`
          : `已收到图片${imgHint}\n\n（当前为界面预览，尚未接入诊断 Agent）`,
        images: [],
        createdAt: Date.now(),
      })
      target.updatedAt = Date.now()
    }, 420)
  }

  function formatTime(ts) {
    return new Date(ts).toLocaleString('zh-CN', {
      month: 'numeric',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return {
    conversations: sortedConversations,
    activeId,
    activeChat,
    selectChat,
    createChat,
    sendMessage,
    formatTime,
    nowLabel,
  }
}
