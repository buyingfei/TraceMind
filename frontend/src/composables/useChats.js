import { computed, onMounted, ref } from 'vue'
import {
  clientMsgId,
  createConversation as apiCreate,
  getMessages,
  listConversations,
  sendMessageStream,
  uploadDataUrl,
} from '../api/chat'

function toUiConversation(row) {
  return {
    id: row.id,
    title: row.title || '新对话',
    updatedAt: row.last_message_at || row.updated_at || row.created_at,
    messageCount: row.message_count || 0,
    messages: [],
    loaded: false,
  }
}

function toUiMessage(msg) {
  return {
    id: msg.id,
    role: msg.role,
    content: msg.content || '',
    images: (msg.images || []).map((img) => ({
      id: img.id,
      url: img.url,
      name: img.relative_path,
    })),
    createdAt: msg.created_at,
    status: msg.status,
    streaming: false,
  }
}

const conversations = ref([])
const activeId = ref(null)
const loadingList = ref(false)
const loadingMessages = ref(false)
const sending = ref(false)
const error = ref('')
const streamStatus = ref('')

export function useChats() {
  const activeChat = computed(
    () => conversations.value.find((c) => c.id === activeId.value) ?? null,
  )

  const sortedConversations = computed(() =>
    [...conversations.value].sort((a, b) => {
      const ta = new Date(a.updatedAt || 0).getTime()
      const tb = new Date(b.updatedAt || 0).getTime()
      return tb - ta
    }),
  )

  async function refreshList() {
    loadingList.value = true
    error.value = ''
    try {
      const data = await listConversations()
      const items = (data.items || []).map(toUiConversation)
      const prev = new Map(conversations.value.map((c) => [c.id, c]))
      conversations.value = items.map((item) => {
        const old = prev.get(item.id)
        if (old?.loaded) {
          return { ...item, messages: old.messages, loaded: true }
        }
        return item
      })
      if (activeId.value == null && conversations.value.length) {
        await selectChat(conversations.value[0].id)
      } else if (
        activeId.value != null &&
        !conversations.value.some((c) => c.id === activeId.value)
      ) {
        activeId.value = conversations.value[0]?.id ?? null
        if (activeId.value != null) await selectChat(activeId.value)
      }
    } catch (e) {
      error.value = e.message || String(e)
    } finally {
      loadingList.value = false
    }
  }

  async function selectChat(id) {
    activeId.value = id
    const chat = conversations.value.find((c) => c.id === id)
    if (!chat || chat.loaded) return
    loadingMessages.value = true
    error.value = ''
    try {
      const data = await getMessages(id)
      chat.messages = (data.items || []).map(toUiMessage)
      chat.loaded = true
    } catch (e) {
      error.value = e.message || String(e)
    } finally {
      loadingMessages.value = false
    }
  }

  async function createChat() {
    error.value = ''
    try {
      const row = await apiCreate('新对话')
      const chat = toUiConversation(row)
      chat.loaded = true
      chat.messages = []
      conversations.value.unshift(chat)
      activeId.value = chat.id
      return chat
    } catch (e) {
      error.value = e.message || String(e)
      throw e
    }
  }

  async function sendMessage({ text, images }) {
    const content = (text || '').trim()
    if (!content && (!images || !images.length)) return
    if (sending.value) return

    sending.value = true
    error.value = ''
    streamStatus.value = '上传中…'
    const tempId = `tmp_${Date.now()}`

    try {
      let chat = activeChat.value
      if (!chat) {
        chat = await createChat()
      }

      const relativePaths = []
      for (const img of images || []) {
        const dataUrl = img.url || img.dataUrl
        if (!dataUrl) continue
        const uploaded = await uploadDataUrl(dataUrl, chat.id)
        relativePaths.push(uploaded.relative_path)
      }

      const target = conversations.value.find((c) => c.id === chat.id)
      if (!target) return

      streamStatus.value = '生成中…'
      let streamBubble = null

      await sendMessageStream(
        chat.id,
        {
          content,
          client_msg_id: clientMsgId(),
          relative_paths: relativePaths,
          call_llm: true,
        },
        ({ event, data }) => {
          if (event === 'meta' && data?.stage === 'user_saved' && data.user_message) {
            const uiUser = toUiMessage(data.user_message)
            if (!target.messages.some((m) => m.id === uiUser.id)) {
              target.messages.push(uiUser)
            }
            if (data.conversation) {
              target.title = data.conversation.title
              target.updatedAt =
                data.conversation.last_message_at || data.conversation.updated_at
              target.messageCount = data.conversation.message_count
            }
            // 占位助手气泡
            streamBubble = {
              id: tempId,
              role: 'assistant',
              content: '',
              images: [],
              createdAt: new Date().toISOString(),
              status: 'streaming',
              streaming: true,
            }
            target.messages.push(streamBubble)
            if (data.stage) streamStatus.value = data.message || streamStatus.value
          } else if (event === 'meta') {
            if (data?.message) streamStatus.value = data.message
            else if (data?.stage === 'vision') streamStatus.value = '正在 OCR 识别…'
            else if (data?.stage === 'retrieve' || data?.stage === 'retrieve_done') {
              streamStatus.value =
                data?.message ||
                (data?.stage === 'retrieve_done'
                  ? `检索完成，命中 ${data?.hit_count ?? 0} 条代码`
                  : '正在检索代码…')
            }
            else if (data?.stage === 'retry') {
              streamStatus.value = data?.message || '公司模型重试中…'
            }
            else if (data?.stage === 'sql_query') {
              streamStatus.value = data?.message || '正在查询业务只读库核对数据…'
            }
            else if (data?.stage === 'sql_done') {
              streamStatus.value = data?.message || '查库完成'
            }
            else if (data?.stage === 'answer' || data?.stage === 'answer_start') {
              streamStatus.value = data?.message || '公司模型生成中…'
            }

          } else if (event === 'delta') {
            if (!streamBubble) {
              streamBubble = {
                id: tempId,
                role: 'assistant',
                content: '',
                images: [],
                createdAt: new Date().toISOString(),
                status: 'streaming',
                streaming: true,
              }
              target.messages.push(streamBubble)
            }
            streamBubble.content += data?.text || ''
          } else if (event === 'done') {
            if (data?.assistant_message) {
              const uiAsst = toUiMessage(data.assistant_message)
              const idx = target.messages.findIndex((m) => m.id === tempId)
              if (idx >= 0) target.messages.splice(idx, 1, uiAsst)
              else if (!target.messages.some((m) => m.id === uiAsst.id)) {
                target.messages.push(uiAsst)
              }
            } else if (streamBubble) {
              streamBubble.streaming = false
              streamBubble.status = 'normal'
            }
            if (data?.conversation) {
              target.title = data.conversation.title
              target.updatedAt =
                data.conversation.last_message_at || data.conversation.updated_at
              target.messageCount = data.conversation.message_count
            }
            streamStatus.value = ''
          } else if (event === 'error') {
            error.value = data?.message || '生成失败'
            if (data?.assistant_message) {
              const uiAsst = toUiMessage(data.assistant_message)
              const idx = target.messages.findIndex((m) => m.id === tempId)
              if (idx >= 0) target.messages.splice(idx, 1, uiAsst)
            } else if (streamBubble) {
              streamBubble.streaming = false
              streamBubble.status = 'failed'
              if (!streamBubble.content) {
                streamBubble.content = `模型调用失败：${error.value}`
              }
            }
            streamStatus.value = ''
          }
        },
      )
      target.loaded = true
    } catch (e) {
      error.value = e.message || String(e)
      streamStatus.value = ''
    } finally {
      sending.value = false
    }
  }

  onMounted(() => {
    refreshList()
  })

  return {
    conversations: sortedConversations,
    activeId,
    activeChat,
    loadingList,
    loadingMessages,
    sending,
    error,
    streamStatus,
    selectChat,
    createChat,
    sendMessage,
    refreshList,
  }
}
