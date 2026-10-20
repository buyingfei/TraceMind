<script setup>
import { computed, nextTick, ref, watch } from 'vue'

const props = defineProps({
  chat: { type: Object, default: null },
  sending: { type: Boolean, default: false },
  error: { type: String, default: '' },
  streamStatus: { type: String, default: '' },
})

const emit = defineEmits(['send', 'create'])

const draft = ref('')
const pendingImages = ref([])
const listRef = ref(null)
const composerRef = ref(null)

const title = computed(() => props.chat?.title || '新对话')
const messages = computed(() => props.chat?.messages || [])
const isEmpty = computed(() => messages.value.length === 0)

watch(
  () => [props.chat?.id, messages.value.length],
  async () => {
    await nextTick()
    if (listRef.value) {
      listRef.value.scrollTop = listRef.value.scrollHeight
    }
  },
)

function fileToPreview(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      resolve({
        id: `img_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        name: file.name || 'paste.png',
        url: String(reader.result),
      })
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

async function addImageFiles(files) {
  const images = Array.from(files || []).filter((f) => f.type.startsWith('image/'))
  for (const file of images) {
    const preview = await fileToPreview(file)
    pendingImages.value.push(preview)
  }
}

async function onPaste(event) {
  const items = event.clipboardData?.items
  if (!items) return
  const imageItems = Array.from(items).filter((item) => item.type.startsWith('image/'))
  if (!imageItems.length) return
  event.preventDefault()
  const files = imageItems.map((item) => item.getAsFile()).filter(Boolean)
  await addImageFiles(files)
}

function removeImage(id) {
  pendingImages.value = pendingImages.value.filter((img) => img.id !== id)
}

function onKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

function submit() {
  if (props.sending) return
  const text = draft.value
  const images = pendingImages.value.map((img) => ({ ...img }))
  if (!text.trim() && !images.length) return
  emit('send', { text, images })
  draft.value = ''
  pendingImages.value = []
  nextTick(() => composerRef.value?.focus())
}

async function onPickFile(event) {
  await addImageFiles(event.target.files)
  event.target.value = ''
}
</script>

<template>
  <section class="panel">
    <header class="topbar">
      <div>
        <h1>{{ title }}</h1>
        <p>支持粘贴截图到输入框 · Shift+Enter 换行</p>
      </div>
      <button class="ghost" type="button" @click="emit('create')">继续新对话</button>
    </header>

    <div ref="listRef" class="messages">
      <div v-if="isEmpty" class="hero">
        <div class="hero-mark" />
        <h2>Diagnose</h2>
        <p>把问题、报错或截图丢进来，右侧可持续开新对话。</p>
        <div class="hints">
          <button type="button" @click="draft = '帮我分析自动入库失败的可能原因'">
            分析自动入库失败
          </button>
          <button type="button" @click="draft = '解释一下 WintogetherWebOpApplication'">
            解释启动类
          </button>
          <button type="button" @click="draft = '这段日志里哪里最可疑？'">
            看日志可疑点
          </button>
        </div>
      </div>

      <div
        v-for="msg in messages"
        :key="msg.id"
        class="row"
        :class="msg.role"
      >
        <div class="bubble">
          <div v-if="msg.images?.length" class="msg-images">
            <img
              v-for="img in msg.images"
              :key="img.id"
              :src="img.url"
              :alt="img.name"
            />
          </div>
          <div v-if="msg.content || msg.streaming" class="text">
            {{ msg.content }}<span v-if="msg.streaming" class="caret" aria-hidden="true" />
          </div>
        </div>
      </div>
    </div>

    <div class="composer-wrap">
      <div v-if="pendingImages.length" class="previews">
        <div v-for="img in pendingImages" :key="img.id" class="preview">
          <img :src="img.url" :alt="img.name" />
          <button type="button" class="x" aria-label="移除图片" @click="removeImage(img.id)">
            ×
          </button>
        </div>
      </div>

      <div class="composer" @paste="onPaste">
        <textarea
          ref="composerRef"
          v-model="draft"
          rows="1"
          placeholder="输入消息，或 Ctrl+V 粘贴图片…"
          @keydown="onKeydown"
        />
        <div class="actions">
          <label class="attach" title="上传图片">
            <input type="file" accept="image/*" multiple hidden @change="onPickFile" />
            <span>图片</span>
          </label>
          <button class="send" type="button" :disabled="sending" @click="submit">
            {{ sending ? '发送中…' : '发送' }}
          </button>
        </div>
      </div>
      <div v-if="streamStatus" class="status-line">{{ streamStatus }}</div>
      <div v-if="error" class="error-line">{{ error }}</div>
      <div class="footnote">SSE 流式输出 · 消息与图片已持久化</div>
    </div>
  </section>
</template>

<style scoped>
.panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  position: relative;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 28px 10px;
}

.topbar h1 {
  margin: 0;
  font-size: 1.15rem;
  font-weight: 650;
}

.topbar p {
  margin: 4px 0 0;
  font-size: 0.8rem;
  color: var(--muted);
}

.ghost {
  height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  color: var(--accent);
  background: rgba(255, 255, 255, 0.7);
  box-shadow: inset 0 0 0 1px var(--line);
  transition: background 0.16s ease, transform 0.16s ease;
}

.ghost:hover {
  background: #fff;
  transform: translateY(-1px);
}

.messages {
  flex: 1;
  overflow: auto;
  padding: 8px 28px 20px;
}

.hero {
  max-width: 560px;
  margin: 12vh auto 0;
  text-align: center;
  animation: rise 0.45s ease both;
}

.hero-mark {
  width: 64px;
  height: 64px;
  margin: 0 auto 16px;
  border-radius: 20px;
  background:
    radial-gradient(circle at 30% 28%, #99f6e4, transparent 50%),
    linear-gradient(145deg, #2dd4bf, #0f766e);
  box-shadow: 0 16px 40px rgba(15, 118, 110, 0.28);
  animation: pulse 2.8s ease-in-out infinite;
}

.hero h2 {
  margin: 0;
  font-size: 2rem;
  letter-spacing: -0.03em;
}

.hero p {
  margin: 10px 0 0;
  color: var(--muted);
  line-height: 1.6;
}

.hints {
  margin-top: 22px;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
}

.hints button {
  padding: 10px 14px;
  border-radius: 999px;
  color: var(--ink-soft);
  background: rgba(255, 255, 255, 0.72);
  box-shadow: inset 0 0 0 1px var(--line);
  transition: transform 0.16s ease, background 0.16s ease;
}

.hints button:hover {
  background: #fff;
  transform: translateY(-1px);
}

.row {
  display: flex;
  margin: 12px 0;
  animation: rise 0.28s ease both;
}

.row.user {
  justify-content: flex-end;
}

.row.assistant {
  justify-content: flex-start;
}

.bubble {
  max-width: min(720px, 86%);
  padding: 12px 14px;
  border-radius: 18px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}

.user .bubble {
  color: #f0fdfa;
  background: linear-gradient(145deg, #0f766e, #115e59);
  border-bottom-right-radius: 6px;
  box-shadow: 0 10px 24px rgba(15, 118, 110, 0.22);
}

.assistant .bubble {
  background: var(--bot-bubble);
  border-bottom-left-radius: 6px;
  box-shadow: var(--shadow);
}

.msg-images {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin-bottom: 8px;
}

.msg-images img {
  border-radius: 12px;
  max-height: 220px;
  object-fit: cover;
  width: 100%;
}

.composer-wrap {
  padding: 0 28px 18px;
}

.previews {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.preview {
  position: relative;
  width: 72px;
  height: 72px;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px var(--line), var(--shadow);
}

.preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.preview .x {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  color: #fff;
  background: rgba(0, 0, 0, 0.55);
  line-height: 1;
}

.composer {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 12px 12px 12px 16px;
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(10px);
  box-shadow: inset 0 0 0 1px var(--line), var(--shadow);
  transition: box-shadow 0.18s ease;
}

.composer:focus-within {
  box-shadow:
    inset 0 0 0 1px rgba(20, 184, 166, 0.45),
    0 18px 40px rgba(15, 118, 110, 0.14);
}

textarea {
  flex: 1;
  resize: none;
  border: none;
  outline: none;
  background: transparent;
  min-height: 28px;
  max-height: 160px;
  padding: 8px 0;
  color: var(--ink);
}

textarea::placeholder {
  color: #9aa8b8;
}

.actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.attach {
  height: 36px;
  padding: 0 12px;
  display: grid;
  place-items: center;
  border-radius: 999px;
  color: var(--accent);
  background: var(--accent-soft);
  cursor: pointer;
}

.send {
  height: 36px;
  padding: 0 16px;
  border-radius: 999px;
  color: #042f2e;
  font-weight: 650;
  background: linear-gradient(135deg, #5eead4, #14b8a6);
  transition: transform 0.16s ease, filter 0.16s ease;
}

.send:hover {
  filter: brightness(1.04);
  transform: translateY(-1px);
}

.footnote {
  margin-top: 8px;
  text-align: center;
  font-size: 0.72rem;
  color: var(--muted);
}

.error-line {
  margin-top: 8px;
  text-align: center;
  font-size: 0.8rem;
  color: var(--danger);
}

.status-line {
  margin-top: 8px;
  text-align: center;
  font-size: 0.8rem;
  color: var(--accent);
}

.caret {
  display: inline-block;
  width: 2px;
  height: 1em;
  margin-left: 2px;
  vertical-align: text-bottom;
  background: currentColor;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.send:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse {
  0%,
  100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.04);
  }
}

@media (max-width: 860px) {
  .topbar,
  .messages,
  .composer-wrap {
    padding-left: 14px;
    padding-right: 14px;
  }

  .hero {
    margin-top: 6vh;
  }
}
</style>
