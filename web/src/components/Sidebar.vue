<script setup>
defineProps({
  conversations: { type: Array, required: true },
  activeId: { type: String, default: null },
})

const emit = defineEmits(['select', 'create'])

function formatTime(ts) {
  return new Date(ts).toLocaleString('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="mark" aria-hidden="true" />
      <div>
        <div class="brand-name">Diagnose</div>
        <div class="brand-sub">代码诊断对话</div>
      </div>
    </div>

    <button class="new-btn" type="button" @click="emit('create')">
      <span class="plus">+</span>
      新对话
    </button>

    <div class="section-label">历史对话</div>
    <div class="list">
      <button
        v-for="item in conversations"
        :key="item.id"
        type="button"
        class="item"
        :class="{ active: item.id === activeId }"
        @click="emit('select', item.id)"
      >
        <div class="item-title">{{ item.title }}</div>
        <div class="item-meta">
          <span>{{ item.messages.length }} 条</span>
          <span>{{ formatTime(item.updatedAt) }}</span>
        </div>
      </button>
      <div v-if="!conversations.length" class="empty">暂无历史，点击上方开始</div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 280px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px 14px;
  color: #ecfdf8;
  background:
    linear-gradient(180deg, rgba(20, 184, 166, 0.12), transparent 28%),
    var(--sidebar);
  border-right: 1px solid rgba(204, 251, 241, 0.08);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 8px 10px;
}

.mark {
  width: 36px;
  height: 36px;
  border-radius: 12px;
  background:
    radial-gradient(circle at 30% 30%, #99f6e4, transparent 55%),
    linear-gradient(145deg, #14b8a6, #0f766e);
  box-shadow: 0 8px 20px rgba(20, 184, 166, 0.28);
}

.brand-name {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.brand-sub {
  margin-top: 2px;
  font-size: 0.78rem;
  color: rgba(236, 253, 248, 0.55);
}

.new-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 42px;
  border-radius: 12px;
  color: #042f2e;
  font-weight: 600;
  background: linear-gradient(135deg, #5eead4, #14b8a6);
  transition: transform 0.18s ease, filter 0.18s ease;
}

.new-btn:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}

.plus {
  font-size: 1.2rem;
  line-height: 1;
}

.section-label {
  padding: 4px 10px 0;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(236, 253, 248, 0.42);
}

.list {
  flex: 1;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-right: 2px;
}

.item {
  text-align: left;
  padding: 12px 12px 10px;
  border-radius: 12px;
  color: rgba(236, 253, 248, 0.88);
  transition: background 0.16s ease, transform 0.16s ease;
}

.item:hover {
  background: var(--sidebar-hover);
}

.item.active {
  background: var(--sidebar-active);
  box-shadow: inset 0 0 0 1px rgba(94, 234, 212, 0.28);
}

.item-title {
  font-size: 0.92rem;
  font-weight: 550;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.item-meta {
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.72rem;
  color: rgba(236, 253, 248, 0.42);
}

.empty {
  padding: 18px 10px;
  font-size: 0.85rem;
  color: rgba(236, 253, 248, 0.4);
}

@media (max-width: 860px) {
  .sidebar {
    width: 72px;
    padding: 14px 8px;
  }

  .brand-name,
  .brand-sub,
  .section-label,
  .item-meta,
  .new-btn {
    display: none;
  }

  .brand {
    justify-content: center;
    padding: 0;
  }

  .item-title {
    font-size: 0;
  }

  .item-title::first-letter {
    font-size: 0.85rem;
  }

  .item {
    display: grid;
    place-items: center;
    min-height: 44px;
  }
}
</style>
