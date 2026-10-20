<script setup>
import Sidebar from './components/Sidebar.vue'
import ChatPanel from './components/ChatPanel.vue'
import { useChats } from './composables/useChats'

const {
  conversations,
  activeId,
  activeChat,
  selectChat,
  createChat,
  sendMessage,
} = useChats()
</script>

<template>
  <div class="shell">
    <Sidebar
      :conversations="conversations"
      :active-id="activeId"
      @select="selectChat"
      @create="createChat"
    />
    <ChatPanel
      :chat="activeChat"
      @send="sendMessage"
      @create="createChat"
    />
  </div>
</template>

<style scoped>
.shell {
  height: 100%;
  display: flex;
  overflow: hidden;
  animation: fade-in 0.4s ease both;
}

@keyframes fade-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>
