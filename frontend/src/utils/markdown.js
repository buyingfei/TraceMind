import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  gfm: true,
  breaks: true,
})

/**
 * 将 Markdown 转为安全 HTML（助手气泡展示用）。
 */
export function renderMarkdown(source) {
  const raw = String(source || '')
  if (!raw.trim()) return ''
  const html = marked.parse(raw, { async: false })
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
  })
}
