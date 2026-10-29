export type Message = { role: 'user' | 'assistant'; content: string }
export type Connection = 'checking' | 'ready' | 'offline'

export const STORAGE_KEY = 'joi.conversation.v1'
export const MAX_HISTORY = 200
export const MAX_INPUT = 240

export function readConversation(): Message[] {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    if (
      Array.isArray(saved) && saved.length % 2 === 0 &&
      saved.every((message, index) =>
        message && message.role === (index % 2 ? 'assistant' : 'user') &&
        typeof message.content === 'string' && message.content.length <= 4000,
      )
    ) return saved.slice(-MAX_HISTORY) as Message[]
  } catch { /* Chat also works when browser storage is unavailable. */ }
  return []
}

export function saveConversation(messages: Message[]): void {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-MAX_HISTORY))) }
  catch { /* A storage restriction must not interrupt a conversation. */ }
}

export async function checkConnection(signal?: AbortSignal): Promise<void> {
  const response = await fetch('/api/status', { signal })
  if (!response.ok) throw new Error('Joi is not connected yet.')
  const result: unknown = await response.json()
  if (!result || typeof result !== 'object' || !('name' in result) || result.name !== 'Joi') {
    throw new Error('Joi is not connected yet.')
  }
}

export async function requestReply(messages: Message[], signal: AbortSignal): Promise<string> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: messages.slice(-39) }),
    signal,
  })
  const result: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = result && typeof result === 'object' && 'error' in result ? result.error : null
    throw new Error(typeof detail === 'string' ? detail : "Joi couldn't finish that reply. Try again?")
  }
  if (!result || typeof result !== 'object' || !('reply' in result) ||
      typeof result.reply !== 'string' || !result.reply.trim()) {
    throw new Error('No reply came through. Your message is ready to try again.')
  }
  return result.reply
}
