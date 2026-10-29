import { useEffect, useRef, useState } from 'react'
import {
  MAX_HISTORY, MAX_INPUT, readConversation, requestReply, saveConversation,
  type Message,
} from './conversation'

export function useConversation() {
  const [messages, setMessages] = useState<Message[]>(readConversation)
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const busy = useRef(false)
  const active = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!sending && messages.length % 2 === 0) saveConversation(messages)
  }, [messages, sending])

  useEffect(() => () => active.current?.abort(), [])

  async function sendMessage(value: string): Promise<boolean> {
    const content = value.trim()
    if (!content || content.length > MAX_INPUT || busy.current) return false
    busy.current = true
    const controller = new AbortController()
    active.current = controller
    const user: Message = { role: 'user', content }
    const previous = messages
    setMessages([...previous, user])
    setDraft('')
    setError(null)
    setSending(true)
    try {
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(60_000)])
      const reply = await requestReply([...previous.slice(-38), user], signal)
      if (controller.signal.aborted) return false
      const assistant: Message = { role: 'assistant', content: reply }
      setMessages([...previous, user, assistant].slice(-MAX_HISTORY))
      return true
    } catch (cause) {
      if (controller.signal.aborted) return false
      setMessages(previous)
      setDraft(content)
      const timeout = cause instanceof Error && cause.name === 'TimeoutError'
      setError(timeout ? 'That reply is taking a little while. Your message is ready to try again.' :
        cause instanceof TypeError ? "Can't reach Joi right now. Your message is saved here—try again in a moment." :
        cause instanceof Error ? cause.message : "Something interrupted that reply. Let's try again.")
      return false
    } finally {
      busy.current = false
      active.current = null
      if (!controller.signal.aborted) setSending(false)
    }
  }

  function newConversation(): void {
    if (busy.current) return
    setMessages([])
    setDraft('')
    setError(null)
    saveConversation([])
  }

  return { messages, draft, setDraft, sending, error, sendMessage, newConversation }
}
