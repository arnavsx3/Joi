import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { STORAGE_KEY, type Message } from './conversation'
import { useConversation } from './useConversation'

const reply = (content: string) => new Response(JSON.stringify({ reply: content }), {
  status: 200, headers: { 'Content-Type': 'application/json' },
})
const history: Message[] = [
  { role: 'user', content: 'My name is Arnav.' },
  { role: 'assistant', content: 'Hi, Arnav.' },
]

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('fetch', vi.fn())
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('conversation migration and requests', () => {
  it('restores the existing vanilla frontend’s conversation without changing its storage key', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    const { result } = renderHook(useConversation)
    expect(result.current.messages).toEqual(history)
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual(history)
  })

  it('does not send corrupted or incomplete stored turns as model context', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ role: 'assistant', content: 'orphan reply' }]))
    const { result } = renderHook(useConversation)
    expect(result.current.messages).toEqual([])
  })

  it('sends the last complete context window while preserving a longer local conversation', async () => {
    const saved: Message[] = Array.from({ length: 60 }, (_, index) => ({
      role: index % 2 ? 'assistant' : 'user', content: `Turn ${index}`,
    }))
    localStorage.setItem(STORAGE_KEY, JSON.stringify(saved))
    vi.mocked(fetch).mockResolvedValue(reply('A new reply'))
    const { result } = renderHook(useConversation)
    await act(async () => { await result.current.sendMessage('One more thought') })
    const body = JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)
    expect(body.messages).toEqual([...saved.slice(-38), { role: 'user', content: 'One more thought' }])
    expect(result.current.messages).toHaveLength(62)
    expect(result.current.messages.at(-1)).toEqual({ role: 'assistant', content: 'A new reply' })
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual(result.current.messages)
  })

  it('restores a failed draft and keeps completed history intact', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    vi.mocked(fetch).mockResolvedValue(new Response(JSON.stringify({ error: 'Please try again.' }), { status: 500 }))
    const { result } = renderHook(useConversation)
    await act(async () => { await result.current.sendMessage('I had a long day.') })
    expect(result.current.draft).toBe('I had a long day.')
    expect(result.current.messages).toEqual(history)
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual(history)
    expect(result.current.error).toBe('Please try again.')
    expect(result.current.sending).toBe(false)
  })

  it('blocks rapid duplicate sends and reset while a reply is pending', async () => {
    let finish!: (response: Response) => void
    vi.mocked(fetch).mockReturnValue(new Promise<Response>((resolve) => { finish = resolve }))
    const { result } = renderHook(useConversation)
    let first!: Promise<boolean>
    let duplicate!: Promise<boolean>
    act(() => {
      first = result.current.sendMessage('Hello Joi')
      duplicate = result.current.sendMessage('Hello Joi')
      result.current.newConversation()
    })
    expect(await duplicate).toBe(false)
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(result.current.sending).toBe(true)
    expect(result.current.messages).toEqual([{ role: 'user', content: 'Hello Joi' }])
    await act(async () => { finish(reply('Hello')); await first })
    expect(result.current.messages).toHaveLength(2)
    act(() => result.current.newConversation())
    expect(result.current.messages).toEqual([])
    expect(localStorage.getItem(STORAGE_KEY)).toBe('[]')
  })

  it('recovers from a non-JSON server response without losing the draft', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('<h1>Unavailable</h1>', { status: 503 }))
    const { result } = renderHook(useConversation)
    await act(async () => { await result.current.sendMessage('Still here?') })
    expect(result.current.draft).toBe('Still here?')
    expect(result.current.error).toContain("Joi couldn't finish")
    expect(result.current.messages).toEqual([])
  })
})
