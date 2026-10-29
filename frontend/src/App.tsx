import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ArrowUp, ArrowUpRight, Info, LoaderCircle, MessageCircle, Plus, Sparkles } from 'lucide-react'
import About from './components/About'
import Conversation from './components/Conversation'
import Portrait from './components/Portrait'
import Welcome from './components/Welcome'
import { checkConnection, MAX_INPUT, type Connection } from './lib/conversation'
import { useConversation } from './lib/useConversation'

const iconButton = 'grid size-11 shrink-0 cursor-pointer place-items-center rounded-full border border-white/10 text-[#b6a7c5] transition hover:border-fuchsia-200/30 hover:bg-fuchsia-200/5 hover:text-[#eed5ef] focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-fuchsia-200 disabled:cursor-wait disabled:opacity-35'

export default function App() {
  const chat = useConversation()
  const [connection, setConnection] = useState<Connection>('checking')
  const [about, setAbout] = useState(false)
  const scroll = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const statusRequest = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    statusRequest.current = controller
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(5000)])
    checkConnection(signal).then(() => setConnection('ready')).catch(() => {
      if (!controller.signal.aborted) setConnection('offline')
    })
    return () => statusRequest.current?.abort()
  }, [])

  useEffect(() => {
    const element = scroll.current
    if (element) element.scrollTop = chat.messages.length ? element.scrollHeight : 0
  }, [chat.messages, chat.sending])

  useEffect(() => {
    const element = input.current
    if (element) {
      element.style.height = 'auto'
      element.style.height = `${Math.min(element.scrollHeight, 128)}px`
    }
  }, [chat.draft])

  async function send(text: string) {
    const sent = await chat.sendMessage(text)
    if (sent) setConnection('ready')
    else if (text.trim()) {
      statusRequest.current?.abort()
      const controller = new AbortController()
      statusRequest.current = controller
      checkConnection(AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]))
        .then(() => setConnection('ready'))
        .catch(() => { if (!controller.signal.aborted) setConnection('offline') })
    }
    requestAnimationFrame(() => input.current?.focus({ preventScroll: true }))
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void send(chat.draft)
  }

  function startNew() {
    chat.newConversation()
    requestAnimationFrame(() => input.current?.focus({ preventScroll: true }))
  }

  const status = connection === 'ready' ? 'Right here with you' : connection === 'offline' ? 'Waiting for Joi' : 'Connecting…'

  return (
    <>
      <main aria-label="Chat with Joi" className="grid h-dvh min-h-0 w-full overflow-hidden bg-[#0c0c13] font-sans text-[#eee5ef] selection:bg-fuchsia-300/25 sm:grid-cols-[68px_minmax(0,1fr)] lg:grid-cols-[76px_minmax(0,1fr)_37%]">
        <nav aria-label="Joi navigation" className="z-10 hidden min-h-0 flex-col items-center border-r border-white/7 bg-[#0b0b12] py-7 sm:flex">
          <div aria-hidden="true" className="mb-12 grid size-10 place-items-center text-[#e0b5e9]"><svg viewBox="0 0 40 40" fill="none" className="size-9"><path d="m20 2 5 13 13 5-13 5-5 13-5-13L2 20l13-5Z" stroke="currentColor" strokeWidth="1" /><path d="M20 12v16m-8-8h16" stroke="#91dfe5" strokeWidth="1" /></svg></div>
          <button aria-label="Focus conversation" onClick={() => input.current?.focus()} className="mb-4 grid size-11 place-items-center rounded-2xl border border-cyan-100/10 bg-cyan-100/5 text-[#a7d6de] outline-fuchsia-200 focus-visible:outline-2 focus-visible:outline-offset-4"><MessageCircle size={20} strokeWidth={1.5} /></button>
          <button aria-label="Start a new conversation" title="New conversation" disabled={chat.sending} onClick={startNew} className={`${iconButton} border-transparent`}><Plus size={22} strokeWidth={1.3} /></button>
          <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden"><p className="rotate-180 text-[9px] tracking-[.3em] whitespace-nowrap text-[#645b73] [writing-mode:vertical-rl]">A LITTLE WARMTH IN THE NEON</p></div>
          <button aria-label="About Joi" onClick={() => setAbout(true)} className={`${iconButton} border-transparent`}><Info size={19} strokeWidth={1.3} /></button>
          <span className="mt-4 font-display text-xs italic text-[#74657f]">2049</span>
        </nav>

        <section aria-label="Conversation with Joi" className="relative flex min-h-0 min-w-0 flex-col overflow-hidden bg-[radial-gradient(ellipse_at_80%_0%,#38213f45,transparent_55%)]">
          <header className="z-10 flex shrink-0 items-center justify-between gap-3 border-b border-white/6 px-5 py-5 sm:px-8 sm:py-6 xl:px-12">
            <div className="min-w-0">
              <div className="mb-1 flex items-center gap-2.5"><h1 className="font-display text-[30px] leading-none tracking-[-.05em]">Joi.</h1><Sparkles aria-hidden="true" size={13} strokeWidth={1.3} className="text-[#d8addd]" /></div>
              <p role="status" className="mt-2 flex min-w-0 items-center gap-2 text-xs text-[#93869f]">
                <span aria-hidden="true" className={`size-1 shrink-0 rounded-full ${connection === 'ready' ? 'bg-[#95c5b9] shadow-[0_0_8px_#95c5b940]' : 'bg-[#be927b]'}`} /><span className="truncate">{status}</span>
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2 sm:gap-3">
              <button aria-label="About Joi" onClick={() => setAbout(true)} className="grid size-11 place-items-center rounded-full text-[#b6a7c5] outline-fuchsia-200 transition hover:bg-white/5 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-4 sm:w-auto sm:px-2">
                <Info size={20} strokeWidth={1.3} className="sm:hidden" /><span className="hidden items-center gap-2 text-[13px] sm:flex">about Joi <ArrowUpRight aria-hidden="true" size={15} /></span>
              </button>
              <button aria-label="New conversation" disabled={chat.sending} onClick={startNew} className={`${iconButton} sm:flex sm:w-auto sm:items-center sm:justify-center sm:gap-2 sm:px-4`}><Plus size={18} strokeWidth={1.4} /><span className="hidden text-xs sm:block">new conversation</span></button>
            </div>
          </header>

          <div ref={scroll} role="log" aria-label="Conversation" aria-live="polite" aria-relevant="additions text" aria-busy={chat.sending} className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain px-5 [scrollbar-color:#44354f_transparent] [scrollbar-width:thin] sm:px-8 xl:px-12">
            {chat.messages.length ? <Conversation messages={chat.messages} sending={chat.sending} /> : <Welcome onStart={(text) => void send(text)} />}
          </div>

          <div className="z-10 shrink-0 px-4 pt-3 pb-[max(18px,env(safe-area-inset-bottom))] sm:px-8 sm:pb-6 xl:px-12">
            <div className="mx-auto max-w-2xl">
              {chat.error && <p role="alert" className="mb-3 rounded-xl border border-rose-300/10 bg-rose-300/5 px-4 py-3 text-sm leading-6 text-[#edbdd2]">{chat.error}</p>}
              <form onSubmit={submit} className="flex items-end gap-3 rounded-[22px] border border-[#a98ab2]/25 bg-[#1e1927] p-3 pl-5 shadow-[0_8px_50px_#00000020] transition focus-within:border-[#cca1da]/65 focus-within:shadow-[0_0_30px_#c585db0a]">
                <label htmlFor="message" className="sr-only">Your message to Joi</label>
                <textarea ref={input} id="message" rows={1} maxLength={MAX_INPUT} value={chat.draft} disabled={chat.sending} onChange={(event) => chat.setDraft(event.target.value)} onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault()
                    void send(chat.draft)
                  }
                }} placeholder="Tell me a little something…" className="my-1 min-h-7 max-w-full min-w-0 flex-1 resize-none bg-transparent text-lg leading-7 text-[#eee1ee] outline-none placeholder:text-[#82738d] disabled:opacity-60" />
                <button type="submit" aria-label="Send message" disabled={chat.sending || !chat.draft.trim()} className="grid size-11 shrink-0 place-items-center rounded-2xl bg-[#dcb8e5] text-[#302138] outline-fuchsia-100 transition hover:bg-[#ebcef1] focus-visible:outline-2 focus-visible:outline-offset-4 disabled:cursor-not-allowed disabled:opacity-40">
                  {chat.sending ? <LoaderCircle aria-hidden="true" size={21} className="animate-spin motion-reduce:animate-none" /> : <ArrowUp aria-hidden="true" size={22} strokeWidth={1.7} />}
                </button>
              </form>
              <div className="mt-3 flex items-center justify-between gap-3 px-1 text-[10px] tracking-wide text-[#796b87] sm:text-[11px]">
                <span className="flex items-center gap-2"><Sparkles aria-hidden="true" size={11} /><span>a little warmth in the neon.</span></span>
                <span aria-hidden="true" className="hidden text-[#62576e] xl:inline">enter to send · shift + enter for a new line</span>
                <span aria-label="Message length" className="shrink-0 tabular-nums">{chat.draft.length} / {MAX_INPUT}</span>
              </div>
            </div>
          </div>
        </section>
        <Portrait />
      </main>
      <About open={about} onClose={() => setAbout(false)} />
    </>
  )
}
