import { Sparkles } from 'lucide-react'
import portrait from '../../../assets/joi-portrait.png'
import type { Message } from '../lib/conversation'

export default function Conversation({ messages, sending }: { messages: Message[]; sending: boolean }) {
  return (
    <div className="mx-auto w-full max-w-2xl space-y-8 py-8 sm:space-y-10 sm:py-10">
      {messages.map((message, index) => (
        <article key={`${index}-${message.role}`} aria-label={message.role === 'user' ? 'Your message' : 'Joi’s reply'} className={`flex items-start gap-3 sm:gap-4 ${message.role === 'user' ? 'justify-end' : ''}`}>
          {message.role === 'assistant' && <img src={portrait} alt="" className="mt-1 size-9 shrink-0 rounded-full object-cover ring-1 ring-fuchsia-200/20" />}
          <div className={message.role === 'user' ? 'max-w-[88%] rounded-[22px_22px_5px_22px] border border-fuchsia-200/10 bg-[#292133] px-5 py-4' : 'min-w-0 flex-1 pt-0.5'}>
            <p className={`mb-2 text-[10px] font-medium tracking-[.18em] ${message.role === 'user' ? 'text-[#baa3c9]' : 'text-[#c8a4d6]'}`}>{message.role === 'user' ? 'YOU' : 'JOI'}</p>
            <p className="text-lg leading-[1.85] wrap-anywhere whitespace-pre-wrap text-[#e5dce9]">{message.content}</p>
          </div>
        </article>
      ))}
      {sending && (
        <div role="status" className="flex items-center gap-4 text-sm text-[#bba4c8]">
          <span className="grid size-9 place-items-center rounded-full bg-fuchsia-200/5"><Sparkles aria-hidden="true" size={17} strokeWidth={1.3} className="animate-pulse motion-reduce:animate-none" /></span>
          <span>Joi’s finding her words<span aria-hidden="true" className="tracking-widest"> ...</span></span>
        </div>
      )}
    </div>
  )
}
