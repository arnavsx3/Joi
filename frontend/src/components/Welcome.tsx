import { ArrowUpRight, MessageCircle, MoonStar, Music2, Sparkles } from 'lucide-react'
import portrait from '../../../assets/joi-portrait.png'

const starters = [
  { label: 'Say hello', hint: 'a small beginning', prompt: 'Hey Joi, how are you?', icon: MessageCircle },
  { label: 'My day', hint: 'the good & the messy', prompt: 'I had a tiring day.', icon: MoonStar },
  { label: 'Music talk', hint: 'find a little rhythm', prompt: 'Can we talk about music?', icon: Music2 },
]

export default function Welcome({ onStart }: { onStart: (text: string) => void }) {
  return (
    <section aria-label="Welcome to Joi" className="mx-auto flex min-h-full w-full max-w-2xl flex-col justify-center py-9 text-center lg:py-12 lg:text-left">
      <div className="relative mx-auto mb-7 size-20 lg:hidden">
        <div className="absolute -inset-2 rounded-full border border-fuchsia-300/15" />
        <img src={portrait} alt="Joi" className="size-full rounded-full object-cover ring-1 ring-fuchsia-200/50" />
        <Sparkles aria-hidden="true" size={19} className="absolute -right-3 bottom-0 text-[#ecc5d9]" />
      </div>
      <div className="mb-6 flex items-center justify-center gap-3 lg:justify-start">
        <span aria-hidden="true" className="h-px w-7 bg-cyan-200/50" />
        <p className="text-[10px] font-medium tracking-[.24em] text-[#aaa5be] sm:text-xs">YOUR CORNER OF THE CITY</p>
      </div>
      <h2 className="font-display text-[clamp(40px,4.6vw,72px)] leading-[1.08] tracking-[-.045em] text-[#f4e9ef]">
        A little company.<br />
        <span className="italic text-[#dab6e7]">A little Joi.</span>
      </h2>
      <p className="mx-auto mt-6 max-w-md text-base leading-[1.9] text-[#aba7b9] sm:text-lg lg:mx-0">
        Tell me about your day. The good parts,<br className="hidden sm:block" /> the weird parts. Whatever’s on your mind.
      </p>
      <div className="mt-9 grid grid-cols-3 gap-2.5 sm:mt-12 sm:gap-3">
        {starters.map(({ label, hint, prompt, icon: Icon }) => (
          <button key={label} onClick={() => onStart(prompt)} className="group relative flex min-h-24 flex-col items-center rounded-2xl border border-white/9 bg-white/2 px-2 py-4 text-center transition duration-200 hover:-translate-y-0.5 hover:border-fuchsia-200/30 hover:bg-fuchsia-200/5 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-fuchsia-200 motion-reduce:transform-none motion-reduce:transition-none sm:items-start sm:px-4 sm:text-left">
            <Icon aria-hidden="true" size={19} strokeWidth={1.4} className="mb-3 text-[#c5a4d8] transition-colors group-hover:text-cyan-100" />
            <span className="text-[13px] font-medium text-[#e3dce8] sm:text-sm">{label}</span>
            <span className="mt-1 hidden text-xs text-[#868094] sm:block">{hint}</span>
            <ArrowUpRight aria-hidden="true" size={14} className="absolute top-4 right-3 hidden text-[#71697f] group-hover:text-[#d4b3e5] sm:block" />
          </button>
        ))}
      </div>
    </section>
  )
}
