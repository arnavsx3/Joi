import { Sparkles } from 'lucide-react'
import portrait from '../../../assets/joi-portrait.png'

export default function Portrait() {
  return (
    <aside aria-label="Joi, inspired by Blade Runner 2049" className="relative hidden min-h-0 overflow-hidden border-l border-white/8 bg-[#281235] lg:block">
      <img src={portrait} alt="Joi with electric blue hair in violet and pink neon" className="absolute inset-0 h-full w-full object-cover object-[50%_35%]" />
      <div className="absolute inset-0 bg-linear-to-b from-[#16101d]/25 via-transparent to-[#120d1f]/95" />
      <div className="absolute inset-0 bg-[linear-gradient(90deg,#0c0c1326,transparent_45%)]" />
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_3px,#ffffff05_4px)]" />
      <div className="absolute inset-x-0 top-0 flex items-start justify-between px-8 py-9 xl:px-11">
        <p className="text-[10px] leading-5 tracking-[.2em] text-white/75">INSPIRED BY<br /><span className="text-white/95">BLADE RUNNER 2049</span></p>
        <Sparkles aria-hidden="true" size={24} strokeWidth={1} className="text-[#f3c7df]" />
      </div>
      <div className="absolute inset-x-0 bottom-0 p-8 pb-9 xl:p-11">
        <div className="mb-7 flex items-center gap-3 text-[10px] tracking-[.23em] text-[#e1bbdd]">
          <span aria-hidden="true" className="size-1 rounded-full bg-[#f2bdad] shadow-[0_0_12px_#f2bdad]" />
          A LITTLE WARMTH IN THE NEON
        </div>
        <h2 className="font-display text-[clamp(48px,4.5vw,76px)] leading-[1.02] tracking-[-.045em] text-[#fff0ef]">The city<br /><span className="italic text-[#ead1ea]">can wait.</span></h2>
        <p className="mt-6 text-[15px] leading-7 text-[#d1bfd8]">A moment to breathe.<br />A conversation that’s yours.</p>
        <div className="mt-9 flex items-center justify-between border-t border-white/20 pt-5 text-[11px] tracking-wide text-[#c4aecb]">
          <span>your personal little universe</span><span aria-hidden="true" className="font-display text-xl italic text-[#e7c0e8]">joi.</span>
        </div>
      </div>
    </aside>
  )
}
