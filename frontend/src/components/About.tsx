import { useEffect, useRef } from 'react'
import { ArrowUpRight, X } from 'lucide-react'
import portrait from '../../../assets/joi-portrait.png'

export default function About({ open, onClose }: { open: boolean; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const element = dialog.current
    if (open && element && !element.open) element.showModal()
    else if (!open && element?.open) element.close()
  }, [open])

  return (
    <dialog ref={dialog} aria-labelledby="about-title" onCancel={onClose} onClose={onClose} onClick={(event) => { if (event.target === event.currentTarget) onClose() }} className="fixed inset-0 m-auto max-h-[calc(100dvh_-_40px)] w-[min(440px,calc(100%_-_40px))] overflow-auto rounded-3xl border border-fuchsia-200/20 bg-[#19131f] p-7 text-[#eee2ee] shadow-2xl backdrop:bg-[#08070e]/85 backdrop:backdrop-blur-md sm:p-9">
      <button aria-label="Close About Joi" onClick={onClose} className="absolute top-3 right-3 grid size-11 place-items-center rounded-full text-[#b5a2bf] transition hover:bg-white/5 hover:text-white focus-visible:outline-2 focus-visible:outline-fuchsia-200"><X size={21} /></button>
      <img src={portrait} alt="Joi" className="mb-8 size-22 rounded-full object-cover ring-1 ring-fuchsia-200/30 shadow-[0_0_40px_#d665d320]" />
      <p className="text-[10px] tracking-[.24em] text-[#d5aa94]">BLADE RUNNER 2049</p>
      <h2 id="about-title" className="mt-5 font-display text-6xl tracking-[-.04em]">Joi.</h2>
      <p className="mt-6 text-lg leading-8 text-[#c8bacf]">Your personal companion, inspired by Joi from <i>Blade Runner 2049</i>.</p>
      <p className="mt-5 font-display text-2xl italic text-[#e3b9e1]">A little warmth in the neon.</p>
      <div className="mt-8 flex items-center justify-between gap-4 border-t border-white/10 pt-5 text-xs leading-6 text-[#a38aad]">
        <span>a personal project · made from scratch</span><ArrowUpRight aria-hidden="true" size={16} className="shrink-0" />
      </div>
    </dialog>
  )
}
