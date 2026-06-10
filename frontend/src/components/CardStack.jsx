import { useState } from 'react'
import { motion, useAnimation } from 'framer-motion'
import JobCard from './JobCard'

const SWIPE_THRESHOLD = 100

export default function CardStack({ job, remaining, onSwipe }) {
  const controls = useAnimation()
  const [leaving, setLeaving] = useState(false)

  async function handleDragEnd(_, info) {
    if (leaving) return
    const x = info.offset.x
    if (Math.abs(x) < SWIPE_THRESHOLD) {
      controls.start({ x: 0, rotate: 0, transition: { type: 'spring', stiffness: 300 } })
      return
    }
    setLeaving(true)
    const direction = x > 0 ? 'right' : 'left'
    await controls.start({
      x: direction === 'right' ? 600 : -600,
      opacity: 0,
      transition: { duration: 0.3 },
    })
    onSwipe(direction)
    setLeaving(false)
  }

  async function handleButton(direction) {
    if (leaving) return
    setLeaving(true)
    await controls.start({
      x: direction === 'right' ? 600 : -600,
      opacity: 0,
      transition: { duration: 0.3 },
    })
    onSwipe(direction)
    setLeaving(false)
  }

  return (
    <div className="relative w-[360px] h-[520px]">
      <p className="absolute -top-8 left-0 right-0 text-center text-slate-400 text-sm">
        {remaining} job{remaining !== 1 ? 's' : ''} remaining
      </p>

      {remaining > 1 && (
        <div className="absolute inset-0 bg-white rounded-2xl shadow-md translate-y-2 scale-95" />
      )}

      <motion.div
        animate={controls}
        drag="x"
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.8}
        onDragEnd={handleDragEnd}
        className="absolute inset-0 bg-white rounded-2xl shadow-xl cursor-grab active:cursor-grabbing select-none"
        initial={{ scale: 0.95, opacity: 0 }}
        whileInView={{ scale: 1, opacity: 1 }}
      >
        <JobCard job={job} />
      </motion.div>

      <div className="absolute -bottom-16 left-0 right-0 flex justify-center gap-8">
        <button
          onClick={() => handleButton('left')}
          className="w-14 h-14 rounded-full bg-white shadow-md text-2xl flex items-center justify-center hover:bg-red-50 transition-colors"
          aria-label="Pass"
        >
          ✕
        </button>
        <button
          onClick={() => handleButton('right')}
          className="w-14 h-14 rounded-full bg-white shadow-md text-2xl flex items-center justify-center hover:bg-green-50 transition-colors"
          aria-label="Like"
        >
          ♥
        </button>
      </div>
    </div>
  )
}
