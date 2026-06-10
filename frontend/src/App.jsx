import { useState } from 'react'
import ResumeForm from './components/ResumeForm'
import CardStack from './components/CardStack'
import LikedJobs from './components/LikedJobs'
import { matchJobs } from './api'

export default function App() {
  const [phase, setPhase] = useState('form')
  const [jobs, setJobs] = useState([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [liked, setLiked] = useState([])
  const [error, setError] = useState(null)

  async function handleSubmit(formData) {
    setPhase('loading')
    setError(null)
    try {
      const results = await matchJobs(formData)
      // Only swipe the explained subset — jobs beyond explain_top_k have no explanation
      const deck = results.slice(0, formData.explain_top_k)
      setJobs(deck)
      setCurrentIndex(0)
      setLiked([])
      setPhase(deck.length === 0 ? 'done' : 'swiping')
    } catch (err) {
      setError(err.message)
      setPhase('form')
    }
  }

  function handleSwipe(direction) {
    if (direction === 'right') {
      setLiked((prev) => [...prev, jobs[currentIndex]])
    }
    const next = currentIndex + 1
    if (next >= jobs.length) {
      setPhase('done')
    } else {
      setCurrentIndex(next)
    }
  }

  function handleRestart() {
    setPhase('form')
    setJobs([])
    setCurrentIndex(0)
    setLiked([])
    setError(null)
  }

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col items-center justify-center p-4">
      {phase === 'form' && (
        <ResumeForm onSubmit={handleSubmit} error={error} isLoading={false} />
      )}
      {phase === 'loading' && (
        <p className="text-slate-500 text-lg">Matching your resume…</p>
      )}
      {phase === 'swiping' && (
        <CardStack
          key={jobs[currentIndex].job_id}
          job={jobs[currentIndex]}
          remaining={jobs.length - currentIndex}
          onSwipe={handleSwipe}
        />
      )}
      {phase === 'done' && (
        <LikedJobs liked={liked} onRestart={handleRestart} />
      )}
    </div>
  )
}
