export default function JobCard({ job }) {
  return (
    <div className="w-full h-full flex flex-col p-6 overflow-y-auto">
      <div className="flex justify-between items-start gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900">{job.title}</h2>
          <p className="text-slate-500 text-sm mt-1">{job.company}</p>
        </div>
        <span className="text-xs text-slate-400 whitespace-nowrap">
          Score: {job.score.toFixed(3)}
        </span>
      </div>
      <a
        href={job.url}
        target="_blank"
        rel="noreferrer"
        className="text-blue-600 text-sm mt-2 hover:underline break-all"
      >
        {job.url}
      </a>
      {job.explanation && (
        <p className="text-slate-700 text-sm mt-4 leading-relaxed">{job.explanation}</p>
      )}
      {job.corpus_warning && (
        <span className="mt-3 self-start text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded">
          Limited job data — explanation may be less accurate
        </span>
      )}
    </div>
  )
}
