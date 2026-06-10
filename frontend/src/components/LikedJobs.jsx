export default function LikedJobs({ liked, onRestart }) {
  return (
    <div className="max-w-xl w-full">
      <h2 className="text-2xl font-bold text-slate-900 mb-1">
        {liked.length > 0
          ? `You liked ${liked.length} job${liked.length !== 1 ? 's' : ''}`
          : 'No jobs liked'}
      </h2>
      <p className="text-slate-500 text-sm mb-6">
        {liked.length > 0
          ? 'Here are the jobs you swiped right on.'
          : 'You passed on all jobs.'}
      </p>

      <div className="flex flex-col gap-3 mb-8">
        {liked.map((job) => (
          <div
            key={job.job_id}
            className="bg-white rounded-xl p-4 shadow-sm border-l-4 border-blue-500"
          >
            <div className="flex justify-between items-start gap-2">
              <div>
                <p className="font-semibold text-slate-900">{job.title}</p>
                <p className="text-slate-500 text-sm">{job.company}</p>
              </div>
              <a
                href={job.url}
                target="_blank"
                rel="noreferrer"
                className="text-blue-600 text-sm hover:underline whitespace-nowrap"
              >
                View Job
              </a>
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={onRestart}
        className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
      >
        Start Over
      </button>
    </div>
  )
}
