import { useState } from 'react'

const SAMPLE_RESUME = `Education: B.S. Statistics & Data Science, Applied Mathematics — UC Santa Barbara (2020–2025)

Experience:

Data Science Research Intern: anomaly detection, unsupervised ML, neural network embeddings, clustering on high-dimensional simulation data using Python, PyTorch, Scikit-learn, Pandas, NumPy

Data Scientist Intern: LLM-powered guidance agent, GenAI architecture, OpenAI API, prompt engineering, NLP pipeline

Data Science Capstone Intern: audio classification CNN (PyTorch)

Full-Stack AI Engineer on Resume-Job Matching System: hybrid semantic retrieval, dense embeddings, cross-encoder reranking, FastAPI, LangChain, DeepSeek V3, ChromaDB, VoyageAI, Cohere, multi-source job ingestion pipeline, SQLite schema design, GitHub Actions automation

Skills: Python, SQL, PyTorch, Scikit-learn, Pandas, NumPy, LangChain, FastAPI, OpenAI API, ChromaDB, VoyageAI, Cohere, GitHub Actions`

export default function ResumeForm({ onSubmit, error, isLoading }) {
  const [resume, setResume] = useState(SAMPLE_RESUME)
  const [topK, setTopK] = useState(10)
  const [explainTopK, setExplainTopK] = useState(3)
  const [education, setEducation] = useState('')
  const [yearsOfExperience, setYearsOfExperience] = useState('')
  const [internship, setInternship] = useState('')
  const [excludeCompanies, setExcludeCompanies] = useState('')
  const [rerankMethod, setRerankMethod] = useState('cohere')

  function handleSubmit(e) {
    e.preventDefault()
    const payload = {
      resume,
      top_k: parseInt(topK) || 10,
      explain_top_k: parseInt(explainTopK) || 3,
      rerank_method: rerankMethod,
    }
    if (education) payload.education = education
    if (yearsOfExperience !== '') payload.years_of_experience = parseInt(yearsOfExperience)
    if (internship !== '') payload.internship = internship === 'true'
    const companies = excludeCompanies.split('\n').map((s) => s.trim()).filter(Boolean)
    if (companies.length > 0) payload.exclude_companies = companies
    onSubmit(payload)
  }

  const labelCls = 'text-xs font-semibold text-slate-600 flex flex-col gap-1'
  const inputCls = 'border border-slate-300 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400'
  const selectCls = inputCls

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white rounded-xl shadow-sm p-6 w-full max-w-2xl flex flex-col gap-4"
    >
      <h1 className="text-lg font-bold text-slate-900">Resume Job Matcher</h1>

      <label className={labelCls}>
        Resume
        <textarea
          value={resume}
          onChange={(e) => setResume(e.target.value)}
          required
          rows={8}
          className={`${inputCls} resize-y min-h-[160px]`}
          placeholder="Paste your resume here..."
        />
      </label>

      <div className="flex flex-wrap gap-4">
        <label className={labelCls}>
          Min Education
          <select value={education} onChange={(e) => setEducation(e.target.value)} className={selectCls}>
            <option value="">None</option>
            <option value="BS">BS</option>
            <option value="MS">MS</option>
            <option value="PhD">PhD</option>
          </select>
        </label>

        <label className={labelCls}>
          Years of Experience
          <input
            type="number"
            min="0"
            value={yearsOfExperience}
            onChange={(e) => setYearsOfExperience(e.target.value)}
            placeholder="Any"
            className={`${inputCls} w-28`}
          />
        </label>

        <label className={labelCls}>
          Role Type
          <select value={internship} onChange={(e) => setInternship(e.target.value)} className={selectCls}>
            <option value="">No preference</option>
            <option value="false">Full-time only</option>
            <option value="true">Internships only</option>
          </select>
        </label>

        <label className={labelCls}>
          Top K
          <input
            type="number"
            min="1"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
            className={`${inputCls} w-20`}
          />
        </label>

        <label className={labelCls}>
          Explain Top K
          <input
            type="number"
            min="1"
            value={explainTopK}
            onChange={(e) => setExplainTopK(e.target.value)}
            className={`${inputCls} w-20`}
          />
        </label>

        <label className={labelCls}>
          Rerank Method
          <select value={rerankMethod} onChange={(e) => setRerankMethod(e.target.value)} className={selectCls}>
            <option value="cohere">Cohere (fast)</option>
            <option value="llm">LLM sliding window</option>
          </select>
        </label>
      </div>

      <label className={labelCls}>
        Exclude Companies
        <textarea
          value={excludeCompanies}
          onChange={(e) => setExcludeCompanies(e.target.value)}
          rows={3}
          className={`${inputCls} resize-y`}
          placeholder={'One company per line\ne.g.\nGoogle LLC\nMeta Platforms, Inc.'}
        />
      </label>

      {error && <p className="text-red-600 text-sm">{error}</p>}

      <button
        type="submit"
        disabled={isLoading}
        className="self-start px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? 'Matching…' : 'Find Matches'}
      </button>
    </form>
  )
}
