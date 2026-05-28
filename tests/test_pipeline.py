import pytest
from pipeline.embed import embed
from pipeline.retrieve import retrieve
from config import VOYAGE_MODEL


@pytest.fixture
def mock_voyage(mocker):
    mock_result = mocker.MagicMock()
    mock_result.embeddings = [[0.1, 0.2, 0.3]]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = mock_result
    mocker.patch("pipeline.embed.voyageai.Client", return_value=mock_client)
    return mock_client


def test_embed_returns_float_list(mock_voyage):
    assert embed("resume text") == [0.1, 0.2, 0.3]


def test_embed_uses_query_input_type(mock_voyage):
    embed("resume text")
    assert mock_voyage.embed.call_args.kwargs["input_type"] == "query"


def test_embed_uses_correct_model(mock_voyage):
    embed("resume text")
    assert mock_voyage.embed.call_args.kwargs["model"] == VOYAGE_MODEL


def test_embed_client_max_retries(mocker):
    mock_result = mocker.MagicMock()
    mock_result.embeddings = [[0.1, 0.2, 0.3]]
    mock_client = mocker.MagicMock()
    mock_client.embed.return_value = mock_result
    ctor = mocker.patch("pipeline.embed.voyageai.Client", return_value=mock_client)
    embed("resume text")
    assert ctor.call_args.kwargs["max_retries"] == 3


_QUERY_VECTOR = [0.1, 0.2, 0.3]
_CHROMA_RESULT = {
    "ids": [["job1", "job2"]],
    "distances": [[0.1, 0.2]],
    "metadatas": [
        [
            {"title": "SWE", "company": "Acme", "job_url": "http://a.com", "max_yoe": 3, "min_education": "BS", "responsibilities": "[]", "qualifications": "[]", "is_internship": 0},
            {"title": "MLE", "company": "Corp", "job_url": "http://b.com", "max_yoe": -1, "min_education": "", "responsibilities": "[]", "qualifications": "[]", "is_internship": 0},
        ]
    ],
}


@pytest.fixture
def mock_chroma(mocker):
    mock_collection = mocker.MagicMock()
    mock_collection.query.return_value = _CHROMA_RESULT
    mock_client = mocker.MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mocker.patch("pipeline.retrieve.chromadb.PersistentClient", return_value=mock_client)
    return mock_collection


def test_retrieve_returns_list_of_dicts(mock_chroma):
    results = retrieve(_QUERY_VECTOR)
    assert len(results) == 2
    assert results[0]["job_id"] == "job1"
    assert results[0]["distance"] == 0.1
    assert results[0]["title"] == "SWE"
    assert results[1]["job_id"] == "job2"


def test_retrieve_passes_filters(mock_chroma):
    filters = {"min_education": {"$eq": "BS"}}
    retrieve(_QUERY_VECTOR, filters=filters)
    call_kwargs = mock_chroma.query.call_args.kwargs
    assert call_kwargs["where"] == filters


def test_retrieve_no_filters(mock_chroma):
    retrieve(_QUERY_VECTOR)
    call_kwargs = mock_chroma.query.call_args.kwargs
    assert "where" not in call_kwargs


from app.routes import _build_filters


def test_build_filters_internship_true():
    f = _build_filters(None, None, True)
    assert f == {"is_internship": {"$eq": 1}}


def test_build_filters_internship_false():
    f = _build_filters(None, None, False)
    assert f == {"is_internship": {"$eq": 0}}


def test_build_filters_internship_none():
    f = _build_filters(None, None, None)
    assert f is None


def test_build_filters_internship_combined_with_yoe():
    f = _build_filters(None, 3, False)
    assert f == {"$and": [
        {"$or": [{"max_yoe": {"$eq": -1}}, {"max_yoe": {"$lte": 3}}]},
        {"is_internship": {"$eq": 0}},
    ]}


def test_build_filters_internship_combined_with_education():
    f = _build_filters("BS", None, True)
    assert f == {"$and": [
        {"min_education": {"$in": ["", "BS"]}},
        {"is_internship": {"$eq": 1}},
    ]}


from pipeline.rerank import rerank, _format_job
from config import COHERE_RERANK_MODEL

_CANDIDATES = [
    {
        "job_id": "job1",
        "title": "ML Engineer",
        "company": "Acme",
        "job_url": "http://a.com",
        "responsibilities": '["Build models", "Deploy pipelines"]',
        "qualifications": '["Python", "PyTorch"]',
    },
    {
        "job_id": "job2",
        "title": "Data Scientist",
        "company": "Corp",
        "job_url": "http://b.com",
        "responsibilities": '["Analyze data"]',
        "qualifications": '["SQL", "Statistics"]',
    },
]


@pytest.fixture
def mock_cohere(mocker):
    r0 = mocker.MagicMock()
    r0.index = 1
    r0.relevance_score = 0.9

    r1 = mocker.MagicMock()
    r1.index = 0
    r1.relevance_score = 0.4

    mock_response = mocker.MagicMock()
    mock_response.results = [r0, r1]

    mock_client = mocker.MagicMock()
    mock_client.rerank.return_value = mock_response

    mocker.patch("pipeline.rerank.cohere.ClientV2", return_value=mock_client)
    return mock_client


def test_rerank_returns_top_k(mock_cohere):
    results = rerank("resume text", _CANDIDATES, top_k=2)
    assert len(results) == 2


def test_rerank_score_field(mock_cohere):
    results = rerank("resume text", _CANDIDATES, top_k=2)
    for r in results:
        assert isinstance(r["score"], float)


def test_rerank_order(mock_cohere):
    results = rerank("resume text", _CANDIDATES, top_k=2)
    assert results[0]["score"] >= results[1]["score"]


def test_rerank_index_mapping(mock_cohere):
    results = rerank("resume text", _CANDIDATES, top_k=2)
    assert results[0]["job_id"] == "job2"  # index=1 ranked first
    assert results[1]["job_id"] == "job1"  # index=0 ranked second


def test_rerank_uses_correct_model(mock_cohere):
    rerank("resume text", _CANDIDATES, top_k=2)
    assert mock_cohere.rerank.call_args.kwargs["model"] == COHERE_RERANK_MODEL


def test_rerank_document_format():
    doc = _format_job(_CANDIDATES[0])
    assert "ML Engineer at Acme" in doc
    assert "Responsibilities:" in doc
    assert "- Build models" in doc
    assert "Qualifications:" in doc
    assert "- Python" in doc


def test_rerank_missing_sections():
    candidate = {"title": "SWE", "company": "X", "responsibilities": "[]", "qualifications": "[]"}
    doc = _format_job(candidate)
    assert "Responsibilities:" not in doc
    assert "Qualifications:" not in doc


from pipeline.generate import explain, _CORPUS_WARNING

_JOB = {
    "title": "ML Engineer",
    "company": "Acme",
    "responsibilities": '["Build models", "Deploy pipelines"]',
    "qualifications": '["Python", "PyTorch"]',
}
_RESUME = "Experienced ML engineer with 3 years of Python and PyTorch experience."


@pytest.fixture
def mock_chain(mocker):
    mock = mocker.MagicMock()
    mocker.patch("pipeline.generate.ChatOpenAI", return_value=mocker.MagicMock())
    mocker.patch("pipeline.generate.ChatPromptTemplate.from_messages", return_value=mocker.MagicMock())
    chain_mock = mocker.MagicMock()
    chain_mock.invoke.return_value = "Strong fit: candidate has Python and PyTorch experience."
    mocker.patch("pipeline.generate.ChatPromptTemplate.from_messages").__or__ = mocker.MagicMock(return_value=chain_mock)

    # Patch at the chain construction level via __or__ chaining
    prompt_mock = mocker.MagicMock()
    prompt_mock.__or__ = mocker.MagicMock(return_value=chain_mock)
    mocker.patch("pipeline.generate.ChatPromptTemplate.from_messages", return_value=prompt_mock)
    return chain_mock


def _patch_chain(mocker, return_value):
    chain_mock = mocker.MagicMock()
    chain_mock.invoke.return_value = return_value
    prompt_mock = mocker.MagicMock()
    llm_mock = mocker.MagicMock()
    llm_mock.__or__ = mocker.MagicMock(return_value=chain_mock)
    prompt_mock.__or__ = mocker.MagicMock(return_value=llm_mock)
    mocker.patch("pipeline.generate.ChatOpenAI", return_value=mocker.MagicMock())
    mocker.patch("pipeline.generate.ChatPromptTemplate.from_messages", return_value=prompt_mock)
    mocker.patch("pipeline.generate.StrOutputParser", return_value=mocker.MagicMock())
    return chain_mock


def test_explain_good_fit_returns_explanation(mocker):
    _patch_chain(mocker, "Strong fit: candidate has Python and PyTorch.")
    explanation, warning = explain(_RESUME, _JOB)
    assert explanation == "Strong fit: candidate has Python and PyTorch."
    assert warning is False


def test_explain_good_fit_no_corpus_warning(mocker):
    _patch_chain(mocker, "Candidate matches all required skills.")
    _, warning = explain(_RESUME, _JOB)
    assert warning is False


def test_explain_llm_returns_corpus_warning_string(mocker):
    _patch_chain(mocker, _CORPUS_WARNING)
    explanation, warning = explain(_RESUME, _JOB)
    assert explanation == _CORPUS_WARNING
    assert warning is True


def test_explain_llm_returns_empty_string(mocker):
    _patch_chain(mocker, "")
    explanation, warning = explain(_RESUME, _JOB)
    assert explanation == _CORPUS_WARNING
    assert warning is True


def test_explain_llm_returns_whitespace_only(mocker):
    _patch_chain(mocker, "   \n  ")
    explanation, warning = explain(_RESUME, _JOB)
    assert explanation == _CORPUS_WARNING
    assert warning is True


def test_explain_llm_call_raises(mocker):
    chain_mock = mocker.MagicMock()
    chain_mock.invoke.side_effect = Exception("API timeout")
    prompt_mock = mocker.MagicMock()
    llm_mock = mocker.MagicMock()
    llm_mock.__or__ = mocker.MagicMock(return_value=chain_mock)
    prompt_mock.__or__ = mocker.MagicMock(return_value=llm_mock)
    mocker.patch("pipeline.generate.ChatOpenAI", return_value=mocker.MagicMock())
    mocker.patch("pipeline.generate.ChatPromptTemplate.from_messages", return_value=prompt_mock)
    mocker.patch("pipeline.generate.StrOutputParser", return_value=mocker.MagicMock())
    explanation, warning = explain(_RESUME, _JOB)
    assert explanation == _CORPUS_WARNING
    assert warning is True


def test_explain_strips_whitespace_from_output(mocker):
    _patch_chain(mocker, "  Great fit for the role.  ")
    explanation, _ = explain(_RESUME, _JOB)
    assert explanation == "Great fit for the role."


def test_explain_passes_title_and_company(mocker):
    chain_mock = _patch_chain(mocker, "Good fit.")
    explain(_RESUME, _JOB)
    call_kwargs = chain_mock.invoke.call_args[0][0]
    assert call_kwargs["title"] == "ML Engineer"
    assert call_kwargs["company"] == "Acme"


def test_explain_passes_resume(mocker):
    chain_mock = _patch_chain(mocker, "Good fit.")
    explain(_RESUME, _JOB)
    call_kwargs = chain_mock.invoke.call_args[0][0]
    assert call_kwargs["resume"] == _RESUME


def test_explain_formats_responsibilities_as_bullets(mocker):
    chain_mock = _patch_chain(mocker, "Good fit.")
    explain(_RESUME, _JOB)
    call_kwargs = chain_mock.invoke.call_args[0][0]
    assert "- Build models" in call_kwargs["responsibilities"]
    assert "- Deploy pipelines" in call_kwargs["responsibilities"]


def test_explain_formats_qualifications_as_bullets(mocker):
    chain_mock = _patch_chain(mocker, "Good fit.")
    explain(_RESUME, _JOB)
    call_kwargs = chain_mock.invoke.call_args[0][0]
    assert "- Python" in call_kwargs["qualifications"]
    assert "- PyTorch" in call_kwargs["qualifications"]


def test_explain_invalid_json_responsibilities(mocker):
    _patch_chain(mocker, "Good fit.")
    job = dict(_JOB, responsibilities="not-json")
    explanation, warning = explain(_RESUME, job)
    assert warning is False  # should not crash; empty lists used as fallback


def test_explain_invalid_json_qualifications(mocker):
    _patch_chain(mocker, "Good fit.")
    job = dict(_JOB, qualifications=None)
    explanation, warning = explain(_RESUME, job)
    assert warning is False


def test_explain_missing_job_fields(mocker):
    _patch_chain(mocker, "Good fit.")
    explanation, warning = explain(_RESUME, {})
    assert warning is False  # graceful fallback to empty strings
