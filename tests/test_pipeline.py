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
            {"title": "SWE", "company": "Acme", "job_url": "http://a.com", "max_yoe": 3, "min_education": "BS", "responsibilities": "[]", "qualifications": "[]"},
            {"title": "MLE", "company": "Corp", "job_url": "http://b.com", "max_yoe": -1, "min_education": "", "responsibilities": "[]", "qualifications": "[]"},
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


def test_retrieve_uses_ef_400(mock_chroma):
    retrieve(_QUERY_VECTOR)
    call_kwargs = mock_chroma.query.call_args.kwargs
    assert call_kwargs["query_params"] == {"hnsw:ef": 400}


def test_retrieve_passes_filters(mock_chroma):
    filters = {"min_education": {"$eq": "BS"}}
    retrieve(_QUERY_VECTOR, filters=filters)
    call_kwargs = mock_chroma.query.call_args.kwargs
    assert call_kwargs["where"] == filters


def test_retrieve_no_filters(mock_chroma):
    retrieve(_QUERY_VECTOR)
    call_kwargs = mock_chroma.query.call_args.kwargs
    assert "where" not in call_kwargs


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
