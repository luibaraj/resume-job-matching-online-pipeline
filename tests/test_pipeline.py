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
