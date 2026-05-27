import pytest
from pipeline.embed import embed
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
