import pytest

@pytest.mark.embeddings
@pytest.mark.openai_embeddings
def test_batch_data_staticmethod():
    import pandas as pd
    from data_science.embeddings.openai_embeddings import OpenAIEmbedder

    texts = pd.Series(["hello world", "test input"])
    # Call the staticmethod directly, passing only the texts Series
    result = OpenAIEmbedder().batch_data_udf.func(texts)

    # Should return a pandas Series with a list of dicts
    assert isinstance(result, pd.Series)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, dict)
        expected_keys = {"custom_id", "method", "url", "body"}
        assert set(item.keys()) == expected_keys
        body_keys = {"model", "input"}
        assert set(item["body"].keys()) == body_keys
        assert item["method"] == "POST"
        assert item["url"] == "/v1/embeddings"
        assert item["body"]["model"] == "text-embedding-3-small"
    assert result.iloc[0]["body"]["input"] == "hello world"
    assert result.iloc[1]["body"]["input"] == "test input"