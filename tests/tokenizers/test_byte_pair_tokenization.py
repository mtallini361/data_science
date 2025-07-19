import pytest

@pytest.mark.tokenizers
@pytest.mark.byte_pair_tokenization
def test_byte_pair_encoding():
    from data_science.tokenizers.byte_pair_tokenization import byte_pair_encoding
    
    corpus = ["low", "lowest", "newest", "widest"]
    num_merges = 3
    vocab = byte_pair_encoding(corpus, num_merges)
    assert 'est' in vocab