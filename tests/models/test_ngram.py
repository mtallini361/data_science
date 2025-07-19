import pytest

@pytest.mark.models
@pytest.mark.ngram
def test_train():
    
    from data_science.models.ngram import NGramModel
    
    corpus = [
        ["We", "still", "need", "to", "get", "rid", "of", "the", "trailer"],
        ["The", "grass", "is", "growing", "nicely"],
        ["I", "need", "to", "get", "measurements", "for", "the", "ac", "units"],
        ["You", "should", "really", "fix", "the", "smell", "downstairs"]
    ]
    
    model = NGramModel()
    
    model.train(corpus)
    prob = model.model.freqdist()[("the",)].freq('ac')
 
    assert (1/3) == prob