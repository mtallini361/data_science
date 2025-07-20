import pytest

@pytest.mark.distances
@pytest.mark.minimal_edit_distance
def test_minimal_edit_distance_initialization():
    from data_science.distances.minimal_edit_distance import MinimalEditDistance
    
    med = MinimalEditDistance()
    
    dist = med.measure("test", "text")
    assert dist == 2, "Expected minimal edit distance to be 1 for 'test' and 'text'"
