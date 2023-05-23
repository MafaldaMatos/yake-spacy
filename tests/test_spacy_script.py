from src.spacy_script import eraseOverlapIntervals


def test_eraseOverlapIntervals():
    result = eraseOverlapIntervals([[1,2], [3,4], [1,3]])
    expected_result = [[1, 2], [3,4]]
    assert result == expected_result


def test_eraseOverlapIntervals_empty():
    result = eraseOverlapIntervals([])
    expected_result = []
    assert result == expected_result