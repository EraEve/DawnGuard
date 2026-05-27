import numpy as np

from utils import semantic_embedding, softmax


def test_softmax_sum_is_one():
    values = softmax([1.0, 2.0, 3.0])
    assert np.isclose(values.sum(), 1.0)


def test_semantic_embedding_keeps_legacy_dimension():
    vector = semantic_embedding("medical retrieval", dim=128)
    assert vector.shape == (128,)
