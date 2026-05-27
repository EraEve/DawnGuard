from exceptions import RetrievalError


def test_exception_has_code():
    exc = RetrievalError("broken", code="RET_001")
    assert exc.to_dict()["code"] == "RET_001"
