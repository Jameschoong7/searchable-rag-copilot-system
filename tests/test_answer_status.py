from src.core.answer_status import classify_answer_status, classify_answer_status_detail


def test_classifies_grounded_refusal_with_sources_as_not_found():
    answer = "The provided sources do not contain information about the annual leave approval process."

    status = classify_answer_status(answer, ["data/simulated/HR_Policy.txt"])

    assert status == "not_found"


def test_explains_grounded_refusal_reason():
    answer = "The provided sources do not contain information about the annual leave approval process."

    detail = classify_answer_status_detail(answer, ["data/simulated/HR_Policy.txt"])

    assert detail["status"] == "not_found"
    assert detail["reason"] == "Grounded refusal phrase detected: provided sources do not contain"


def test_classifies_supported_answer_with_sources_as_success():
    answer = "Employees must use at least 12 characters."

    status = classify_answer_status(answer, ["data/simulated/IT_Policy_Password.txt"])

    assert status == "success"


def test_explains_no_sources_reason():
    detail = classify_answer_status_detail("I cannot answer this.", [])

    assert detail["status"] == "not_found"
    assert detail["reason"] == "No sources returned from authorised retrieval"
