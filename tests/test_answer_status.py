from src.core.answer_status import classify_answer_status


def test_classifies_grounded_refusal_with_sources_as_not_found():
    answer = "The provided sources do not contain information about the annual leave approval process."

    status = classify_answer_status(answer, ["data/simulated/HR_Policy.txt"])

    assert status == "not_found"


def test_classifies_supported_answer_with_sources_as_success():
    answer = "Employees must use at least 12 characters."

    status = classify_answer_status(answer, ["data/simulated/IT_Policy_Password.txt"])

    assert status == "success"
