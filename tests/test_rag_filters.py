from src.rag.engine import document_matches_filters


def test_department_filter_matches_allowed_department():
    document = {
        "department": "IT",
        "allowed_departments": ["IT", "Engineering"],
        "file_type": "TXT",
    }

    assert document_matches_filters(document, "Engineering", "All") is True


def test_department_filter_rejects_unrelated_department():
    document = {
        "department": "IT",
        "allowed_departments": ["IT", "Engineering"],
        "file_type": "TXT",
    }

    assert document_matches_filters(document, "HR", "All") is False
