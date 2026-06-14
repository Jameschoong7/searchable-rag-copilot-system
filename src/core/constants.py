SYSTEM_ADMIN_ROLE = "System Admin"
PROJECT_MANAGER_ROLE = "Project Manager"
GENERAL_EMPLOYEE_ROLE = "General Employee"

ROLE_OPTIONS = [
    SYSTEM_ADMIN_ROLE,
    PROJECT_MANAGER_ROLE,
    GENERAL_EMPLOYEE_ROLE,
]

FILTER_ALL = "All"

DEPARTMENT_OPTIONS = [
    "IT",
    "Engineering",
    "HR",
    "Security",
    "Operations",
]

ROLE_HIERARCHY = {
    SYSTEM_ADMIN_ROLE: [SYSTEM_ADMIN_ROLE],
    PROJECT_MANAGER_ROLE: [SYSTEM_ADMIN_ROLE, PROJECT_MANAGER_ROLE],
    GENERAL_EMPLOYEE_ROLE: [
        SYSTEM_ADMIN_ROLE,
        PROJECT_MANAGER_ROLE,
        GENERAL_EMPLOYEE_ROLE,
    ],
}


def expand_allowed_roles(selected_roles: list[str]) -> list[str]:
    """Expand selected minimum roles into the actual roles allowed to access a document."""
    expanded_roles = []

    for selected_role in selected_roles:
        for role in ROLE_HIERARCHY.get(selected_role, []):
            if role not in expanded_roles:
                expanded_roles.append(role)

    return expanded_roles


def expand_allowed_departments(selected_departments: list[str]) -> list[str]:
    """Expand All department selection into every supported simulated department."""
    if FILTER_ALL in selected_departments:
        return DEPARTMENT_OPTIONS

    return selected_departments