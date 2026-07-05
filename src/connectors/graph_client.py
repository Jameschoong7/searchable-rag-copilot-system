import requests

from src.connectors.graph_auth import get_graph_access_token
from src.core.config import read_app_config
from urllib.parse import quote


GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


def get_graph_headers() -> dict:
    """Build Microsoft Graph request headers from the configured delegated token."""
    app_config = read_app_config()

    if not app_config.graph_connector_enabled:
        raise RuntimeError("Graph connector is disabled. Set GRAPH_CONNECTOR_ENABLED=true.")

    access_token = get_graph_access_token()

    return {
        "Authorization": f"Bearer {access_token}",
    }


def graph_get(path: str) -> dict:
    """Run a simple GET request against Microsoft Graph v1.0."""
    normalized_path = path if path.startswith("/") else f"/{path}"

    response = requests.get(
        f"{GRAPH_BASE_URL}{normalized_path}",
        headers=get_graph_headers(),
        timeout=30,
    )

    response.raise_for_status()
    return response.json()


def get_current_graph_user() -> dict:
    """Return the current delegated Graph user profile."""
    return graph_get("/me")


def list_onedrive_root_children() -> list[dict]:
    """List files and folders in the signed-in user's OneDrive root."""
    result = graph_get("/me/drive/root/children")
    return result.get("value", [])


def list_configured_onedrive_root_children() -> list[dict]:
    """List files/folders under the configured OneDrive connector root."""
    app_config = read_app_config()
    return list_onedrive_children_by_path(app_config.graph_onedrive_root_path)


def list_onedrive_children_by_item_id(item_id: str) -> list[dict]:
    """List files and folders inside one OneDrive folder item."""
    result = graph_get(f"/me/drive/items/{item_id}/children")
    return result.get("value", [])


def is_onedrive_folder(item: dict) -> bool:
    """Check whether a OneDrive item is a folder."""
    return "folder" in item


def is_onedrive_file(item: dict) -> bool:
    """Check whether a OneDrive item is a file."""
    return "file" in item


def graph_get_bytes(path: str) -> bytes:
    """Run a GET request against Microsoft Graph and return raw response bytes."""
    normalized_path = path if path.startswith("/") else f"/{path}"

    response = requests.get(
        f"{GRAPH_BASE_URL}{normalized_path}",
        headers=get_graph_headers(),
        timeout=60,
    )

    response.raise_for_status()
    return response.content


def download_onedrive_file_by_item_id(item_id: str) -> bytes:
    """Download raw bytes for one OneDrive file item."""
    return graph_get_bytes(f"/me/drive/items/{item_id}/content")


def list_onedrive_children_by_path(folder_path: str) -> list[dict]:
    """List OneDrive files/folders under a specific root-relative folder path."""
    normalized_folder_path = folder_path.strip("/")

    result = graph_get(
        f"/me/drive/root:/{normalized_folder_path}:/children"
    )

    return result.get("value", [])


def build_onedrive_child_path(parent_path: str, child_name: str) -> str:
    """Build a root-relative OneDrive path for a child item."""
    return f"{parent_path.rstrip('/')}/{child_name}"


def list_onedrive_files_recursive(folder_path: str | None = None) -> list[dict]:
    """List all files under a OneDrive folder path recursively."""
    app_config = read_app_config()
    root_path = folder_path or app_config.graph_onedrive_root_path

    discovered_files = []

    for item in list_onedrive_children_by_path(root_path):
        item_path = build_onedrive_child_path(root_path, item["name"])

        if is_onedrive_folder(item):
            discovered_files.extend(
                list_onedrive_files_recursive(item_path)
            )
        elif is_onedrive_file(item):
            file_item = item.copy()
            file_item["connector_path"] = item_path
            discovered_files.append(file_item)

    return discovered_files


def graph_path_id(item_id: str) -> str:
    """URL-encode a Graph item ID before placing it into a path."""
    return quote(item_id, safe="")


def list_onenote_notebooks() -> list[dict]:
    """List OneNote notebooks for the signed-in Graph user."""
    result = graph_get("/me/onenote/notebooks")
    return result.get("value", [])


def list_onenote_notebook_sections(notebook_id: str) -> list[dict]:
    """List sections inside one OneNote notebook."""
    encoded_notebook_id = graph_path_id(notebook_id)
    result = graph_get(f"/me/onenote/notebooks/{encoded_notebook_id}/sections")
    return result.get("value", [])


def list_onenote_section_pages(section_id: str) -> list[dict]:
    """List pages inside one OneNote section."""
    encoded_section_id = graph_path_id(section_id)
    result = graph_get(f"/me/onenote/sections/{encoded_section_id}/pages")
    return result.get("value", [])


def build_onenote_connector_path(
    notebook_name: str,
    section_name: str,
    page_title: str,
) -> str:
    """Build a readable connector path for a OneNote page."""
    return f"/OneNote/{notebook_name}/{section_name}/{page_title}"


def list_onenote_pages_recursive() -> list[dict]:
    """List OneNote pages with notebook and section context."""
    app_config = read_app_config()
    notebook_filter = app_config.graph_onenote_notebook_filter
    discovered_pages = []

    for notebook in list_onenote_notebooks():
        notebook_name = notebook.get("displayName", "Untitled Notebook")

        if notebook_filter and notebook_name.lower() != notebook_filter.lower():
            continue

        for section in list_onenote_notebook_sections(notebook["id"]):
            section_name = section.get("displayName", "Untitled Section")

            for page in list_onenote_section_pages(section["id"]):
                page_title = page.get("title", "Untitled Page")
                page_item = page.copy()
                page_item["notebook_name"] = notebook_name
                page_item["section_name"] = section_name
                page_item["connector_path"] = build_onenote_connector_path(
                    notebook_name=notebook_name,
                    section_name=section_name,
                    page_title=page_title,
                )
                discovered_pages.append(page_item)

    return discovered_pages


def download_onenote_page_content_by_id(page_id: str) -> bytes:
    """Download raw HTML content for one OneNote page."""
    encoded_page_id = graph_path_id(page_id)
    return graph_get_bytes(f"/me/onenote/pages/{encoded_page_id}/content")