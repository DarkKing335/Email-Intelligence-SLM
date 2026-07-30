"""Email Intelligence dashboard (US-4.5 / 4.6 / 5.4).

A Streamlit frontend over the Epic 3 API: review queue, keyword search, and a
human-in-the-loop review screen (approve / edit / reject).
"""

from email_dashboard.api_client import ApiClient, ApiError

__all__ = ["ApiClient", "ApiError"]
