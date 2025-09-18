import requests

class NoteUtil:
    @staticmethod
    def create_note(api_url: str, api_key: str, note_title: str, note_content: str, app: str) -> str:
        """
        Create a note in any note-taking application using its API.

        :param api_url: Base URL for the note-taking API.
        :param api_key: API key for authentication.
        :param note_title: Title of the note.
        :param note_content: Content of the note.
        :param app: Name of the application (e.g., "Notion", "Evernote").
        :return: Confirmation of note creation.
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "title": note_title,
            "content": note_content,
            "app": app
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            return f"Note titled '{note_title}' created in {app} successfully."
        except requests.exceptions.RequestException as e:
            return f"Failed to create note: {e}"

# Example usage:
# NoteUtil.create_note(
#     api_url="https://api.example.com/create_note",
#     api_key="your_api_key",
#     note_title="Meeting Notes",
#     note_content="Discuss project timelines and deliverables.",
#     app="Notion"
# )
