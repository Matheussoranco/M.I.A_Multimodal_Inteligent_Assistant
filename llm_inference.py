from openai import OpenAI
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import webbrowser
import os

class LLMInference:
    def __init__(self, model_id='mistral:instruct', url='http://localhost:11434/v1', api_key='ollama', llama_model_path=None):
        """
        Initialize the LLMInference with support for multiple APIs and LLama models.

        :param model_id: Model identifier for API-based models.
        :param url: Base URL for the API.
        :param api_key: API key for authentication.
        :param llama_model_path: Path to local LLama 70B model (if applicable).
        """
        self.model_id = model_id
        self.url = url
        self.api_key = api_key
        self.client = OpenAI(base_url=self.url, api_key=self.api_key)
        self.llama_model_path = llama_model_path

    def query_model(self, text):
        """
        Query the model using the API or local LLama model.

        :param text: User input to the model.
        :return: Model response as text.
        """
        if self.llama_model_path:
            return self._query_llama(text)
        else:
            return self._query_api(text)

    def _query_api(self, text):
        """
        Query the API-based model.

        :param text: User input to the model.
        :return: Model response as text.
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are Liva, a helpful assistant. You provide single-sentence, accurate answers to the user's question."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content

    def _query_llama(self, text):
        """
        Query the LLama model locally (placeholder for actual LLama integration).

        :param text: User input to the model.
        :return: Model response as text.
        """
        # Placeholder: Integrate with LLama-specific library like HuggingFace's Transformers.
        # Example: `from transformers import AutoModelForCausalLM, AutoTokenizer`
        return "LLama model response (placeholder)"

    def write_email(self, recipient, subject, body):
        """
        Compose and send an email.

        :param recipient: Recipient email address.
        :param subject: Email subject.
        :param body: Email body content.
        :return: Confirmation of email sent.
        """
        # Placeholder for actual email sending logic (e.g., using smtplib or a third-party service like SendGrid).
        return f"Email to {recipient} with subject '{subject}' sent successfully."

    def write_message(self, platform, recipient, message):
        """
        Send a message on a specified platform.

        :param platform: Messaging platform (e.g., WhatsApp, Slack).
        :param recipient: Recipient's identifier on the platform.
        :param message: Message content.
        :return: Confirmation of message sent.
        """
        # Placeholder for actual messaging logic (e.g., using platform-specific APIs).
        return f"Message to {recipient} on {platform} sent successfully."

    def take_note(self, app, content):
        """
        Create a note in a specified app.

        :param app: Note-taking app (e.g., Notion, Evernote).
        :param content: Content of the note.
        :return: Confirmation of note saved.
        """
        # Placeholder for integration with app-specific APIs.
        return f"Note saved in {app} with content: '{content}'"

    def open_program_or_website(self, target):
        """
        Open a program or website.

        :param target: Path to the program or URL of the website.
        :return: Confirmation of action.
        """

        if target.startswith("http"):
            webbrowser.open(target)
            return f"Website {target} opened."
        else:
            os.system(f"open {target}")  # Adjust for Windows/Linux if needed
            return f"Program {target} opened."

    def autofill_login(self, url, username, password):
        """
        Autofill login fields on a website.

        :param url: URL of the login page.
        :param username: Username for login.
        :param password: Password for login.
        :return: Confirmation of autofill success.
        """

        driver = webdriver.Chrome()  # Adjust driver path if needed
        driver.get(url)

        try:
            user_field = driver.find_element(By.NAME, "username")
            pass_field = driver.find_element(By.NAME, "password")

            user_field.send_keys(username)
            pass_field.send_keys(password)
            pass_field.send_keys(Keys.RETURN)
            return "Login fields filled successfully."
        except Exception as e:
            return f"Failed to autofill login: {e}"

# Example usage:
# llm = LLMInference(llama_model_path="path/to/llama/model")
# print(llm.query_model("What's the weather today?"))
