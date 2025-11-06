import requests


class MessageUtil:
    @staticmethod
    def send_message(
        api_url: str,
        api_key: str,
        recipient_id: str,
        message: str,
        platform: str,
    ) -> str:
        """
        Send a message on any platform using a specified API.

        :param api_url: Base URL for the messaging API.
        :param api_key: API key for authentication.
        :param recipient_id: Identifier of the recipient on the platform.
        :param message: Message content.
        :param platform: Name of the platform (e.g., "WhatsApp", "Slack", "Telegram").
        :return: Confirmation of message sent.
        """
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "recipient_id": recipient_id,
            "message": message,
            "platform": platform,
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()

            return (
                f"Message to {recipient_id} on {platform} sent successfully."
            )
        except requests.exceptions.RequestException as e:
            return f"Failed to send message: {e}"


# Example usage:
# MessageUtil.send_message(
#     api_url="https://api.example.com/send_message",
#     api_key="your_api_key",
#     recipient_id="12345",
#     message="Hello, this is a test message!",
#     platform="WhatsApp"
# )
