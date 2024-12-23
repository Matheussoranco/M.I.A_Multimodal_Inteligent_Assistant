import webbrowser
import os
import platform

class AutomationUtil:
    @staticmethod
    def open_website(url):
        """
        Open a website in the default web browser.

        :param url: URL of the website to open.
        :return: Confirmation message.
        """
        try:
            webbrowser.open(url)
            return f"Website {url} opened successfully."
        except Exception as e:
            return f"Failed to open website: {e}"

    @staticmethod
    def open_program(program_path):
        """
        Open a program based on the operating system.

        :param program_path: Path to the program executable.
        :return: Confirmation message.
        """
        try:
            system_platform = platform.system()

            if system_platform == "Windows":
                os.startfile(program_path)
            elif system_platform == "Darwin":  # macOS
                os.system(f"open {program_path}")
            elif system_platform == "Linux":
                os.system(f"xdg-open {program_path}")
            else:
                return "Unsupported operating system."

            return f"Program at {program_path} opened successfully."
        except Exception as e:
            return f"Failed to open program: {e}"

    @staticmethod
    def open_file(file_path):
        """
        Open a file using the default application for its type.

        :param file_path: Path to the file to open.
        :return: Confirmation message.
        """
        try:
            system_platform = platform.system()

            if system_platform == "Windows":
                os.startfile(file_path)
            elif system_platform == "Darwin":  # macOS
                os.system(f"open {file_path}")
            elif system_platform == "Linux":
                os.system(f"xdg-open {file_path}")
            else:
                return "Unsupported operating system."

            return f"File at {file_path} opened successfully."
        except Exception as e:
            return f"Failed to open file: {e}"

# Example usage:
# AutomationUtil.open_website("https://www.example.com")
# AutomationUtil.open_program("/path/to/program")
# AutomationUtil.open_file("/path/to/file")
