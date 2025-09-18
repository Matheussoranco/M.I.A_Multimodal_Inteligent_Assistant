import webbrowser
import os
import platform
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from typing import Optional

class AutomationUtil:
    @staticmethod
    def open_website(url: str) -> str:
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
    def open_program(program_path: str) -> str:
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
    def open_file(file_path: str) -> str:
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

    @staticmethod
    def autofill_login(url: str, username: str, password: str) -> str:
        """
        Autofill login fields on a website using Selenium.

        :param url: URL of the login page.
        :param username: Username for login.
        :param password: Password for login.
        :return: Confirmation of autofill success.
        """
        try:
            driver = webdriver.Chrome()  # Ensure the correct WebDriver is installed and in PATH
            driver.get(url)

            # Locate the username and password fields
            username_field = driver.find_element(By.NAME, "username")
            password_field = driver.find_element(By.NAME, "password")

            # Fill in the fields
            username_field.send_keys(username)
            password_field.send_keys(password)
            password_field.send_keys(Keys.RETURN)

            return f"Login fields filled successfully for {url}."
        except Exception as e:
            return f"Failed to autofill login: {e}"
        finally:
            driver.quit()

# Example usage:
# AutomationUtil.open_website("https://www.example.com")
# AutomationUtil.open_program("/path/to/program")
# AutomationUtil.open_file("/path/to/file")
# AutomationUtil.autofill_login("https://www.example.com/login", "myusername", "mypassword")
