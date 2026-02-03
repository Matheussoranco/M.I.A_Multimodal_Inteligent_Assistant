"""
System Control: Cross-platform system actions (file, process, clipboard, etc.)
"""

import os
import platform
import shutil
import subprocess
import shlex

import psutil
import pyperclip


class SystemControl:
    @staticmethod
    def open_file(path):
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    @staticmethod
    def list_files(directory):
        return os.listdir(directory)

    @staticmethod
    def copy_file(src, dst):
        shutil.copy2(src, dst)

    @staticmethod
    def move_file(src, dst):
        shutil.move(src, dst)

    @staticmethod
    def delete_file(path):
        os.remove(path)

    @staticmethod
    def run_command(cmd):
        if not cmd:
            return ""

        unsafe_shell = os.getenv("MIA_UNSAFE_SHELL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        if unsafe_shell:
            completed = subprocess.run(
                str(cmd),
                shell=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout + completed.stderr

        argv = (
            [str(part) for part in cmd]
            if isinstance(cmd, (list, tuple))
            else shlex.split(str(cmd), posix=(os.name != "nt"))
        )
        if not argv:
            return ""

        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout + completed.stderr

    @staticmethod
    def get_clipboard():
        return pyperclip.paste()

    @staticmethod
    def set_clipboard(text):
        pyperclip.copy(text)

    @staticmethod
    def list_processes():
        return [(p.pid, p.name()) for p in psutil.process_iter()]

    @staticmethod
    def kill_process(pid):
        p = psutil.Process(pid)
        p.terminate()

    # Add more system actions as needed (clipboard, process, etc.)
