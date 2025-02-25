from safety.sanitizer import ActionSanitizer

class AutomationUtil:
    def __init__(self):
        self.sanitizer = ActionSanitizer()

    @staticmethod
    def execute_action(action_command):
        """Enhanced action execution with security checks"""
        try:
            self.sanitizer.validate_action(action_command)
            
            if action_command.startswith("OPEN_URL:"):
                url = action_command[9:]
                return webbrowser.open(url)
                
            elif action_command.startswith("RUN_SCRIPT:"):
                # Sandboxed execution
                return self._execute_safe_python(action_command[11:])
                
            else:
                return "Unsupported action type"
                
        except SecurityViolation as e:
            return f"Blocked dangerous action: {e}"

    def _execute_safe_python(self, code):
        """Sandboxed code execution"""
        # Implement restricted Python execution
        return "Code execution result"