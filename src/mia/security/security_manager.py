"""
Security Manager: Handles user data protection, permissions, and explainability.
"""
import time
import logging
from typing import Dict, List, Set, Optional
from pathlib import Path

# Import custom exceptions and error handling
from ..exceptions import SecurityError, ValidationError, ConfigurationError
from ..error_handler import global_error_handler, with_error_handling

logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self, config_manager=None):
        self.data_policies: Dict[str, bool] = {
            # Default safe actions
            "read_file": False,  # Requires explicit permission
            "write_file": False,
            "execute_command": False,
            "web_search": True,  # Generally safe
            "calendar_access": False,
            "send_email": False,
            "system_control": False
        }
        self.action_history: List[Dict] = []
        self.blocked_paths: Set[str] = {
            "/etc/passwd", "/etc/shadow", "C:\\Windows\\System32",
            "/root", "C:\\Users\\Administrator"
        }
        
    @with_error_handling(global_error_handler, fallback_value=False)
    def check_permission(self, action: str, context: Optional[Dict] = None) -> bool:
        """Check if action is permitted with comprehensive validation."""
        if not action:
            raise ValidationError("Empty action provided", "EMPTY_ACTION")
            
        if not isinstance(action, str):
            raise ValidationError("Action must be a string", "INVALID_ACTION_TYPE")
        
        # Log the permission check
        self._log_action_attempt(action, context)
        
        try:
            # Basic action permission
            if action not in self.data_policies:
                logger.warning(f"Unknown action attempted: {action}")
                raise SecurityError(f"Unknown action: {action}", "UNKNOWN_ACTION")
                
            if not self.data_policies[action]:
                logger.info(f"Action denied by policy: {action}")
                return False
                
            # Additional context-based security checks
            if context:
                if not self._validate_context(action, context):
                    raise SecurityError(f"Context validation failed for action: {action}", 
                                      "CONTEXT_VALIDATION_FAILED")
                    
            logger.debug(f"Action permitted: {action}")
            return True
            
        except (SecurityError, ValidationError):
            raise
        except Exception as e:
            raise SecurityError(f"Permission check failed: {str(e)}", "PERMISSION_CHECK_ERROR")

    def _validate_context(self, action: str, context: Dict) -> bool:
        """Validate action context for security with comprehensive checks."""
        try:
            if action in ["read_file", "write_file"] and "path" in context:
                return self._validate_file_path(context["path"])
                    
            if action == "execute_command" and "command" in context:
                return self._validate_command(context["command"])
                
            if action == "web_search" and "query" in context:
                return self._validate_web_query(context["query"])
                
            return True
            
        except Exception as e:
            logger.error(f"Context validation error for {action}: {e}")
            return False
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        try:
            file_path = str(file_path)
            
            # Block access to sensitive system files
            for blocked_path in self.blocked_paths:
                if blocked_path.lower() in file_path.lower():
                    logger.warning(f"Blocked access to sensitive path: {file_path}")
                    return False
            
            # Check for path traversal attempts
            if ".." in file_path or "~" in file_path:
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return False
                
            # Validate path format
            try:
                Path(file_path).resolve()
            except (OSError, ValueError) as e:
                logger.warning(f"Invalid path format: {file_path} - {e}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"File path validation error: {e}")
            return False
    
    def _validate_command(self, command: str) -> bool:
        """Validate command for security."""
        try:
            command = command.lower()
            
            # Block dangerous commands
            dangerous_commands = [
                "rm -rf", "del /f", "format", "fdisk", "regedit", 
                "shutdown", "reboot", "halt", "poweroff", "sudo rm",
                "dd if=", "mkfs", "chmod 777", "chown root"
            ]
            
            if any(cmd in command for cmd in dangerous_commands):
                logger.warning(f"Blocked dangerous command: {command}")
                return False
                
            # Check for command injection attempts
            injection_patterns = [";", "&&", "||", "|", "&", "`", "$"]
            if any(pattern in command for pattern in injection_patterns):
                logger.warning(f"Command injection attempt detected: {command}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Command validation error: {e}")
            return False
    
    def _validate_web_query(self, query: str) -> bool:
        """Validate web search query for security."""
        try:
            # Check for malicious patterns
            malicious_patterns = ["<script>", "javascript:", "data:", "file://"]
            if any(pattern in query.lower() for pattern in malicious_patterns):
                logger.warning(f"Malicious web query detected: {query}")
                return False
                
            # Check query length
            if len(query) > 1000:
                logger.warning(f"Web query too long: {len(query)} characters")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Web query validation error: {e}")
            return False
        
    def _log_action_attempt(self, action: str, context: Optional[Dict] = None):
        """Log action attempts for audit trail with error handling."""
        try:
            entry = {
                "action": action,
                "context": context,
                "timestamp": time.time(),
                "allowed": self.data_policies.get(action, False)
            }
            
            self.action_history.append(entry)
            
            # Keep only last 1000 entries
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to log action attempt: {e}")

    def set_policy(self, action: str, allowed: bool):
        """Set permission policy for an action with validation."""
        if not action or not isinstance(action, str):
            raise ValidationError("Invalid action provided", "INVALID_ACTION")
            
        if not isinstance(allowed, bool):
            raise ValidationError("Permission value must be boolean", "INVALID_PERMISSION")
            
        try:
            self.data_policies[action] = allowed
            logger.info(f"Policy updated: {action} = {allowed}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to set policy: {str(e)}", "POLICY_SET_FAILED")

    def explain_action(self, action: str) -> str:
        """Return detailed explanation for action with error handling."""
        try:
            if not action:
                return "No action specified"
                
            allowed = self.data_policies.get(action, False)
            explanation = f"Action '{action}' is {'ALLOWED' if allowed else 'DENIED'}."
            
            if not allowed:
                if action in ["read_file", "write_file"]:
                    explanation += " File operations require explicit permission for security."
                elif action == "execute_command":
                    explanation += " Command execution is restricted to prevent system damage."
                elif action == "system_control":
                    explanation += " System control requires administrator privileges."
                elif action == "web_search":
                    explanation += " Web search is generally allowed but may be restricted in some contexts."
                else:
                    explanation += " This action is not permitted by current security policy."
                    
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain action {action}: {e}")
            return f"Unable to explain action: {action}"
        
        return explanation
        
    def has_scope(self, scope: str) -> bool:
        """Check if a specific scope is allowed."""
        try:
            if not scope or not isinstance(scope, str):
                return False
                
            # Map scopes to actions for permission checking
            scope_to_action = {
                "files.read": "read_file",
                "files.write": "write_file", 
                "system": "execute_command",
                "web": "web_search",
                "messaging": "send_email",
                "iot": "system_control",
                "memory.read": "web_search",  # Using web_search as proxy for memory read
                "memory.write": "web_search",  # Using web_search as proxy for memory write
                "productivity": "calendar_access"
            }
            
            action = scope_to_action.get(scope)
            if action:
                return self.data_policies.get(action, False)
                
            # Default to False for unknown scopes
            logger.debug(f"Unknown scope requested: {scope}")
            return False
            
        except Exception as e:
            logger.error(f"Scope check failed for {scope}: {e}")
            return False

    def has_scopes(self, scopes: Set[str]) -> bool:
        """Check if all specified scopes are allowed."""
        try:
            if not scopes:
                return True
                
            return all(self.has_scope(scope) for scope in scopes)
            
        except Exception as e:
            logger.error(f"Scopes check failed: {e}")
            return False
