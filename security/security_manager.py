"""
Security Manager: Handles user data protection, permissions, and explainability.
"""
import hashlib
import time
import logging
from typing import Dict, List, Set, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SecurityManager:
    def __init__(self):
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
        
    def check_permission(self, action: str, context: Optional[Dict] = None) -> bool:
        """Check if action is permitted with context validation."""
        # Log the permission check
        self._log_action_attempt(action, context)
        
        # Basic action permission
        if action not in self.data_policies:
            logger.warning(f"Unknown action attempted: {action}")
            return False
            
        if not self.data_policies[action]:
            return False
            
        # Additional context-based security checks
        if context:
            if not self._validate_context(action, context):
                return False
                
        return True

    def _validate_context(self, action: str, context: Dict) -> bool:
        """Validate action context for security."""
        if action in ["read_file", "write_file"] and "path" in context:
            file_path = str(context["path"])
            
            # Block access to sensitive system files
            for blocked_path in self.blocked_paths:
                if blocked_path.lower() in file_path.lower():
                    logger.warning(f"Blocked access to sensitive path: {file_path}")
                    return False
                    
        if action == "execute_command" and "command" in context:
            command = context["command"].lower()
            # Block dangerous commands
            dangerous_commands = ["rm -rf", "del /f", "format", "fdisk", "regedit"]
            if any(cmd in command for cmd in dangerous_commands):
                logger.warning(f"Blocked dangerous command: {command}")
                return False
                
        return True
        
    def _log_action_attempt(self, action: str, context: Optional[Dict] = None):
        """Log action attempts for audit trail."""
        self.action_history.append({
            "action": action,
            "context": context,
            "timestamp": time.time(),
            "allowed": self.data_policies.get(action, False)
        })
        
        # Keep only last 1000 entries
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]

    def set_policy(self, action: str, allowed: bool):
        """Set permission policy for an action."""
        self.data_policies[action] = allowed
        logger.info(f"Policy updated: {action} = {allowed}")

    def explain_action(self, action: str) -> str:
        """Return detailed explanation for action."""
        allowed = self.data_policies.get(action, False)
        explanation = f"Action '{action}' is {'ALLOWED' if allowed else 'DENIED'}."
        
        if not allowed:
            if action in ["read_file", "write_file"]:
                explanation += " File operations require explicit permission for security."
            elif action == "execute_command":
                explanation += " Command execution is restricted to prevent system damage."
            elif action == "system_control":
                explanation += " System control requires administrator privileges."
        
        return explanation
        
    def get_audit_trail(self, limit: int = 50) -> List[Dict]:
        """Get recent action audit trail."""
        return self.action_history[-limit:]
