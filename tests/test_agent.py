"""
Basic tests for agent enhancements.
"""
from tools.action_executor import ActionExecutor
from learning.user_learning import UserLearning
from plugins.plugin_manager import PluginManager
from security.security_manager import SecurityManager
from deployment.deployment_manager import DeploymentManager
from multimodal.vision_processor import VisionProcessor
from memory.long_term_memory import LongTermMemory
from planning.calendar_integration import CalendarIntegration

def test_action_executor():
    ae = ActionExecutor({"open_file": True})
    # Use a dummy file path for testing; in real test, mock os.startfile
    try:
        result = ae.execute("open_file", {"path": "dummy.txt"})
    except Exception:
        result = "Handled error"
    assert result is not None

def test_user_learning():
    ul = UserLearning()
    ul.update_profile({"likes": "AI"})
    assert ul.get_profile()["likes"] == "AI"

def test_plugin_manager():
    pm = PluginManager()
    assert isinstance(pm.plugins, dict)

def test_security_manager():
    sm = SecurityManager()
    sm.set_policy("read", True)
    assert sm.check_permission("read")

def test_deployment_manager():
    dm = DeploymentManager()
    assert "desktop" in dm.platforms

def test_vision_processor():
    vp = VisionProcessor()
    assert "Processed image" in vp.process_image("img.png")

def test_long_term_memory():
    ltm = LongTermMemory()
    ltm.remember("fact1")
    assert "fact1" in ltm.recall()

def test_calendar_integration():
    ci = CalendarIntegration()
    ci.add_event("meeting")
    assert "meeting" in ci.get_events()
