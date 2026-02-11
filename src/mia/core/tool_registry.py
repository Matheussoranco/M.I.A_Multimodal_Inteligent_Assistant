"""
Tool Registry — OpenAI-compatible tool/function definitions for M.I.A.

Defines the complete tool catalogue the agent can invoke.  Every tool is
described using the OpenAI *function calling* JSON schema so it works
out-of-the-box with OpenAI, Ollama (llama3+), Anthropic, Groq, and any
other provider that accepts the same format.

For models that do **not** support native tool calling, the helper
``get_tool_descriptions_text()`` produces a plain-text summary suitable
for a ReAct-style prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definition Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _tool(
    name: str,
    description: str,
    properties: Dict[str, Any] | None = None,
    required: List[str] | None = None,
) -> Dict[str, Any]:
    """Shorthand to build an OpenAI function-calling tool dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def _prop(desc: str, type_: str = "string", **kw: Any) -> Dict[str, Any]:
    """Shorthand for a JSON Schema property."""
    d: Dict[str, Any] = {"type": type_, "description": desc}
    d.update(kw)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# Core Tool Definitions (OpenAI function-calling format)
# ═══════════════════════════════════════════════════════════════════════════════

CORE_TOOLS: List[Dict[str, Any]] = [
    # ── Web & Research ──────────────────────────────────────────────
    _tool(
        "web_search",
        "Search the web for current information on any topic. Use when the "
        "user asks about recent events, facts, or anything requiring "
        "up-to-date information.",
        {"query": _prop("The search query")},
        ["query"],
    ),
    _tool(
        "open_url",
        "Open a URL in the user's default web browser.",
        {"url": _prop("Full URL to open (e.g. https://example.com)")},
        ["url"],
    ),
    _tool(
        "web_scrape",
        "Scrape and extract text content from a web page URL.",
        {"url": _prop("The URL to scrape content from")},
        ["url"],
    ),
    _tool(
        "research_topic",
        "Perform deep research on a topic by combining web search, "
        "wikipedia, and other sources into a comprehensive summary.",
        {"topic": _prop("The topic to research in depth")},
        ["topic"],
    ),
    _tool(
        "wikipedia_search",
        "Search Wikipedia for encyclopedic information on a topic.",
        {"query": _prop("The Wikipedia search query")},
        ["query"],
    ),
    _tool(
        "youtube_search",
        "Search YouTube for videos on a topic.",
        {"query": _prop("The YouTube search query")},
        ["query"],
    ),

    # ── File System ─────────────────────────────────────────────────
    _tool(
        "create_file",
        "Create a new file with the specified content.",
        {
            "path": _prop("File path to create"),
            "content": _prop("Content to write to the file"),
        },
        ["path", "content"],
    ),
    _tool(
        "read_file",
        "Read the contents of a file from the filesystem.",
        {"path": _prop("File path to read")},
        ["path"],
    ),
    _tool(
        "write_file",
        "Write or overwrite content to a file at the given path.",
        {
            "path": _prop("File path to write"),
            "content": _prop("Content to write"),
        },
        ["path", "content"],
    ),
    _tool(
        "delete_file",
        "Delete a file from the filesystem.",
        {"path": _prop("File path to delete")},
        ["path"],
    ),
    _tool(
        "move_file",
        "Move or rename a file from one path to another.",
        {
            "src": _prop("Source file path"),
            "dst": _prop("Destination file path"),
        },
        ["src", "dst"],
    ),
    _tool(
        "search_file",
        "Search for files by name pattern in a directory tree.",
        {
            "name": _prop("File name or glob pattern to search for"),
            "directory": _prop("Directory to start search from (default: '.')"),
        },
        ["name"],
    ),
    _tool(
        "create_directory",
        "Create a new directory (including parent directories).",
        {"path": _prop("Directory path to create")},
        ["path"],
    ),
    _tool(
        "open_directory",
        "Open a directory in the system file manager.",
        {"path": _prop("Directory path to open")},
        ["path"],
    ),

    # ── Shell / System ──────────────────────────────────────────────
    _tool(
        "run_command",
        "Execute a shell command on the user's system and return its "
        "stdout/stderr output. Use for system tasks like listing files, "
        "checking disk space, running scripts, installing packages, etc.",
        {"command": _prop("The shell command to execute")},
        ["command"],
    ),
    _tool("get_system_info",
        "Get information about the user's system (OS, CPU, memory, disk "
        "space, running processes).",
    ),

    # ── Code ────────────────────────────────────────────────────────
    _tool(
        "create_code",
        "Generate code in a specified programming language based on a "
        "natural language description, optionally saving to a file.",
        {
            "language": _prop("Programming language (python, javascript, etc.)"),
            "description": _prop("Description of what the code should do"),
            "filename": _prop("Optional filename to save the code to"),
        },
        ["language", "description"],
    ),
    _tool(
        "analyze_code",
        "Read and analyze a code file for issues, improvements, and "
        "provide suggestions.",
        {"path": _prop("Path to the code file to analyze")},
        ["path"],
    ),

    # ── Notes & Documentation ──────────────────────────────────────
    _tool(
        "make_note",
        "Create a note with content and optional title for future reference.",
        {
            "content": _prop("Note content"),
            "title": _prop("Optional note title"),
        },
        ["content"],
    ),
    _tool("read_notes", "Read all saved notes."),
    _tool(
        "search_notes",
        "Search through saved notes by keyword or phrase.",
        {"query": _prop("Search query for notes")},
        ["query"],
    ),

    # ── Documents ───────────────────────────────────────────────────
    _tool(
        "create_document",
        "Create a document (Word DOCX or PDF). Supports report and "
        "proposal templates.",
        {
            "title": _prop("Document title"),
            "content": _prop("Document body content (Markdown OK)"),
            "format": _prop("Output format", enum=["docx", "pdf"]),
            "template": _prop("Document template", enum=["report", "proposal"]),
        },
        ["title", "content"],
    ),

    # ── Spreadsheets ────────────────────────────────────────────────
    _tool(
        "create_sheet",
        "Create a new spreadsheet (XLSX/CSV) with the given data.",
        {
            "filename": _prop("Output filename (e.g. report.xlsx)"),
            "data": _prop("Data as list of rows (each row is a list)", type_="array"),
        },
        ["filename"],
    ),
    _tool(
        "read_sheet",
        "Read the contents of a spreadsheet file.",
        {"filename": _prop("Path to the spreadsheet file")},
        ["filename"],
    ),
    _tool(
        "write_sheet",
        "Write data rows to an existing spreadsheet.",
        {
            "filename": _prop("Path to the spreadsheet file"),
            "data": _prop("Data rows to write", type_="array"),
        },
        ["filename", "data"],
    ),

    # ── Presentations ───────────────────────────────────────────────
    _tool(
        "create_presentation",
        "Create a new PowerPoint presentation.",
        {
            "filename": _prop("Output filename (e.g. deck.pptx)"),
            "title": _prop("Presentation title"),
            "content": _prop("Content for slides"),
        },
        ["filename"],
    ),
    _tool(
        "add_presentation_slide",
        "Add a slide to an existing PowerPoint presentation.",
        {
            "filename": _prop("Path to the .pptx file"),
            "title": _prop("Slide title"),
            "bullets": _prop("Bullet points for the slide", type_="array"),
        },
        ["filename"],
    ),

    # ── Memory ──────────────────────────────────────────────────────
    _tool(
        "store_memory",
        "Store a fact or piece of information in long-term memory for "
        "future reference.",
        {
            "fact": _prop("The information to remember"),
            "category": _prop(
                "Category for organisation (e.g. 'preference', 'fact', 'task')"
            ),
        },
        ["fact"],
    ),
    _tool(
        "search_memory",
        "Search long-term memory for previously stored information.",
        {"query": _prop("Search query")},
        ["query"],
    ),
    _tool(
        "link_memory_nodes",
        "Create a relationship link between two memory entries.",
        {
            "source": _prop("Source memory node"),
            "target": _prop("Target memory node"),
            "relation": _prop("Relationship type"),
        },
        ["source", "target", "relation"],
    ),

    # ── Communication ───────────────────────────────────────────────
    _tool(
        "send_email",
        "Send an email to a recipient.",
        {
            "to": _prop("Recipient email address"),
            "subject": _prop("Email subject line"),
            "body": _prop("Email body content"),
        },
        ["to", "subject", "body"],
    ),
    _tool(
        "send_whatsapp",
        "Send a WhatsApp message.",
        {
            "to": _prop("Recipient phone number or contact"),
            "message": _prop("Message text to send"),
        },
        ["to", "message"],
    ),
    _tool(
        "send_telegram",
        "Send a Telegram message.",
        {
            "to": _prop("Recipient chat ID or username"),
            "message": _prop("Message text to send"),
        },
        ["to", "message"],
    ),
    _tool(
        "send_message",
        "Send a message via the best available channel.",
        {
            "to": _prop("Recipient identifier"),
            "message": _prop("Message text to send"),
            "channel": _prop("Channel to use (email, whatsapp, telegram)"),
        },
        ["to", "message"],
    ),

    # ── Calendar & Scheduling ───────────────────────────────────────
    _tool(
        "calendar_event",
        "Create, update, or query calendar events.",
        {
            "action": _prop("Action to perform", enum=["create", "list", "delete"]),
            "title": _prop("Event title"),
            "date": _prop("Event date (YYYY-MM-DD)"),
            "time": _prop("Event time (HH:MM)"),
            "duration": _prop("Duration in minutes", type_="integer"),
            "description": _prop("Event description"),
        },
        ["action"],
    ),

    # ── Desktop Automation ──────────────────────────────────────────
    _tool(
        "desktop_open_app",
        "Open an application on the user's desktop.",
        {"app_path": _prop("Application name or path to open")},
        ["app_path"],
    ),
    _tool(
        "desktop_close_app",
        "Close a running application.",
        {"app_name": _prop("Name of the application to close")},
        ["app_name"],
    ),
    _tool(
        "desktop_type_text",
        "Type text at the current cursor position on screen.",
        {"text": _prop("Text to type")},
        ["text"],
    ),
    _tool(
        "desktop_click",
        "Perform a mouse click at screen coordinates or on an element.",
        {
            "x": _prop("X coordinate", type_="integer"),
            "y": _prop("Y coordinate", type_="integer"),
            "button": _prop("Mouse button", enum=["left", "right", "middle"]),
        },
    ),
    _tool(
        "desktop_send_keys",
        "Send keyboard shortcuts or key combinations.",
        {"keys": _prop("Key combination (e.g. 'ctrl+c', 'alt+tab')")},
        ["keys"],
    ),
    _tool(
        "desktop_get_text",
        "Read text from the active window or screen region using OCR.",
        {
            "region": _prop(
                "Optional screen region as {x, y, width, height}",
                type_="object",
            ),
        },
    ),

    # ── Clipboard & Notifications ───────────────────────────────────
    _tool(
        "clipboard_copy",
        "Copy text to the system clipboard.",
        {"text": _prop("Text to copy to clipboard")},
        ["text"],
    ),
    _tool("clipboard_paste", "Paste and return the current clipboard contents."),
    _tool(
        "show_notification",
        "Show a desktop notification to the user.",
        {
            "title": _prop("Notification title"),
            "message": _prop("Notification message body"),
        },
        ["title", "message"],
    ),

    # ── Application Control ─────────────────────────────────────────
    _tool(
        "launch_app",
        "Launch an application by name.",
        {"app": _prop("Application name to launch")},
        ["app"],
    ),
    _tool(
        "close_app",
        "Close an application by name.",
        {"app": _prop("Application name to close")},
        ["app"],
    ),
    _tool(
        "open_application",
        "Open a system application.",
        {"app_name": _prop("Application name")},
        ["app_name"],
    ),

    # ── Smart Home ──────────────────────────────────────────────────
    _tool(
        "control_device",
        "Control a smart home device (lights, thermostat, etc.).",
        {
            "device_type": _prop("Type of device (light, thermostat, switch, etc.)"),
            "action": _prop("Action to perform (on, off, set, toggle)"),
            "value": _prop("Optional value (brightness level, temperature)"),
        },
        ["device_type", "action"],
    ),

    # ── OCR & Document Intelligence ─────────────────────────────────
    _tool(
        "ocr_extract_text",
        "Extract text from an image or scanned document using OCR.",
        {"path": _prop("Path to the image or document file")},
        ["path"],
    ),
    _tool(
        "ocr_analyze_document",
        "Analyze a document image for layout, tables, and text regions.",
        {"path": _prop("Path to the document image")},
        ["path"],
    ),

    # ── Embeddings ──────────────────────────────────────────────────
    _tool(
        "embed_text",
        "Generate a vector embedding for a piece of text.",
        {"text": _prop("The text to embed")},
        ["text"],
    ),
    _tool(
        "embed_similarity",
        "Compare the semantic similarity between two pieces of text.",
        {
            "text1": _prop("First text"),
            "text2": _prop("Second text"),
        },
        ["text1", "text2"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return the tool definition dict whose function name matches *name*."""
    for tool in CORE_TOOLS:
        if tool["function"]["name"] == name:
            return tool
    return None


def get_tool_names() -> List[str]:
    """Return a flat list of every registered tool name."""
    return [t["function"]["name"] for t in CORE_TOOLS]


def get_tool_descriptions_text() -> str:
    """Render tool descriptions as plain text for ReAct-style prompts.

    Example output::

        - web_search: Search the web …
          Parameters: query (required): The search query
    """
    lines: List[str] = []
    for tool in CORE_TOOLS:
        fn = tool["function"]
        props = fn["parameters"].get("properties", {})
        required = set(fn["parameters"].get("required", []))
        params: List[str] = []
        for k, v in props.items():
            req = "required" if k in required else "optional"
            desc = v.get("description", v.get("type", "string"))
            params.append(f"{k} ({req}): {desc}")
        param_str = ", ".join(params) if params else "none"
        lines.append(
            f"  - {fn['name']}: {fn['description']}\n"
            f"    Parameters: {param_str}"
        )
    return "\n".join(lines)


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> bool:
    """Return *True* if all required arguments are present for *tool_name*."""
    tool = get_tool_by_name(tool_name)
    if not tool:
        # Unknown tool — let the executor handle it
        return True
    required = tool["function"]["parameters"].get("required", [])
    return all(arg in args for arg in required)
