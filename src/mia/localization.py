"""
Localization module for M.I.A - Multimodal Intelligent Assistant.
Provides internationalization support for English and Portuguese.
"""

import os
from typing import Any, Dict, Optional


class Localization:
    """Handles localization for M.I.A interface."""

    def __init__(self, language: str = "en"):
        """
        Initialize localization.

        Args:
            language: Language code ("en" or "pt")
        """
        self.language = language.lower()
        if self.language not in ["en", "pt"]:
            self.language = "en"

        self.strings = self._load_strings()

    def _load_strings(self) -> Dict[str, Any]:
        """Load localized strings."""
        strings = {
            "en": {
                "app_title": "M.I.A - Multimodal Intelligent Assistant",
                "app_version": "Version: 0.1.0",
                "initializing": "Initializing components...",
                "ollama_warning": "Warning: Could not connect to Ollama. Attempting to use alternative LLM provider.",
                "ollama_success": "Connected to Ollama successfully!",
                "commands_available": "\nAvailable commands:",
                "help_command": "  'help' - Show help",
                "exit_command": "  'exit' - Exit",
                "agent_commands": "\nAgent Commands:",
                "create_file_help": "  'create file [name]' - Create a new file",
                "make_note_help": "  'make note [title]' - Create a note",
                "analyze_code_help": "  'analyze code [file]' - Analyze code file",
                "search_file_help": "  'search file [name]' - Search for files",
                "thinking": "M.I.A: Thinking...",
                "no_response": "Could not generate a response.",
                "no_input": "No input detected.",
                "llm_unavailable": "LLM is not available. Check configuration.",
                "processing_error": "Error processing: {error}",
                "exiting": "Exiting M.I.A. See you later!",
                "help_title": "\nM.I.A Commands",
                "help_separator": "----------------------------------------",
                "quit_help": "  quit   - Exit M.I.A",
                "help_help": "  help   - Show this help message",
                "audio_help": "  audio  - Switch to audio input mode",
                "text_help": "  text   - Switch to text input mode",
                "status_help": "  status - Show current system status",
                "models_help": "  models - List available models",
                "clear_help": "  clear  - Clear conversation context",
                "agent_title": "\nAgent Commands",
                "welcome_message": "Welcome to M.I.A - Multimodal Intelligent Assistant!",
                "cleanup_message": "Cleaning up resources...",
                "agent_file_created": "Agent: Created file: {filename}",
                "agent_note_saved": "Agent: Note saved to mia_notes.md",
                "agent_code_analysis": "Agent: {result}",
                "agent_file_error": "Agent: Error creating file: {error}",
                "agent_note_error": "Agent: Error creating note: {error}",
                "agent_analysis_error": "Agent: Error analyzing code: {error}",
                "agent_search_error": "Agent: Error searching file: {error}",
                "agent_specify_file": "Agent: Please specify a file to analyze (e.g., 'analyze code main.py')",
                "agent_specify_search": "Agent: Please specify a filename to search",
                "file_created_timestamp": "File created by M.I.A Agent on {timestamp}",
                "status_connected": "LLM Connected",
                "status_issues": "LLM Connection Issues",
                "audio_waiting_hotword": "Say '{hotword}' to activate",
                "audio_hotword_timeout": "Hotword not detected in time.",
                "audio_push_to_talk": "Press and hold space to speak",
                "audio_push_to_talk_cancel": "Capture cancelled.",
                "audio_listening": "Listening... (Ctrl+C to switch to text)",
                "audio_no_speech": "No speech detected.",
                "audio_transcription_failed": "Could not transcribe audio.",
                "audio_hotword_detected": "Hotword detected ({confidence}%)",
            },
            "pt": {
                "app_title": "M.I.A - Assistente Inteligente Multimodal",
                "app_version": "Versão: 0.1.0",
                "initializing": "Inicializando componentes...",
                "ollama_warning": "Aviso: Não foi possível conectar ao Ollama. Tentando usar provedor alternativo de LLM.",
                "ollama_success": "Conectado ao Ollama com sucesso!",
                "commands_available": "\nComandos disponíveis:",
                "help_command": "  'help' - Mostrar ajuda",
                "exit_command": "  'exit' - Sair",
                "agent_commands": "\nComandos do Agente:",
                "create_file_help": "  'criar arquivo [nome]' - Criar arquivo",
                "make_note_help": "  'fazer nota [título]' - Criar nota",
                "analyze_code_help": "  'analisar código [arquivo]' - Analisar código",
                "search_file_help": "  'buscar arquivo [nome]' - Buscar arquivo",
                "thinking": "M.I.A: Pensando...",
                "no_response": "Não foi possível gerar uma resposta.",
                "no_input": "Nenhuma entrada detectada.",
                "llm_unavailable": "LLM não está disponível. Verifique a configuração.",
                "processing_error": "Erro ao processar: {error}",
                "exiting": "Saindo do M.I.A. Até logo!",
                "help_title": "\nComandos M.I.A",
                "help_separator": "────────────────────────────────────────",
                "quit_help": "  quit   - Sair do M.I.A",
                "help_help": "  help   - Mostrar esta mensagem de ajuda",
                "audio_help": "  audio  - Alternar para modo de entrada de áudio",
                "text_help": "  text   - Alternar para modo de entrada de texto",
                "status_help": "  status - Mostrar status atual do sistema",
                "models_help": "  models - Listar modelos disponíveis",
                "clear_help": "  clear  - Limpar contexto da conversa",
                "agent_title": "\n Comandos do Agente",
                "welcome_message": "Bem-vinda à M.I.A - Assistente Inteligente Multimodal!",
                "cleanup_message": "Limpando recursos...",
                "agent_file_created": "Agente: Arquivo criado: {filename}",
                "agent_note_saved": "Agente: Nota salva em mia_notes.md",
                "agent_code_analysis": "Agente: {result}",
                "agent_file_error": "Agente: Erro ao criar arquivo: {error}",
                "agent_note_error": "Agente: Erro ao criar nota: {error}",
                "agent_analysis_error": "Agente: Erro ao analisar código: {error}",
                "agent_search_error": "Agente: Erro ao buscar arquivo: {error}",
                "agent_specify_file": "Agente: Por favor, especifique um arquivo para analisar (ex: 'analisar código main.py')",
                "agent_specify_search": "Agente: Por favor, especifique um nome de arquivo para buscar",
                "file_created_timestamp": "Arquivo criado pelo Agente M.I.A em {timestamp}",
                "status_connected": "LLM Conectado",
                "status_issues": "Problemas de Conexão LLM",
                "audio_waiting_hotword": "Diga '{hotword}' para ativar",
                "audio_hotword_timeout": "Hotword não detectada a tempo.",
                "audio_push_to_talk": "Pressione e segure espaço para falar",
                "audio_push_to_talk_cancel": "Captura cancelada.",
                "audio_listening": "Escutando... (Ctrl+C para texto)",
                "audio_no_speech": "Nenhuma fala detectada.",
                "audio_transcription_failed": "Não foi possível transcrever o áudio.",
                "audio_hotword_detected": "Hotword detectada ({confidence}%)",
            },
        }
        return strings.get(self.language, strings["en"])

    def get(self, key: str, **kwargs) -> str:
        """
        Get localized string.

        Args:
            key: String key
            **kwargs: Format arguments

        Returns:
            Localized string with format arguments applied
        """
        text = self.strings.get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, ValueError):
                return text
        return text

    def set_language(self, language: str):
        """
        Change language.

        Args:
            language: Language code ("en" or "pt")
        """
        if language.lower() in ["en", "pt"]:
            self.language = language.lower()
            self.strings = self._load_strings()


# Global localization instance
_localization = None


def init_localization(language: Optional[str] = None) -> Localization:
    """
    Initialize global localization.

    Args:
        language: Language code, or None to use environment variable

    Returns:
        Localization instance
    """
    global _localization

    if language is None:
        # Check environment variable
        language = os.getenv("MIA_LANGUAGE", "en")

    _localization = Localization(language)
    return _localization


def get_localization() -> Localization:
    """Get current localization instance."""
    global _localization
    if _localization is None:
        _localization = Localization()
    return _localization


def _(key: str, **kwargs) -> str:
    """
    Quick access to localized strings.

    Args:
        key: String key
        **kwargs: Format arguments

    Returns:
        Localized string
    """
    return get_localization().get(key, **kwargs)
