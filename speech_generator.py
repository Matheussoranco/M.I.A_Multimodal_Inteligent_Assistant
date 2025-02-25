from memory.knowledge_graph import AgentMemory

class SpeechGenerator:
    def __init__(self, device, llama_model_path=None):
        self.memory = AgentMemory()
        # Rest of existing initialization

    def synthesize_audio(self, text):
        """Enhanced with contextual voice modulation"""
        context = self.memory.retrieve_context(text_embedding)
        female_speaker_id = 7317 if "urgent" in context else 7306
        # Rest of existing synthesis logic