from multimodal.processor import MultimodalProcessor

class SpeechProcessor:
    def __init__(self, device, stt_model):
        self.multimodal = MultimodalProcessor()
        self.device = device
        # Rest of existing initialization

    def transcribe_audio(self, audio_data):
        """Enhanced with multimodal context"""
        result = self.multimodal.process_audio(audio_data)
        if 'error' in result:
            raise AudioProcessingError(result['error'])
        return {
            'text': result['text'],
            'emotion': result['emotion'],
            'partial': [False]
        }