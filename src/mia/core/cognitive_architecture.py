import torch
import warnings
from typing import Dict, Any, Optional, Union

# Import custom exceptions and error handling
from ..exceptions import VisionProcessingError, InitializationError, ValidationError
from ..error_handler import global_error_handler, with_error_handling, safe_execute

# Suppress transformers warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*slow.*processor.*")
    warnings.filterwarnings("ignore", message=".*use_fast.*")
    try:
        from transformers import CLIPProcessor, CLIPModel
        HAS_CLIP = True
    except ImportError:
        CLIPProcessor = None
        CLIPModel = None
        HAS_CLIP = False

import logging
logger = logging.getLogger(__name__)

class MIACognitiveCore:
    def __init__(self, llm_client, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.llm = llm_client
        self.device = device
        self.vision_processor: Optional[Any] = None
        self.vision_model: Optional[Any] = None
        
        # Initialize vision components with proper error handling
        self._init_vision_components()
        
        self.working_memory = []
        
    def _init_vision_components(self):
        """Initialize vision components with comprehensive error handling."""
        if not HAS_CLIP:
            logger.warning("CLIP components not available - vision processing disabled")
            return
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                warnings.filterwarnings("ignore", message=".*slow.*processor.*")
                warnings.filterwarnings("ignore", message=".*use_fast.*")
                
                if CLIPProcessor is None or CLIPModel is None:
                    raise InitializationError("CLIP components not imported", "IMPORT_ERROR")
                
                try:
                    self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                except Exception as e:
                    raise InitializationError(f"Failed to load CLIP processor: {str(e)}", 
                                           "PROCESSOR_LOAD_FAILED")
                
                try:
                    self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                except Exception as e:
                    raise InitializationError(f"Failed to load CLIP model: {str(e)}", 
                                           "MODEL_LOAD_FAILED")
                
                # Move model to device if available
                if self.vision_model is not None and hasattr(self.vision_model, 'to'):
                    try:
                        self.vision_model = self.vision_model.to(self.device)
                        logger.info(f"Vision model moved to device: {self.device}")
                    except Exception as e:
                        logger.warning(f"Failed to move vision model to device {self.device}: {e}")
                        
                logger.info("Vision components initialized successfully")
                        
        except InitializationError:
            raise
        except Exception as e:
            raise InitializationError(f"Unexpected error initializing vision components: {str(e)}", 
                                   "VISION_INIT_FAILED")
        
    @with_error_handling(global_error_handler, fallback_value={"error": "Processing failed"})
    def process_multimodal_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text/image/audio inputs with comprehensive error handling."""
        if not inputs:
            raise ValidationError("Empty inputs provided", "EMPTY_INPUT")
            
        processed = {}
        
        try:
            if 'image' in inputs:
                processed['vision'] = self._process_image(inputs['image'])
                
            if 'audio' in inputs:
                processed['text'] = self._transcribe_audio(inputs['audio'])
                
            if 'text' in inputs:
                processed['text'] = self._analyze_text(inputs['text'])
                
            return self._reasoning_pipeline(processed)
            
        except Exception as e:
            logger.error(f"Multimodal processing error: {e}")
            if isinstance(e, (ValidationError, VisionProcessingError)):
                raise e
            raise ValidationError(f"Multimodal processing failed: {str(e)}", "PROCESSING_FAILED")
    
    def _process_image(self, image) -> Dict[str, Any]:
        """Process image using CLIP vision model with comprehensive error handling."""
        if self.vision_processor is None or self.vision_model is None:
            raise VisionProcessingError("Vision processing not available", "VISION_NOT_AVAILABLE")
        
        try:
            # Validate image input
            if image is None:
                raise VisionProcessingError("Image input is None", "NULL_IMAGE")
                
            if not hasattr(image, 'size'):  # Check if it's a PIL Image
                raise VisionProcessingError("Invalid image format - expected PIL Image", "INVALID_FORMAT")
            
            # Process image with CLIP processor
            try:
                inputs = self.vision_processor(images=image, return_tensors="pt")
            except Exception as e:
                raise VisionProcessingError(f"CLIP processor failed: {str(e)}", "PROCESSOR_ERROR")
            
            # Move inputs to device if possible
            try:
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, dict):
                    # Move tensor values to device
                    for key, value in inputs.items():
                        if hasattr(value, 'to'):
                            inputs[key] = value.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to move inputs to device: {e}")
            
            # Get image features
            try:
                with torch.no_grad():
                    image_features = self.vision_model.get_image_features(**inputs)
            except Exception as e:
                raise VisionProcessingError(f"Feature extraction failed: {str(e)}", "FEATURE_EXTRACTION_ERROR")
            
            return {
                "features": image_features, 
                "success": True,
                "device": str(self.device)
            }
            
        except VisionProcessingError:
            raise
        except Exception as e:
            raise VisionProcessingError(f"Unexpected vision processing error: {str(e)}", 
                                      "UNEXPECTED_ERROR")
    
    def _transcribe_audio(self, audio_data):
        """Transcribe audio to text - placeholder for audio processing."""
        # This should integrate with speech_processor
        return f"Transcribed: {audio_data}"
        
    def _analyze_text(self, text):
        """Analyze text input."""
        return text
        
    def _reasoning_pipeline(self, context):
        """Chain-of-Thought reasoning with visual grounding"""
        prompt = f"""Analyze this multimodal context:
        {context}
        
        Perform step-by-step reasoning considering:
        1. Visual elements in the scene
        2. Historical context from memory
        3. Possible action paths"""
        
        # Use proper LLM query method
        try:
            if hasattr(self.llm, 'query'):
                response = self.llm.query(prompt)
            elif hasattr(self.llm, 'query_model'):
                response = self.llm.query_model(prompt)
            else:
                # Fallback for different LLM interfaces
                response = str(self.llm)
                
            return {
                'text': response or "No response generated",
                'embedding': []  # Placeholder for embedding
            }
        except Exception as e:
            return {
                'text': f"Error in reasoning pipeline: {e}",
                'embedding': []
            }

# Usage:
# core = MIACognitiveCore(llm_client)
# result = core.process_multimodal_input({'image': image_pil, 'text': "Describe this scene"})