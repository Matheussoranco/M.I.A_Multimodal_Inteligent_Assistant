import speech_recognition as sr
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional

class MultimodalProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.vision_cache: Dict[str, Any] = {}
        
    def process_audio(self, audio_data) -> Dict[str, Any]:
        """Convert speech to text with emotion analysis"""
        try:
            # Use Google Speech Recognition as primary method
            text = getattr(self.recognizer, 'recognize_google')(audio_data)  # type: ignore
            return {
                'text': text,
                'emotion': self._analyze_emotion(audio_data)
            }
        except (sr.UnknownValueError, AttributeError):
            return {"error": "Could not understand audio"}
        except sr.RequestError:
            # Try alternative recognition methods
            try:
                # Try Sphinx (offline) as fallback
                text = getattr(self.recognizer, 'recognize_sphinx')(audio_data)  # type: ignore
                return {
                    'text': text,
                    'emotion': self._analyze_emotion(audio_data)
                }
            except (sr.UnknownValueError, sr.RequestError, AttributeError):
                return {"error": "Could not process audio"}
            
    def process_image(self, image_path):
        """Analyze image content"""
        img = Image.open(image_path)
        return {
            'size': img.size,
            'dominant_color': self._get_dominant_color(img),
            'text_ocr': self._extract_text(img)
        }
    
    def _analyze_emotion(self, audio):
        """Analyze emotion from audio using basic signal processing"""
        try:
            # Convert audio to numpy array
            if hasattr(audio, 'get_raw_data'):
                audio_np = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            else:
                audio_np = np.array(audio, dtype=np.float32)
            
            # Convert to float and normalize
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            
            # Basic feature extraction
            # Energy (volume level)
            energy = np.sum(audio_np ** 2) / len(audio_np)
            
            # Zero crossing rate (measure of noisiness)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / len(audio_np)
            
            # Simple pitch estimation using autocorrelation
            if len(audio_np) > 100:
                corr = np.correlate(audio_np[:1000], audio_np[:1000], mode='full')
                corr = corr[len(corr)//2:]
                if len(corr) > 10:
                    peak_index = np.argmax(corr[10:100]) + 10
                    pitch_estimate = 16000 / peak_index if peak_index > 0 else 0
                else:
                    pitch_estimate = 0
            else:
                pitch_estimate = 0
            
            # Simple emotion classification based on features
            if energy > 0.05:  # High energy
                if zero_crossings > 0.2:  # Noisy
                    return "excited"
                else:
                    return "angry"
            elif energy < 0.001:  # Low energy
                return "sad"
            elif pitch_estimate > 200:  # High pitch
                return "happy"
            elif zero_crossings > 0.15:  # Moderately noisy
                return "anxious"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Emotion analysis failed: {e}")
            return "neutral"

    def _get_dominant_color(self, img):
        """Extract dominant color from image using k-means clustering"""
        try:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for faster processing
            img = img.resize((100, 100))
            
            # Get pixel data
            pixels = np.array(img)
            pixels = pixels.reshape(-1, 3)
            
            # Simple approach: find most frequent color
            # Convert to tuples for counting
            pixel_tuples = [tuple(pixel) for pixel in pixels]
            
            # Count occurrences
            from collections import Counter
            color_counts = Counter(pixel_tuples)
            
            # Get most common color
            dominant_color = color_counts.most_common(1)[0][0]
            
            # Convert to hex format
            hex_color = '#{:02x}{:02x}{:02x}'.format(*dominant_color)
            
            # Also return RGB values
            return {
                'hex': hex_color,
                'rgb': dominant_color,
                'name': self._color_name(dominant_color)
            }
            
        except Exception as e:
            print(f"Dominant color extraction failed: {e}")
            return {'hex': '#808080', 'rgb': (128, 128, 128), 'name': 'unknown'}

    def _color_name(self, rgb):
        """Get approximate color name from RGB values"""
        r, g, b = rgb
        
        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g + b:
            return "red"
        elif g > r + b:
            return "green"
        elif b > r + g:
            return "blue"
        elif abs(r - g) < 30 and abs(r - b) < 30 and abs(g - b) < 30:
            if r > 150:
                return "gray"
            else:
                return "dark gray"
        elif r > 150 and g > 100:
            return "yellow"
        elif r > 150 and b > 100:
            return "magenta"
        elif g > 150 and b > 100:
            return "cyan"
        else:
            return "unknown"

    def _extract_text(self, img):
        """Extract text from image using OCR"""
        try:
            # Try to use pytesseract if available
            try:
                import pytesseract
                # Convert PIL to string format pytesseract expects
                text = pytesseract.image_to_string(img)
                return text.strip() if text.strip() else ""
            except ImportError:
                # Fallback: simple OCR using PIL (very basic)
                return self._basic_ocr(img)
        except Exception as e:
            print(f"OCR text extraction failed: {e}")
            return ""

    def _basic_ocr(self, img):
        """Basic OCR fallback using PIL image analysis"""
        try:
            # This is a very simple fallback - in practice, you'd want proper OCR
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            
            # Get image dimensions
            width, height = img.size
            
            # Simple text detection by looking for high contrast areas
            # This is just a placeholder - real OCR would be much more sophisticated
            pixels = list(img.getdata())
            
            # Calculate basic statistics
            mean_brightness = sum(pixels) / len(pixels)
            contrast = sum(abs(p - mean_brightness) for p in pixels) / len(pixels)
            
            if contrast > 30:  # High contrast might indicate text
                return "[Text detected - OCR not available. Install pytesseract for full OCR functionality]"
            else:
                return ""
                
        except Exception as e:
            print(f"Basic OCR failed: {e}")
            return ""

# Usage:
# processor = MultimodalProcessor()
# audio_analysis = processor.process_audio(audio_data)