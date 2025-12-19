"""
Document Intelligence: OCR + LLM integration for intelligent document processing.
Combines OCR text extraction with LLM-powered analysis and structured data extraction.
"""

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from ..config_manager import ConfigManager
from ..exceptions import ConfigurationError, InitializationError
from ..llm.llm_manager import LLMManager
from ..multimodal.ocr_processor import OCRProcessor, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class DocumentAnalysisResult:
    """Result from document analysis."""
    
    raw_text: str
    structured_data: Dict[str, Any]
    summary: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    key_value_pairs: Dict[str, str] = field(default_factory=dict)
    tables: List[List[List[str]]] = field(default_factory=list)
    confidence: float = 0.0
    document_type: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_text": self.raw_text,
            "structured_data": self.structured_data,
            "summary": self.summary,
            "entities": self.entities,
            "key_value_pairs": self.key_value_pairs,
            "tables": self.tables,
            "confidence": self.confidence,
            "document_type": self.document_type,
            "language": self.language,
            "metadata": self.metadata,
        }


class DocumentIntelligence:
    """
    Intelligent document processing combining OCR and LLM capabilities.
    
    This class provides advanced document understanding by:
    1. Extracting text using OCR (multiple engine support)
    2. Using LLM to understand and structure the content
    3. Enabling function calling for automated workflows
    """

    # Document type detection prompts
    DOCUMENT_TYPES = {
        "invoice": ["invoice", "bill", "total", "amount due", "payment"],
        "receipt": ["receipt", "transaction", "purchased", "store", "cash"],
        "contract": ["agreement", "contract", "parties", "terms", "signed"],
        "form": ["form", "please fill", "applicant", "date of birth"],
        "letter": ["dear", "sincerely", "regards", "letter"],
        "report": ["report", "analysis", "findings", "conclusion"],
        "id_card": ["id", "identification", "passport", "license", "birth"],
        "resume": ["resume", "cv", "experience", "education", "skills"],
        "menu": ["menu", "appetizer", "main course", "dessert", "price"],
    }

    def __init__(
        self,
        ocr_processor: Optional[OCRProcessor] = None,
        llm_manager: Optional[LLMManager] = None,
        config_manager: Optional[ConfigManager] = None,
        auto_init: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize DocumentIntelligence.
        
        Args:
            ocr_processor: Pre-configured OCR processor
            llm_manager: Pre-configured LLM manager
            config_manager: Configuration manager
            auto_init: Whether to auto-initialize components
        """
        self.config_manager = config_manager or ConfigManager()
        self._ocr: Optional[OCRProcessor] = ocr_processor
        self._llm: Optional[LLMManager] = llm_manager
        
        # Function registry for LLM function calling
        self._functions: Dict[str, Callable] = {}
        self._function_schemas: List[Dict[str, Any]] = []
        
        # Register default functions
        self._register_default_functions()
        
        if auto_init:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure OCR and LLM are initialized."""
        import sys
        is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "true"
        
        if self._ocr is None:
            try:
                self._ocr = OCRProcessor(
                    config_manager=self.config_manager,
                    auto_detect=not is_testing,
                )
            except Exception as e:
                logger.warning(f"OCR initialization failed: {e}")
        
        if self._llm is None:
            try:
                self._llm = LLMManager(
                    config_manager=self.config_manager,
                    auto_detect=not is_testing,
                )
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")

    @property
    def ocr(self) -> Optional[OCRProcessor]:
        """Get the OCR processor."""
        return self._ocr

    @property
    def llm(self) -> Optional[LLMManager]:
        """Get the LLM manager."""
        return self._llm

    def _register_default_functions(self) -> None:
        """Register default functions for LLM function calling."""
        self.register_function(
            name="extract_key_values",
            description="Extract key-value pairs from document text",
            parameters={
                "type": "object",
                "properties": {
                    "pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                            },
                        },
                    },
                },
            },
            function=self._extract_key_values,
        )
        
        self.register_function(
            name="extract_entities",
            description="Extract named entities (people, organizations, dates, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                        },
                    },
                },
            },
            function=self._extract_entities,
        )
        
        self.register_function(
            name="extract_table",
            description="Extract tabular data from document",
            parameters={
                "type": "object",
                "properties": {
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            function=self._extract_table,
        )

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable,
    ) -> None:
        """Register a function for LLM function calling.
        
        Args:
            name: Function name
            description: Function description
            parameters: JSON schema for parameters
            function: The callable function
        """
        self._functions[name] = function
        self._function_schemas.append({
            "name": name,
            "description": description,
            "parameters": parameters,
        })

    def _extract_key_values(self, pairs: List[Dict[str, str]]) -> Dict[str, str]:
        """Extract key-value pairs helper."""
        return {p.get("key", ""): p.get("value", "") for p in pairs if p.get("key")}

    def _extract_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entities helper."""
        return entities

    def _extract_table(
        self, headers: List[str], rows: List[List[str]]
    ) -> List[List[str]]:
        """Extract table helper."""
        return [headers] + rows

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        document_type: Optional[str] = None,
        extract_tables: bool = True,
        extract_entities: bool = True,
        summarize: bool = True,
        custom_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> DocumentAnalysisResult:
        """Analyze a document image with OCR and LLM.
        
        Args:
            image: Image path or PIL Image
            document_type: Type of document (auto-detected if not provided)
            extract_tables: Whether to extract tables
            extract_entities: Whether to extract named entities
            summarize: Whether to generate a summary
            custom_prompt: Custom extraction prompt
            schema: JSON schema for structured extraction
            
        Returns:
            DocumentAnalysisResult with extracted data
        """
        self._ensure_initialized()
        
        if self._ocr is None:
            raise InitializationError(
                "OCR processor not available",
                "OCR_NOT_AVAILABLE",
            )
        
        # Step 1: Extract text using OCR
        ocr_result = self._ocr.extract_text(image, return_boxes=True)
        raw_text = ocr_result.text if isinstance(ocr_result, OCRResult) else ocr_result
        confidence = ocr_result.confidence if isinstance(ocr_result, OCRResult) else 1.0
        
        if not raw_text.strip():
            return DocumentAnalysisResult(
                raw_text="",
                structured_data={},
                confidence=0.0,
                metadata={"error": "No text extracted from image"},
            )
        
        # Step 2: Detect document type if not provided
        if document_type is None or document_type == "auto":
            document_type = self._detect_document_type(raw_text)
        
        # Step 3: Use LLM for intelligent extraction
        result = DocumentAnalysisResult(
            raw_text=raw_text,
            structured_data={},
            confidence=confidence,
            document_type=document_type,
        )
        
        if self._llm is None or not self._llm._available:
            # Return basic OCR result without LLM enhancement
            result.metadata["llm_enhanced"] = False
            return result
        
        # Step 4: Extract structured data with LLM
        if schema:
            structured = self._extract_with_schema(raw_text, schema)
            result.structured_data = structured
        elif custom_prompt:
            structured = self._extract_with_prompt(raw_text, custom_prompt)
            result.structured_data = structured
        else:
            # Use document-type specific extraction
            structured = self._extract_by_document_type(raw_text, document_type)
            result.structured_data = structured
        
        # Step 5: Extract additional features
        if extract_entities:
            result.entities = self._llm_extract_entities(raw_text)
        
        if extract_tables:
            result.tables = self._llm_extract_tables(raw_text)
        
        if summarize:
            result.summary = self._llm_summarize(raw_text, document_type)
        
        # Extract key-value pairs
        result.key_value_pairs = self._llm_extract_key_values(raw_text)
        
        result.metadata["llm_enhanced"] = True
        result.metadata["ocr_provider"] = self._ocr.provider
        result.metadata["llm_provider"] = self._llm.provider
        
        return result

    def _detect_document_type(self, text: str) -> str:
        """Detect document type from text content."""
        text_lower = text.lower()
        
        scores = {}
        for doc_type, keywords in self.DOCUMENT_TYPES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            return max(scores, key=lambda k: scores[k])
        return "general"

    def _extract_with_schema(
        self, text: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract data according to JSON schema."""
        prompt = f"""Extract information from the following document text according to the JSON schema provided.

Document Text:
{text}

JSON Schema:
{json.dumps(schema, indent=2)}

Return the extracted data as valid JSON matching the schema.
Only return the JSON, no additional text."""
        
        response = self._llm.query(prompt)  # type: ignore[union-attr]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_extraction": response}

    def _extract_with_prompt(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract data using custom prompt."""
        full_prompt = f"""{prompt}

Document Text:
{text}

Return the extracted data as JSON."""
        
        response = self._llm.query(full_prompt)  # type: ignore[union-attr]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_extraction": response}

    def _extract_by_document_type(
        self, text: str, document_type: str
    ) -> Dict[str, Any]:
        """Extract data based on document type."""
        schemas = {
            "invoice": {
                "invoice_number": "string",
                "date": "string",
                "due_date": "string",
                "vendor_name": "string",
                "vendor_address": "string",
                "customer_name": "string",
                "customer_address": "string",
                "line_items": [{"description": "string", "quantity": "number", "unit_price": "number", "total": "number"}],
                "subtotal": "number",
                "tax": "number",
                "total": "number",
                "payment_terms": "string",
            },
            "receipt": {
                "store_name": "string",
                "store_address": "string",
                "date": "string",
                "time": "string",
                "items": [{"name": "string", "quantity": "number", "price": "number"}],
                "subtotal": "number",
                "tax": "number",
                "total": "number",
                "payment_method": "string",
                "transaction_id": "string",
            },
            "id_card": {
                "document_type": "string",
                "full_name": "string",
                "date_of_birth": "string",
                "id_number": "string",
                "nationality": "string",
                "issue_date": "string",
                "expiry_date": "string",
                "issuing_authority": "string",
            },
            "resume": {
                "name": "string",
                "email": "string",
                "phone": "string",
                "address": "string",
                "summary": "string",
                "experience": [{"title": "string", "company": "string", "period": "string", "description": "string"}],
                "education": [{"degree": "string", "institution": "string", "year": "string"}],
                "skills": ["string"],
            },
            "contract": {
                "contract_type": "string",
                "parties": [{"name": "string", "role": "string"}],
                "effective_date": "string",
                "expiry_date": "string",
                "key_terms": ["string"],
                "obligations": ["string"],
                "signatures": [{"name": "string", "date": "string"}],
            },
        }
        
        schema = schemas.get(document_type, {
            "title": "string",
            "content_summary": "string",
            "key_information": ["string"],
            "dates": ["string"],
            "amounts": ["string"],
            "names": ["string"],
        })
        
        return self._extract_with_schema(text, {"type": "object", "properties": schema})

    def _llm_extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using LLM."""
        prompt = f"""Extract all named entities from the following text. 
Identify: PERSON, ORGANIZATION, DATE, MONEY, LOCATION, EMAIL, PHONE, ID_NUMBER.

Text:
{text}

Return as JSON array: [{{"text": "entity text", "type": "ENTITY_TYPE"}}]
Only return the JSON array, no additional text."""
        
        response = self._llm.query(prompt)  # type: ignore[union-attr]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def _llm_extract_tables(self, text: str) -> List[List[List[str]]]:
        """Extract tabular data using LLM."""
        prompt = f"""If the following text contains any tabular data, extract it.

Text:
{text}

Return as JSON array of tables, where each table is an array of rows, and each row is an array of cells.
Example: [[["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]]
If no tables found, return empty array: []
Only return the JSON array, no additional text."""
        
        response = self._llm.query(prompt)  # type: ignore[union-attr]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    def _llm_summarize(self, text: str, document_type: str) -> str:
        """Generate document summary using LLM."""
        prompt = f"""Summarize the following {document_type} document in 2-3 sentences.
Focus on the most important information.

Text:
{text}

Summary:"""
        
        return self._llm.query(prompt)  # type: ignore[union-attr]

    def _llm_extract_key_values(self, text: str) -> Dict[str, str]:
        """Extract key-value pairs using LLM."""
        prompt = f"""Extract all key-value pairs from the following text.
Look for patterns like "Label: Value" or "Field = Value".

Text:
{text}

Return as JSON object: {{"key1": "value1", "key2": "value2"}}
Only return the JSON object, no additional text."""
        
        response = self._llm.query(prompt)  # type: ignore[union-attr]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    def analyze_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        **kwargs: Any,
    ) -> List[DocumentAnalysisResult]:
        """Analyze multiple document images.
        
        Args:
            images: List of images to analyze
            **kwargs: Arguments passed to analyze_image
            
        Returns:
            List of DocumentAnalysisResult
        """
        results = []
        for image in images:
            try:
                result = self.analyze_image(image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze image: {e}")
                results.append(DocumentAnalysisResult(
                    raw_text="",
                    structured_data={},
                    confidence=0.0,
                    metadata={"error": str(e)},
                ))
        return results

    def compare_documents(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image],
    ) -> Dict[str, Any]:
        """Compare two documents and identify differences.
        
        Args:
            image1: First document image
            image2: Second document image
            
        Returns:
            Comparison results
        """
        # Analyze both documents
        result1 = self.analyze_image(image1)
        result2 = self.analyze_image(image2)
        
        if self._llm is None:
            return {
                "document1": result1.to_dict(),
                "document2": result2.to_dict(),
                "comparison": "LLM not available for comparison",
            }
        
        # Use LLM to compare
        prompt = f"""Compare the following two documents and identify key differences.

Document 1:
{result1.raw_text}

Document 2:
{result2.raw_text}

Provide:
1. Summary of similarities
2. Key differences
3. Which fields changed and how

Return as JSON:
{{
    "similarities": ["list of similarities"],
    "differences": ["list of differences"],
    "changed_fields": {{"field_name": {{"old": "value1", "new": "value2"}}}}
}}"""
        
        response = self._llm.query(prompt)
        
        try:
            comparison = json.loads(response)
        except json.JSONDecodeError:
            comparison = {"raw_comparison": response}
        
        return {
            "document1": result1.to_dict(),
            "document2": result2.to_dict(),
            "comparison": comparison,
        }

    def query_document(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
    ) -> str:
        """Ask a question about a document.
        
        Args:
            image: Document image
            question: Question to answer
            
        Returns:
            Answer to the question
        """
        self._ensure_initialized()
        
        if self._ocr is None:
            raise InitializationError("OCR not available", "OCR_NOT_AVAILABLE")
        
        # Extract text
        ocr_result = self._ocr.extract_text(image)
        text = ocr_result.text if isinstance(ocr_result, OCRResult) else ocr_result
        
        if self._llm is None:
            return f"Cannot answer question. LLM not available. Document text: {text[:500]}..."
        
        # Query with LLM
        prompt = f"""Based on the following document text, answer the question.

Document:
{text}

Question: {question}

Answer:"""
        
        return self._llm.query(prompt)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the DocumentIntelligence instance."""
        return {
            "ocr_available": self._ocr is not None and self._ocr.is_available,
            "llm_available": self._llm is not None and self._llm._available,
            "ocr_provider": self._ocr.provider if self._ocr else None,
            "llm_provider": self._llm.provider if self._llm else None,
            "registered_functions": list(self._functions.keys()),
            "supported_document_types": list(self.DOCUMENT_TYPES.keys()),
        }
