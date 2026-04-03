
import logging
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal, QObject

# --- Production-Ready API Imports ---
# These libraries must be installed in your environment:
# pip install google-generativeai openai
try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPICallError, PermissionDenied
except ImportError:
    genai = None
    GoogleAPICallError = None
    PermissionDenied = None

try:
    from openai import OpenAI, APIError, AuthenticationError, RateLimitError
except ImportError:
    OpenAI = None
    APIError = None
    AuthenticationError = None
    RateLimitError = None


logger = logging.getLogger(__name__)

class AIQueryWorker(QThread):
    """
    Worker thread for sending queries to an AI model API in the background.
    Supports response streaming, cancellation, and detailed error handling.
    """
    # Emits a chunk of the AI's response as it's being generated.
    partial_result_ready = pyqtSignal(str)
    
    # Emits the complete, final response from the AI.
    finished_result_ready = pyqtSignal(str)
    
    # Emits a user-friendly error message if the query fails.
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 api_key: str,
                 model_name: str,
                 prompt: str,
                 service_type: str,
                 base_url: Optional[str] = None,
                 parent: Optional[QObject] = None):
        """
        Args:
            api_key: The API key for the selected service.
            model_name: The specific model to query (e.g., "gemini-1.5-pro-latest").
            prompt: The user's query or instruction for the AI.
            service_type: The AI service to use ("gemini", "openai", "openrouter").
            base_url: The base URL for services like OpenAI or OpenRouter.
            parent: Optional QObject parent.
        """
        super().__init__(parent)
        self.api_key = api_key
        self.model_name = model_name
        self.prompt = prompt
        self.service_type = service_type.lower()
        self.base_url = base_url
        
        self._is_running = True
        self._full_response_text = ""

    def stop(self):
        """
        Requests the worker to stop processing. This will interrupt the
        streaming of a response and prevent further signals from being emitted.
        """
        logger.info(f"Stop requested for AI worker (model: {self.model_name}).")
        self._is_running = False

    def run(self):
        """
        Executes the AI query by dispatching to the appropriate service handler.
        """
        if not self._is_running:
            logger.warning("AI worker run aborted as it was stopped before starting.")
            return

        logger.info(f"AI worker started for service: '{self.service_type}', model: '{self.model_name}'")
        
        try:
            if not self.api_key:
                raise ValueError("API key is missing. Please configure it in the settings.")

            if self.service_type == "gemini":
                self._run_gemini_query()
            elif self.service_type in ["openai", "openrouter"]:
                self._run_openai_compatible_query()
            else:
                raise ValueError(f"Unsupported AI service type: '{self.service_type}'")

            if self._is_running:
                logger.info(f"AI query to {self.model_name} completed successfully.")
                self.finished_result_ready.emit(self._full_response_text)

        except (ValueError, ImportError) as e:
            logger.warning(f"AI worker configuration error: {e}")
            if self._is_running:
                self.error_occurred.emit(str(e))
        except Exception as e:
            logger.error(f"An unexpected error occurred during AI query to {self.model_name}: {e}", exc_info=True)
            if self._is_running:
                self.error_occurred.emit(f"An unexpected error occurred: {str(e)}")
        finally:
            if not self._is_running:
                logger.info(f"AI worker (model: {self.model_name}) was stopped during execution.")

    def _run_gemini_query(self):
        """Handles the query to the Google Gemini API."""
        if not genai:
            raise ImportError("Google Generative AI library not found. Please run 'pip install google-generativeai'.")

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response_stream = model.generate_content(self.prompt, stream=True)
            
            for chunk in response_stream:
                if not self._is_running:
                    # If stop() was called, break the loop immediately.
                    response_stream.close()
                    break
                
                if chunk.text:
                    self.partial_result_ready.emit(chunk.text)
                    self._full_response_text += chunk.text

        except PermissionDenied as e:
            raise ValueError(f"Invalid Gemini API Key. Please check your credentials. Details: {e.args[0]}")
        except GoogleAPICallError as e:
            raise RuntimeError(f"A Google API error occurred: {e.args[0]}")

    def _run_openai_compatible_query(self):
        """Handles queries to OpenAI and OpenRouter APIs."""
        if not OpenAI:
            raise ImportError("OpenAI library not found. Please run 'pip install openai'.")
        
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": self.prompt}],
                stream=True,
            )
            
            for chunk in stream:
                if not self._is_running:
                    # If stop() was called, break the loop immediately.
                    stream.close()
                    break
                
                content = chunk.choices[0].delta.content
                if content:
                    self.partial_result_ready.emit(content)
                    self._full_response_text += content

        except AuthenticationError:
            raise ValueError(f"Invalid API Key for {self.service_type.capitalize()}. Please check your credentials.")
        except RateLimitError:
            raise RuntimeError("API rate limit exceeded. Please wait and try again, or check your plan and usage details.")
        except APIError as e:
            # Catches other API-related errors (e.g., model not found, server errors)
            raise RuntimeError(f"An API error occurred: {e.message}")