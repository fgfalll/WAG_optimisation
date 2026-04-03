import logging
import httpx
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

class ModelListWorker(QThread):
    models_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_key: str, base_url: str, endpoint: str, service_type: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.endpoint = endpoint
        self.service_type = service_type

    def run(self):
        try:
            url = f"{self.base_url.rstrip('/')}{self.endpoint}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            with httpx.Client() as client:
                response = client.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
            data = response.json()
            
            if self.service_type == "openai" or self.service_type == "openrouter":
                models = sorted([m["id"] for m in data.get("data", [])])
                self.models_ready.emit(models)
            else:
                # Handle other services if needed
                self.error_occurred.emit(f"Model fetching for '{self.service_type}' is not implemented.")

        except httpx.RequestError as e:
            logger.error(f"HTTP request error while fetching models: {e}")
            self.error_occurred.emit(f"Network error: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP status error while fetching models: {e}")
            self.error_occurred.emit(f"API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching models: {e}")
            self.error_occurred.emit(f"An unexpected error occurred: {e}")
