# src/inference/base_inferencer.py
from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseInferencer(ABC):
    """
    The blueprint for all inference engines. 
    Keeps the main application clean and model-agnostic.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Weights not found at {self.model_path}")
            
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self):
        """Loads the model into RAM/VRAM."""
        pass

    @abstractmethod
    def predict(self, source: str, show: bool = False, save: bool = False):
        """Runs the inference loop (must yield results one by one for video streams)."""
        pass

    @abstractmethod
    def get_summary(self, result) -> dict:
        """Translates the model's complex output into a simple dictionary."""
        pass
