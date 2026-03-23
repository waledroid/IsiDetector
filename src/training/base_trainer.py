import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path

# Import the Hook registry so the base trainer can trigger them
from src.shared.registry import HOOKS

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """
    The universal contract for all model trainers in the Parcel-Sense project.
    Handles configuration, directory management, and the Hook execution lifecycle.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get('model_type', 'unknown_model')
        
        # 1. Setup Output Directories
        self.output_dir = Path(config.get('output_dir', 'models')) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Universal State Variables (Populated during training)
        self.model = None
        self.current_epoch = 0
        self.current_loss = 0.0
        
        # 3. Initialize Hooks dynamically from the YAML config
        self.hooks = []
        hook_names = config.get('hooks', [])
        for h_name in hook_names:
            try:
                hook_class = HOOKS.get(h_name)
                self.hooks.append(hook_class())
                logger.debug(f"🪝 Attached Hook: {h_name}")
            except KeyError:
                logger.error(f"❌ Hook '{h_name}' requested in config but not found in Registry.")

    def call_hooks(self, stage: str):
        """
        Broadcasts the current stage to all registered hooks.
        Stages: 'before_train', 'before_epoch', 'after_epoch', 'after_train'
        """
        for hook in self.hooks:
            # Check if the hook has a method for this specific stage
            if hasattr(hook, stage):
                hook_method = getattr(hook, stage)
                hook_method(self)  # Pass the trainer instance to the hook so it can read metrics

    # ==========================================
    # ENFORCED METHODS (Children MUST implement)
    # ==========================================

    @abstractmethod
    def build_model(self):
        """Initializes the architecture and loads it to the GPU."""
        pass

    @abstractmethod
    def train(self):
        """The main training loop. MUST include self.call_hooks() at appropriate stages."""
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        """Runs validation and returns a dictionary of metrics (mAP, etc)."""
        pass

    @abstractmethod
    def export(self, format: str = 'onnx'):
        """Exports the trained weights to the deployment format (OpenVINO/ONNX)."""
        pass
