import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from src.training.base_trainer import BaseTrainer
from src.shared.registry import TRAINERS

try:
    # 🛑 CHANGED: Now importing the Segmentation specific models!
    from rfdetr import RFDETRSegMedium, RFDETRSegSmall
except ImportError:
    raise ImportError("❌ Missing dependency. Run: pip install rfdetr")

logger = logging.getLogger(__name__)

@TRAINERS.register('rfdetr')
class RFDETRTrainer(BaseTrainer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_size = config.get('model_size', 'm') 
        self.dataset_path = Path(config.get('dataset_path'))
        self.history = [] 
        
    def build_model(self):
        logger.info(f"🏗️ Initializing RF-DETR SEGMENTATION ({self.model_size}) with DINOv2...")
        if self.model_size == 'm':
            self.model = RFDETRSegMedium()
        else:
            self.model = RFDETRSegSmall()

    def train(self):
        if self.model is None:
            self.build_model()
            
        # 1. TIME-STAMPED LOGGING FOLDERS
        run_date = datetime.now().strftime("%d-%m-%Y_%H%M")
        base_project_dir = self.output_dir.parent / self.model_name
        self.output_dir = base_project_dir / run_date
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 4)
        resolution = self.config.get('image_size', 640)
        optim_cfg = self.config.get('optimizer', {})
        tricks = self.config.get('training_tricks', {})
        es_cfg = self.config.get('early_stopping', {})
        
        # 2. INJECT CALLBACK FOR LOSS CURVES
        def log_metrics_callback(data):
            self.history.append(data)
            
        if not hasattr(self.model, 'callbacks'):
            self.model.callbacks = {"on_fit_epoch_end": []}
        self.model.callbacks["on_fit_epoch_end"].append(log_metrics_callback)

        logger.info(f"🔥 Starting RF-DETR SEGMENTATION training (Resolution: {resolution}px)...")
        logger.info(f"📂 Outputting logs & weights to: {self.output_dir}")
        self.call_hooks('before_train')
        
        # 3. NATIVE TRAINING LOOP
        self.model.train(
            dataset_dir=str(self.dataset_path),
            output_dir=str(self.output_dir),
            epochs=epochs,
            batch_size=batch_size,
            resolution=resolution,
            
            lr=optim_cfg.get('lr', 1e-4),
            lr_encoder=optim_cfg.get('lr_encoder', 1e-5),
            weight_decay=optim_cfg.get('weight_decay', 1e-4),
            
            use_ema=tricks.get('use_ema', True),
            grad_accum_steps=tricks.get('grad_accum_steps', 4),
            
            early_stopping=es_cfg.get('enabled', True),
            early_stopping_patience=es_cfg.get('patience', 7),

            # 🛑 ADD THIS LINE: Forces the logger to update every 1 batch
            log_every_n_steps=1
        )
        
        self.call_hooks('after_train')
        self._plot_loss_curves()

    def _plot_loss_curves(self):
        if not self.history:
            return
            
        logger.info("📈 Generating Loss Curves...")
        df = pd.DataFrame(self.history)
        
        plt.figure(figsize=(10, 6))
        if 'train_loss' in df.columns:
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
        if 'test_loss' in df.columns:
            plt.plot(df['epoch'], df['test_loss'], label='Val Loss', color='orange')
            
        plt.title('RF-DETR Segmentation Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = self.output_dir / 'loss_curves.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"✅ Loss curves saved to {plot_path}")

    def evaluate(self) -> dict:
        return {} 

    def export(self, format: str = 'onnx'):
        """Exports the RF-DETR Segmentation model to ONNX."""
        if format.lower() != 'onnx':
            logger.warning(f"⚠️ RF-DETR only supports ONNX export natively. Ignoring requested format: {format}")
            
        logger.info("📦 Exporting RF-DETR Seg to ONNX...")
        
        # RF-DETR's export method takes an output directory, not a format string.
        # We pass simplify=True to optimize the graph for faster edge inference.
        self.model.export(
            output_dir=str(self.output_dir), 
            simplify=True
        ) 
        
        # By default, Roboflow names the output file 'inference_model.onnx'
        onnx_path = self.output_dir / "inference_model.onnx"
        logger.info(f"✅ ONNX Export complete: {onnx_path}")
        
        return str(onnx_path)
