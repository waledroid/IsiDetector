import sys
import os
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

from src.training.base_trainer import BaseTrainer
from src.shared.registry import TRAINERS

logger = logging.getLogger(__name__)

@TRAINERS.register('yolov26')
@TRAINERS.register('yolo')
class YOLOTrainer(BaseTrainer):
    """
    Concrete implementation of the YOLO segmentation model.
    Fully driven by the configs/train.yaml file.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_size = config.get('model_size', 'm') 
        self.dataset_path = Path(config.get('dataset_path', 'data/isi_3k_dataset'))
        
        # Ensure YOLO has its required data.yaml, built directly from our config!
        self.data_yaml_path = self._prepare_data_yaml()
        
    def _prepare_data_yaml(self) -> str:
        """Dynamically generates the data.yaml using variables from train.yaml."""
        yaml_path = self.dataset_path / 'data.yaml'
           
       # If it doesn't exist, create it using config variables
        if not yaml_path.exists():
           data_dict = {
               'path': str(self.dataset_path.absolute()),
               # 🛑 FIX: Point explicitly to the images subdirectories
               'train': 'images/train',
               'val': 'images/val', 
               'test': 'images/test', # Optional, but good practice
               'nc': self.config.get('nc', 2), 
               'names': self.config.get('class_names', ['carton', 'polybag'])
           }
           with open(yaml_path, 'w') as f:
               yaml.dump(data_dict, f, default_flow_style=False)
           logger.info(f"📄 Generated YOLO data.yaml at {yaml_path} with classes: {data_dict['names']}")
           
        return str(yaml_path)

    def build_model(self):
        """Initializes the YOLO engine."""
        model_name = f"yolov8{self.model_size}-seg.pt" # Using v8 naming convention until v26 wheel is live
        logger.info(f"🏗️ Building Model: {model_name}")
        self.model = YOLO(model_name)

    def _inject_hooks(self):
        """Bridges Ultralytics callbacks to our BaseTrainer Hooks."""
        def on_train_epoch_end(trainer):
            self.current_epoch = trainer.epoch
            
            # 🛑 FIX: Bulletproof loss extraction using raw PyTorch tensors
            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                self.current_loss = float(trainer.tloss.sum())
            else:
                self.current_loss = 0.0
                
            self.call_hooks('after_epoch')

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
    

    def train(self):
        """Executes the training loop with our injected hooks."""
        if self.model is None:
            self.build_model()
            
        self._inject_hooks()
        
        epochs = self.config.get('epochs', 30)
        batch_size = self.config.get('batch_size', 16)
        img_size = self.config.get('image_size', 640)
        
        # 🛑 FIX 2: Dynamically grab YOLO augmentation parameters from config
        yolo_aug_keys = ['hsv_h', 'hsv_s', 'hsv_v', 'fliplr', 'flipud', 'mosaic', 'scale', 'translate', 'degrees']
        yolo_kwargs = {k: v for k, v in self.config.items() if k in yolo_aug_keys}
        
        logger.info(f"🔥 Starting YOLO training for {epochs} epochs at {img_size}px...")
        if yolo_kwargs:
            logger.info(f"🧬 Applied Augmentations: {yolo_kwargs}")
            
        self.call_hooks('before_train')
        
        # Launch native Ultralytics training
        self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=str(self.output_dir.parent),
            name=self.model_name,
            device=0, 
            exist_ok=True,
            amp=self.config.get('mixed_precision', True),
            workers=4,
            verbose=False,  # 🛑 SET THIS TO FALSE to stop batch spam
            plots=True,     # Keep this True to see your validation charts
            **yolo_kwargs
        )
        
        self.call_hooks('after_train')

    def evaluate(self) -> dict:
        """Runs validation and extracts mAP metrics cleanly."""
        logger.info("📐 Running validation scan (this takes a few seconds)...")
        
        # 🛑 ULTIMATE FIX: Forcefully suppress stdout to kill the glitchy progress bar
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # Run validation silently
            results = self.model.val(data=self.data_yaml_path, verbose=False)
        finally:
            # ALWAYS restore stdout, even if validation crashes
            sys.stdout.close()
            sys.stdout = original_stdout
        
        # Extract the metrics
        box_map50 = results.box.map50
        box_map = results.box.map
        mask_map50 = results.seg.map50 if hasattr(results, 'seg') else 0.0
        mask_map = results.seg.map if hasattr(results, 'seg') else 0.0
        speed_ms = sum(results.speed.values())
        
        # 📊 Print a clean, industrial Executive Summary
        print("\n" + "═" * 55)
        print(f" {'VALIDATION EXECUTIVE SUMMARY':^53} ")
        print("═" * 55)
        print(f" {'Metric':<20} | {'Bounding Box':<14} | {'Polygon Mask':<14}")
        print("-" * 55)
        print(f" {'mAP @ 50':<20} | {box_map50:<14.4f} | {mask_map50:<14.4f}")
        print(f" {'mAP @ 50-95':<20} | {box_map:<14.4f} | {mask_map:<14.4f}")
        print("-" * 55)
        print(f" {'Inference Speed':<20} | {speed_ms:.2f} ms per image")
        print("═" * 55 + "\n")
        
        metrics = {
            'mAP50': float(box_map50),
            'mAP50_95': float(box_map),
            'speed_ms': float(speed_ms)
        }
        return metrics

    def export(self, format: str = 'onnx'):
        """Exports the model for Edge deployment."""
        logger.info(f"📦 Exporting model to {format}...")
        
        # 🛑 FIX: Removed int8=True as it is not natively supported for ONNX in v8.4
        export_path = self.model.export(format=format) 
        
        logger.info(f"✅ Export complete: {export_path}")
        return export_path
