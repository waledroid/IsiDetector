import sys
import os
import yaml
import logging
import torch
import gc
from datetime import datetime
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
    Fully driven by the configs/train.yaml and optimizer.yaml files.
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
                'train': 'images/train',
                'val': 'images/val', 
                'test': 'images/test', 
                'nc': self.config.get('nc', 2), 
                'names': self.config.get('class_names', ['carton', 'polybag'])
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(data_dict, f, default_flow_style=False)
            logger.info(f"📄 Generated YOLO data.yaml at {yaml_path} with classes: {data_dict['names']}")
            
        return str(yaml_path)

    def build_model(self):
        """Initializes the YOLO engine, either fresh or from a checkpoint."""
        resume_path = self.config.get('resume_path')
        
        # If we are resuming, load the checkpoint file directly
        if resume_path:
            logger.info(f"🏗️ Loading Checkpoint for Resume: {resume_path}")
            self.model = YOLO(resume_path)
        else:
            model_name = f"yolov8{self.model_size}-seg.pt"
            logger.info(f"🏗️ Building Fresh Model: {model_name}")
            self.model = YOLO(model_name)

    def _inject_hooks(self):
        """Bridges Ultralytics callbacks to our BaseTrainer Hooks."""
        def on_train_epoch_end(trainer):
            self.current_epoch = trainer.epoch
            
            # Bulletproof loss extraction using raw PyTorch tensors
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
        
        # 1. Base Setup Parameters
        epochs = self.config.get('epochs', 300) 
        batch_size = self.config.get('batch_size', 16)
        img_size = self.config.get('image_size', 640)
        
        # 2. Extract Optimizer, Early Stopping, and Checkpoint configs safely
        optim_cfg = self.config.get('optimizer', {})
        sched_cfg = optim_cfg.get('scheduler', {})
        es_cfg = self.config.get('early_stopping', {})
        ckpt_cfg = self.config.get('checkpoint', {})

        # 3. Dynamically grab YOLO augmentation parameters from config
        yolo_aug_keys = ['hsv_h', 'hsv_s', 'hsv_v', 'fliplr', 'flipud', 'mosaic', 'scale', 'translate', 'degrees']
        yolo_kwargs = {k: v for k, v in self.config.items() if k in yolo_aug_keys}
        
        # Check if we are in Resume Mode
        is_resuming = bool(self.config.get('resume_path'))
        
        if is_resuming:
            logger.info(f"⏩ Fast-forwarding training to interrupted epoch (Max {epochs})...")
        else:
            logger.info(f"🔥 Starting Fresh YOLO training for {epochs} epochs at {img_size}px...")
            
        if yolo_kwargs:
            logger.info(f"🧬 Applied Augmentations: {yolo_kwargs}")
            
        self.call_hooks('before_train')
        
        # 4. Calculate YOLO's `lrf` safely (lrf = min_lr / base_lr)
        lr0 = optim_cfg.get('lr', 0.01)
        eta_min = sched_cfg.get('eta_min', 0.0001)
        lrf = (eta_min / lr0) if lr0 > 0 else 0.01
        
        # 5. DYNAMIC RUN FOLDER GENERATION
        run_date = datetime.now().strftime("%d-%m-%Y")
        base_project_dir = self.output_dir.parent / self.model_name 
        
        # Update internal output_dir for accurate logging
        self.output_dir = base_project_dir / run_date

        # Launch native Ultralytics training
        self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            resume=is_resuming,
            batch=batch_size,
            imgsz=img_size,
            
            # ROUTE TO TIMESTAMPED FOLDER
            project=str(base_project_dir),
            name=run_date,
            exist_ok=False, 
            
            device=0, 
            amp=self.config.get('mixed_precision', True),
            workers=2,      # 🛑 Locked at 2 for memory stability inside WSL
            verbose=False,
            plots=True,
            
            # OPTIMIZER INJECTION
            optimizer=optim_cfg.get('type', 'auto'),
            lr0=lr0,
            lrf=lrf,
            weight_decay=optim_cfg.get('weight_decay', 0.0005),
            warmup_epochs=sched_cfg.get('warmup_epochs', 3.0),
            cos_lr=(sched_cfg.get('type') == 'CosineAnnealing'),
            
            # EARLY STOPPING INJECTION
            patience=es_cfg.get('patience', 50) if es_cfg.get('enabled', True) else 0,
            save_period=ckpt_cfg.get('save_frequency', -1),
            
            **yolo_kwargs
        )
        
        self.call_hooks('after_train')

    def evaluate(self) -> dict:
        """Runs validation with a minimal memory footprint to prevent WSL crashes."""
        logger.info("📐 Running Lightweight Evaluation (Reduced Workers)...")
        
        # 🧹 MEMORY CLEANUP: Empty the trash before the final evaluation spike
        logger.info("🧹 Sweeping System RAM and GPU cache...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Forcefully suppress stdout to kill the glitchy progress bar
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            # 🛑 Run validation silently with restricted workers to prevent OOM
            results = self.model.val(
                data=self.data_yaml_path, 
                batch=8,    # Lower batch size specifically for final validation
                workers=2,  # Lock to 2 workers to prevent RAM spikes
                verbose=False
            )
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
        """
        High-Speed Production Export.
        Bakes NMS and coordinate scaling directly into the ONNX graph.
        """
        logger.info(f"📦 Exporting model to {format} for production...")
        
        if format == 'onnx':
            # Use specific IsiDetector deployment flags
            export_path = self.model.export(
                format='onnx',
                imgsz=self.config.get('image_size', 640),
                opset=12,
                nms=True,         # Bakes Non-Max Suppression into graph
                simplify=True,    # Graph optimization
                dynamic=False     # Static input for maximum RTX 5070 throughput
            )
        else:
            export_path = self.model.export(format=format) 

        logger.info(f"✅ Production Export complete: {export_path}")
        return export_path
