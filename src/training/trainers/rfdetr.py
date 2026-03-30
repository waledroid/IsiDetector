import os

# Must be set BEFORE any CUDA/PyTorch operations are performed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        import torch
        import types
        from rfdetr.utilities import box_ops
        import torch.nn.functional as F

        torch.cuda.empty_cache()
        logger.info(f"🏗️ Initializing RF-DETR SEGMENTATION ({self.model_size}) with DINOv2...")
        
        if self.model_size == 'm':
            self.model = RFDETRSegMedium()
        else:
            self.model = RFDETRSegSmall()

        # 🛑 MONKEY-PATCH: Memory Safe Post-Processing
        # This prevents CUDA OOM by offloading large mask interpolations (e.g. 5MP images) to CPU
        original_postprocess = self.model.model.postprocess 
        num_select = self.config.get('num_select', 100)
        
        def patched_forward(post_self, outputs, target_sizes):
            out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
            out_masks = outputs.get("pred_masks", None)

            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), post_self.num_select, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            results = []
            if out_masks is not None:
                for i in range(out_masks.shape[0]):
                    res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
                    k_idx = topk_boxes[i]
                    masks_i = torch.gather(
                        out_masks[i],
                        0,
                        k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
                    )
                    h, w = target_sizes[i].tolist()
                    
                    # 🛡️ SAFETY CHECK: If mask interpolation > 1MP, do it on CPU to save VRAM
                    if h * w > 1024 * 1024:
                        device = masks_i.device
                        masks_i = F.interpolate(
                            masks_i.cpu().unsqueeze(1),
                            size=(int(h), int(w)),
                            mode="bilinear",
                            align_corners=False,
                        ).to(device)
                    else:
                        masks_i = F.interpolate(
                            masks_i.unsqueeze(1),
                            size=(int(h), int(w)),
                            mode="bilinear",
                            align_corners=False,
                        )
                    res_i["masks"] = masks_i > 0.0
                    results.append(res_i)
            else:
                results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
            return results

        # Apply the patch
        self.model.model.postprocess.forward = types.MethodType(patched_forward, self.model.model.postprocess)
        self.model.model.postprocess.num_select = num_select
        self.num_select = num_select # 🛑 SAVE to instance for the train method
        logger.info(f"🛡️ Memory-Safe Post-Processing active (num_select: {num_select})")

    def train(self):
        if self.model is None:
            self.build_model()
            
        # 1. TIME-STAMPED LOGGING FOLDERS
        run_date = datetime.now().strftime("%d-%m-%Y_%H%M")
        base_project_dir = self.output_dir.parent / self.model_name
        self.output_dir = base_project_dir / run_date
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = self.config.get('epochs', 100)
        # 🛡️ SAFE DEFAULTS: Use config or fallback to stable settings
        batch_size = self.config.get('batch_size', 2)
        resolution = self.config.get('image_size', 448) # 448 is safer than 640 for 12GB VRAM
        mixed_precision = self.config.get('mixed_precision', True)
        optim_cfg = self.config.get('optimizer', {})
        tricks = self.config.get('training_tricks', {})
        es_cfg = self.config.get('early_stopping', {})
        
        # 2. INJECT CALLBACK FOR LOSS CURVES + MEMORY SAFETY
        def log_metrics_callback(data):
            self.history.append(data)
            # 🧨 ULTRA-AGGRESSIVE: Clear VRAM and System RAM after every epoch
            import torch
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            
        if not hasattr(self.model, 'callbacks'):
            self.model.callbacks = {"on_fit_epoch_end": []}
        self.model.callbacks["on_fit_epoch_end"].append(log_metrics_callback)

        logger.info(f"🔥 Starting RF-DETR SEGMENTATION training (Resolution: {resolution}px)...")
        import torch
        torch.cuda.empty_cache() # 💥 Clear cache right before training
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

            # 🛠️ Pass num_select if the train method supports it (it usually does via config)
            num_select=self.num_select,

            # ⚙️ ADDITIONAL STABILITY FLAGS
            mixed_precision=mixed_precision,
            num_workers=tricks.get('num_workers', 1), # Aggressively low for WSL stability (1 default)
            pin_memory=tricks.get('pin_memory', False), # Disable if RAM is tight

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
