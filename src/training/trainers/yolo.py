import sys
import os
import contextlib
import yaml
import logging
import torch
import gc
from pathlib import Path
from ultralytics import YOLO

from src.training.base_trainer import BaseTrainer
from src.shared.registry import TRAINERS

logger = logging.getLogger(__name__)


@TRAINERS.register('yolov26')
@TRAINERS.register('yolo')
class YOLOTrainer(BaseTrainer):
    """Trainer for YOLOv12 instance segmentation, wrapping Ultralytics.

    Registered under ``'yolo'`` and ``'yolov26'`` — both keys resolve to
    this class. Set ``model_type: "yolo"`` in ``configs/train.yaml`` to
    activate it.

    Key behaviours:

    - Auto-generates ``data/isi_3k_dataset/data.yaml`` from
      ``train.yaml`` on first run (single source of truth for class
      names and dataset paths).
    - Bridges Ultralytics' ``on_train_epoch_end`` callback to
      :meth:`~src.training.base_trainer.BaseTrainer.call_hooks` so all
      registered hooks (e.g. ``IndustrialLogger``) fire correctly.
    - Reads augmentation keys directly from the config dict and injects
      them into ``model.train()`` as keyword arguments.
    - Supports seamless resume: pass ``--resume path/to/last.pt`` to
      ``run_train.py`` and training continues from the last checkpoint.

    Attributes:
        model_size: One of ``'n'``, ``'s'``, ``'m'``, ``'l'``, ``'x'``.
            Determines which pretrained weights to load.
        dataset_path: ``Path`` to the YOLO-format dataset root.
        data_yaml_path: ``str`` path to the generated ``data.yaml``.

    Example:
        ```python
        import yaml
        from scripts.run_train import _deep_merge
        from src.training.trainers.yolo import YOLOTrainer

        with open('configs/train.yaml') as f:
            config = yaml.safe_load(f)
        with open('configs/optimizers/yolo_optim.yaml') as f:
            config = _deep_merge(config, yaml.safe_load(f))

        trainer = YOLOTrainer(config)
        trainer.train()
        metrics = trainer.evaluate()
        trainer.export(format='onnx')
        ```
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_size = config.get('model_size', 'm')
        self.dataset_path = Path(config.get('dataset_path', 'data/isi_3k_dataset'))
        self.data_yaml_path = self._prepare_data_yaml()

    def _prepare_data_yaml(self) -> str:
        """Generate the ``data.yaml`` that Ultralytics requires, if missing.

        Reads ``nc`` and ``class_names`` from ``train.yaml``, making the
        config the single source of truth. Does nothing if the file
        already exists.

        Returns:
            Absolute path to ``data.yaml`` as a string.
        """
        yaml_path = self.dataset_path / 'data.yaml'

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
        """Load YOLO weights — fresh pretrained or from a checkpoint.

        Checks ``config['resume_path']`` (set by ``--resume`` CLI flag).
        If present, loads that checkpoint; otherwise loads
        ``yolo26{model_size}-seg.pt`` pretrained weights.
        """
        resume_path = self.config.get('resume_path')

        if resume_path:
            logger.info(f"🏗️ Loading Checkpoint for Resume: {resume_path}")
            self.model = YOLO(resume_path)
        else:
            model_name = f"yolo26{self.model_size}-seg.pt"
            logger.info(f"🏗️ Building Fresh Model: {model_name}")
            self.model = YOLO(model_name)

    def _inject_framework_hooks(self):
        """Bridge Ultralytics ``on_train_epoch_end`` to BaseTrainer hooks.

        Wires an Ultralytics callback that:

        1. Copies ``trainer.epoch`` → ``self.current_epoch``.
        2. Extracts the scalar total loss from ``trainer.tloss``.
        3. Extracts per-component losses from ``trainer.loss_items``
           into ``self.loss_components`` (keys: ``box``, ``seg``,
           ``cls``, ``dfl``).
        4. Calls ``self.call_hooks('after_epoch')``.

        This is the bridge between Ultralytics' internal callback
        system and IsiDetector's hook system.
        """
        def on_train_epoch_end(trainer):
            self.current_epoch = trainer.epoch

            if hasattr(trainer, 'tloss') and trainer.tloss is not None:
                self.current_loss = float(trainer.tloss.sum())
            else:
                self.current_loss = 0.0

            # Populate per-component losses for IndustrialLogger
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                try:
                    items = trainer.loss_items.tolist()
                    self.loss_components = {k: float(v) for k, v in zip(['box', 'seg', 'cls', 'dfl'], items)}
                except Exception:
                    self.loss_components = {}

            self.call_hooks('after_epoch')

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

    def train(self):
        """Run the full YOLO training pipeline.

        Execution order:

        1. :meth:`build_model` (if model not already loaded).
        2. :meth:`_setup_run_dir` — creates ``models/yolo/DD-MM-YYYY/``.
        3. :meth:`_inject_framework_hooks` — wires Ultralytics callbacks.
        4. ``call_hooks('before_train')``.
        5. ``model.train(...)`` — full Ultralytics training loop.
        6. ``call_hooks('after_train')``.

        All hyperparameters (lr, scheduler, augmentations, early stopping)
        are read from the merged config. Augmentation keys present in the
        config (``fliplr``, ``mosaic``, ``hsv_h``, etc.) are injected
        automatically into Ultralytics via ``**kwargs``.

        Note:
            Workers are locked at 2 for WSL memory stability.
            ``device=0`` targets the first GPU.
        """
        if self.model is None:
            self.build_model()

        # 1. Timestamped output dir — set before hooks so before_train sees correct path
        self._setup_run_dir(fmt="%d-%m-%Y")

        self._inject_framework_hooks()

        # 2. Base parameters
        epochs = self.config.get('epochs', 300)
        batch_size = self.config.get('batch_size', 16)
        img_size = self.config.get('image_size', 640)

        # 3. Augmentation keys (YOLO-specific, pulled dynamically from config)
        yolo_aug_keys = ['hsv_h', 'hsv_s', 'hsv_v', 'fliplr', 'flipud', 'mosaic', 'scale', 'translate', 'degrees']
        yolo_kwargs = {k: v for k, v in self.config.items() if k in yolo_aug_keys}

        is_resuming = bool(self.config.get('resume_path'))

        if is_resuming:
            logger.info(f"⏩ Fast-forwarding training to interrupted epoch (Max {epochs})...")
        else:
            logger.info(f"🔥 Starting Fresh YOLO training for {epochs} epochs at {img_size}px...")

        if yolo_kwargs:
            logger.info(f"🧬 Applied Augmentations: {yolo_kwargs}")

        logger.info(f"📂 Outputting logs & weights to: {self.output_dir}")
        self.call_hooks('before_train')

        # 4. Calculate YOLO's lrf (lrf = min_lr / base_lr)
        sched_cfg = self.optim_cfg.get('scheduler', {})
        lr0 = self.optim_cfg.get('lr', 0.01)
        eta_min = sched_cfg.get('eta_min', 0.0001)
        lrf = (eta_min / lr0) if lr0 > 0 else 0.01

        # 5. Launch native Ultralytics training
        self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            resume=is_resuming,
            batch=batch_size,
            imgsz=img_size,

            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True,

            device=0,
            amp=self.config.get('mixed_precision', True),
            workers=2,      # Locked at 2 for memory stability inside WSL
            verbose=False,
            plots=True,

            optimizer=self.optim_cfg.get('type', 'auto'),
            lr0=lr0,
            lrf=lrf,
            weight_decay=self.optim_cfg.get('weight_decay', 0.0005),
            warmup_epochs=sched_cfg.get('warmup_epochs', 3.0),
            cos_lr=(sched_cfg.get('type') == 'CosineAnnealing'),

            patience=self.es_cfg.get('patience', 50) if self.es_cfg.get('enabled', True) else 0,
            save_period=self.ckpt_cfg.get('save_frequency', -1),

            **yolo_kwargs
        )

        self.call_hooks('after_train')

    def evaluate(self) -> dict:
        """Run post-training validation with WSL-safe memory management.

        Flushes GPU and RAM before validation to avoid OOM on the
        memory spike caused by loading the full validation set.
        Stdout is suppressed to hide Ultralytics' progress bar glitches.

        Returns:
            A metrics dictionary with keys:

            - ``'mAP50'`` — bounding-box mAP @ IoU 0.50
            - ``'mAP50_95'`` — bounding-box mAP @ IoU 0.50–0.95
            - ``'mask_mAP50'`` — polygon mask mAP @ 0.50
            - ``'mask_mAP50_95'`` — polygon mask mAP @ 0.50–0.95
            - ``'speed_ms'`` — total inference time per image in ms
        """
        logger.info("📐 Running Lightweight Evaluation (Reduced Workers)...")
        self._flush_memory()

        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            results = self.model.val(
                data=self.data_yaml_path,
                batch=8,    # Lower batch for final validation to prevent OOM
                workers=2,
                verbose=False
            )

        box_map50 = results.box.map50
        box_map = results.box.map
        mask_map50 = results.seg.map50 if hasattr(results, 'seg') else 0.0
        mask_map = results.seg.map if hasattr(results, 'seg') else 0.0
        speed_ms = sum(results.speed.values())

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

        return {
            'mAP50': float(box_map50),
            'mAP50_95': float(box_map),
            'mask_mAP50': float(mask_map50),
            'mask_mAP50_95': float(mask_map),
            'speed_ms': float(speed_ms),
        }

    def export(self, format: str = 'onnx'):
        """Export trained weights to a deployment format.

        For ONNX exports, Non-Max Suppression (NMS) and coordinate
        scaling are baked directly into the graph, producing a
        self-contained model file ready for the
        :class:`~src.inference.onnx_inferencer.OptimizedONNXInferencer`.

        Args:
            format: Export format. ``'onnx'`` (default) uses
                deployment-optimised flags (``opset=12``,
                ``simplify=True``, ``nms=True``, ``dynamic=False``).
                Any other string is passed directly to Ultralytics.

        Returns:
            Path to the exported model file as a string.
        """
        logger.info(f"📦 Exporting model to {format} for production...")

        if format == 'onnx':
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
