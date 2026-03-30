"""
Optimized ONNX Inference Engine for GPU (CUDA) Web Deployment
Forces GPU usage with CUDA providers
Compatible with ONNX Runtime GPU 1.24.4+
"""

import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import time
import logging
import sys
import os
from dataclasses import dataclass

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now import ONNX Runtime with error handling
try:
    import onnxruntime as ort
    
    # Print version and available providers for debugging
    logger.info(f"ONNX Runtime version: {ort.__version__}")
    logger.info(f"Available providers: {ort.get_available_providers()}")
    
except ImportError as e:
    print(f"❌ Failed to import onnxruntime: {e}")
    print("   Please install: pip install onnxruntime-gpu==1.24.4")
    sys.exit(1)
except Exception as e:
    print(f"❌ ONNX Runtime error: {e}")
    sys.exit(1)

from src.inference.base_inferencer import BaseInferencer


@dataclass
class InferenceStats:
    """Track inference performance metrics"""
    preprocess_time: float = 0.0
    inference_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    total_preprocess: float = 0.0
    total_inference: float = 0.0
    total_postprocess: float = 0.0
    
    def update(self, preprocess: float, inference: float, postprocess: float):
        """Update stats with latest frame times"""
        self.preprocess_time = preprocess
        self.inference_time = inference
        self.postprocess_time = postprocess
        self.total_time = preprocess + inference + postprocess
        self.fps = 1.0 / self.total_time if self.total_time > 0 else 0
        self.frame_count += 1
        self.total_preprocess += preprocess
        self.total_inference += inference
        self.total_postprocess += postprocess
    
    def get_avg(self) -> dict:
        """Get average statistics"""
        if self.frame_count == 0:
            return {
                'preprocess_ms': 0,
                'inference_ms': 0,
                'postprocess_ms': 0,
                'total_ms': 0,
                'fps': 0,
                'frames_processed': 0
            }
        
        return {
            'preprocess_ms': (self.total_preprocess / self.frame_count) * 1000,
            'inference_ms': (self.total_inference / self.frame_count) * 1000,
            'postprocess_ms': (self.total_postprocess / self.frame_count) * 1000,
            'total_ms': (self.total_time / self.frame_count) * 1000,
            'fps': self.fps,
            'frames_processed': self.frame_count
        }
    
    def reset(self):
        """Reset all statistics"""
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.total_time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.total_preprocess = 0.0
        self.total_inference = 0.0
        self.total_postprocess = 0.0


class OptimizedONNXInferencer(BaseInferencer):
    """
    High-Performance ONNX Inference Engine optimized for GPU (CUDA)
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        enable_profiling: bool = False,
        gpu_device_id: int = 0,
        debug: bool = True,
        device: str = None  # None = auto, "cpu" = force CPU, "cuda" = force GPU
    ):
        """
        Initialize optimized ONNX inferencer for GPU deployment
        """
        # Call parent constructor (stores self.device)
        super().__init__(model_path, conf_threshold, device)
        
        # Override parent attributes
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # IMPORTANT: class_names must be a DICTIONARY for run_live.py compatibility
        self.class_names = {}  # Dictionary mapping class_id to name
        
        self.enable_profiling = enable_profiling
        self.stats = InferenceStats()
        self.gpu_device_id = gpu_device_id
        self.debug = debug
        self.debug_counter = 0
        self.is_initialized = False
        
        # Model metadata
        self.input_shape = None
        self.input_name = None
        self.model_h = None
        self.model_w = None
        self.is_rfdetr = False
        self.nc = 2  # Default to 2 classes for your model
        self.output_names = []
        
        # Buffers
        self.input_buffer = None
        self.output_buffers = {}
        
        try:
            # 1. Configure session options
            self._configure_session_options()
            
            # 2. Initialize inference session with CUDA provider
            self.session = self._create_session(model_path, gpu_device_id)
            
            # 3. Verify GPU is actually being used
            self._verify_gpu_usage()
            
            # 4. Get model metadata
            self._get_model_metadata()
            
            # 5. Pre-allocate buffers
            self._preallocate_buffers()
            
            # 6. Load class names from config
            self._load_class_names()
            
            # 7. Ensure class_names is a dictionary
            self._ensure_class_names_dict()
            
            # 8. Warm up the model
            self._warmup()
            
            self.is_initialized = True
            
            logger.info(f"✅ Model loaded successfully: {model_path}")
            logger.info(f"   Input shape: {self.input_shape}")
            logger.info(f"   Model type: {'RF-DETR' if self.is_rfdetr else 'YOLO'}")
            logger.info(f"   Number of classes: {self.nc}")
            logger.info(f"   Classes: {list(self.class_names.values())}")
            logger.info(f"   Confidence threshold: {self.conf_threshold}")
            logger.info(f"   Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def _load_model(self):
        """Implement abstract method from BaseInferencer"""
        return self.session if hasattr(self, 'session') else None
    
    def _ensure_class_names_dict(self):
        """Ensure class_names is a dictionary with .values() and .get() methods"""
        if not isinstance(self.class_names, dict):
            if isinstance(self.class_names, list):
                # Convert list to dict
                self.class_names = {i: name for i, name in enumerate(self.class_names)}
                logger.info(f"📋 Converted class_names list to dict")
            else:
                # Default classes
                self.class_names = {0: "Carton", 1: "Polybag"}
                logger.info(f"📋 Using default class names")
    
    def _load_class_names(self):
        """Load class names from config file"""
        try:
            # Try to load from data config
            config_path = Path('data/isi_3k_dataset/data.yaml')
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                    names = data_config.get('names', [])
                    if names:
                        self.class_names = {i: name for i, name in enumerate(names)}
                        logger.info(f"📋 Loaded {len(self.class_names)} class names from config")
                        return
        except Exception as e:
            logger.debug(f"Could not load class names from config: {e}")
        
        # Default class names
        self.class_names = {0: "Carton", 1: "Polybag"}
        logger.info(f"📋 Using default class names: {list(self.class_names.values())}")
    
    def _configure_session_options(self):
        """Configure ONNX Runtime session options for optimal performance"""
        self.sess_options = ort.SessionOptions()
        
        # Graph optimization
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Memory optimizations
        self.sess_options.enable_cpu_mem_arena = False
        self.sess_options.enable_mem_pattern = True
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Thread settings
        self.sess_options.intra_op_num_threads = 2
        self.sess_options.inter_op_num_threads = 2
        
        # Reduce logging
        self.sess_options.log_severity_level = 3
    
    def _create_session(self, model_path: str, gpu_device_id: int) -> ort.InferenceSession:
        """Create ONNX Runtime session — GPU by default, CPU if forced or unavailable."""

        available_providers = ort.get_available_providers()
        logger.info(f"Available providers: {available_providers}")

        force_cpu = self.device == "cpu"

        if force_cpu:
            providers = ['CPUExecutionProvider']
            logger.info("💻 Forced CPU provider (device='cpu')")
        elif 'CUDAExecutionProvider' in available_providers:
            cuda_options = {
                'device_id': gpu_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
            logger.info("🎯 Using CUDA provider (GPU)")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("💻 CUDA unavailable — falling back to CPU")

        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=self.sess_options,
                providers=providers
            )
            return session
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    def _verify_gpu_usage(self):
        """Log which execution provider is actually active."""
        providers = self.session.get_providers()
        if 'CUDAExecutionProvider' in providers:
            logger.info("✅ GPU: CUDA execution provider active")
        else:
            logger.info("💻 CPU execution provider active")
    
    def _get_model_metadata(self):
        """Extract and store model metadata"""
        # Get input details
        self.input_node = self.session.get_inputs()[0]
        self.input_name = self.input_node.name
        self.input_shape = self.input_node.shape
        
        # Handle dynamic batch size
        if self.input_shape[0] is None:
            self.input_shape = (1, self.input_shape[1], self.input_shape[2], self.input_shape[3])
        
        # Get input dimensions
        if len(self.input_shape) == 4:
            self.model_h, self.model_w = self.input_shape[2], self.input_shape[3]
        else:
            self.model_h, self.model_w = 640, 640
        
        # Detect model type
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.is_rfdetr = any(name in ['dets', 'pred_logits', 'bboxes'] for name in self.output_names)
        
        # Get number of classes from output shape
        try:
            output_shape = self.session.get_outputs()[0].shape
            logger.info(f"Output shape: {output_shape}")
            
            if len(output_shape) == 3:
                # Shape [1, 300, 38] -> 38 = 4(box) + nc + 32(mask)
                self.nc = output_shape[2] - 4 - 32
                if self.nc <= 0:
                    self.nc = 2  # Default for your model
                logger.info(f"Inferred number of classes: {self.nc}")
        except Exception as e:
            logger.warning(f"Could not infer number of classes: {e}")
            self.nc = 2
    
    def _preallocate_buffers(self):
        """Pre-allocate memory buffers"""
        try:
            if self.input_shape:
                self.input_buffer = np.zeros(self.input_shape, dtype=np.float32)
                logger.info(f"✅ Pre-allocated input buffer: {self.input_shape}")
        except Exception as e:
            logger.warning(f"Buffer pre-allocation failed: {e}")
            self.input_buffer = None
    
    def _warmup(self):
        """Warm up the model"""
        logger.info("🔥 Warming up model...")
        try:
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            for i in range(3):
                _ = self.session.run(None, {self.input_name: dummy_input})
            logger.info("✅ Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Optimized preprocessing"""
        # Resize
        img = cv2.resize(frame, (self.model_w, self.model_h), 
                        interpolation=cv2.INTER_LINEAR)
        
        # Convert HWC to CHW and normalize
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.ascontiguousarray(img[np.newaxis, ...])
        
        # Reuse buffer if possible
        if self.input_buffer is not None and img.shape == self.input_buffer.shape:
            np.copyto(self.input_buffer, img)
            return self.input_buffer
        
        return img
    
    def predict_frame(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame
        """
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Inference
        try:
            outputs = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return sv.Detections.empty()
        
        # Postprocess
        orig_h, orig_w = frame.shape[:2]
        
        try:
            if self.is_rfdetr:
                detections = self._postprocess_rfdetr(outputs, orig_w, orig_h)
            else:
                detections = self._postprocess_yolo(outputs, orig_w, orig_h)
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            detections = sv.Detections.empty()
        
        return detections
    
    def _postprocess_yolo(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        """YOLO postprocessing with debug output"""
        if len(outputs) == 0:
            return sv.Detections.empty()
        
        # Get predictions - shape [1, 300, 38]
        preds = outputs[0][0]  # Remove batch dimension -> [300, 38]
        
        # Debug first few frames
        self.debug_counter += 1
        if self.debug and self.debug_counter <= 10:
            logger.info(f"\n{'='*50}")
            logger.info(f"🔍 DEBUG FRAME {self.debug_counter}")
            logger.info(f"{'='*50}")
            logger.info(f"Predictions shape: {preds.shape}")
            logger.info(f"Number of classes (nc): {self.nc}")
            logger.info(f"Confidence threshold: {self.conf_threshold}")
        
        # Check if we have detections
        if preds.shape[0] == 0:
            if self.debug and self.debug_counter <= 10:
                logger.warning("No predictions in output!")
            return sv.Detections.empty()
        
        # Extract components
        boxes = preds[:, :4]  # [300, 4]
        scores = preds[:, 4:4+self.nc]  # [300, nc]
        mask_coeffs = preds[:, 4+self.nc:]  # [300, 32]
        
        # Get best class and confidence
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Debug: show statistics
        if self.debug and self.debug_counter <= 10:
            logger.info(f"Boxes shape: {boxes.shape}")
            logger.info(f"Scores shape: {scores.shape}")
            logger.info(f"Confidence stats:")
            logger.info(f"  Min: {confidences.min():.4f}")
            logger.info(f"  Max: {confidences.max():.4f}")
            logger.info(f"  Mean: {confidences.mean():.4f}")
            logger.info(f"  Std: {confidences.std():.4f}")
            
            # Show top 10 confidences
            top_indices = np.argsort(confidences)[-10:][::-1]
            logger.info(f"\nTop 10 confidences:")
            for i, idx in enumerate(top_indices):
                logger.info(f"  {i+1}. Conf: {confidences[idx]:.4f}, Class: {class_ids[idx]}, Box: {boxes[idx]}")
            
            # Check if any confidence is above threshold
            above_thresh = confidences > self.conf_threshold
            logger.info(f"\nDetections above threshold: {np.sum(above_thresh)}")
            
            if np.sum(above_thresh) == 0:
                logger.info(f"💡 SUGGESTION: Lower confidence threshold to {confidences.max():.3f} to see detections")
        
        # Apply threshold
        mask = confidences > self.conf_threshold
        
        if not np.any(mask):
            if self.debug and self.debug_counter <= 10:
                logger.warning(f"No detections above threshold {self.conf_threshold}")
            return sv.Detections.empty()
        
        # Filter
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        
        # Scale boxes to original resolution
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        
        # Clip to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
        
        # Validate boxes
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not np.all(valid):
            boxes = boxes[valid]
            class_ids = class_ids[valid]
            confidences = confidences[valid]
            
            if len(boxes) == 0:
                if self.debug and self.debug_counter <= 10:
                    logger.warning("All boxes were invalid after clipping")
                return sv.Detections.empty()
        
        # Final debug output
        if self.debug and self.debug_counter <= 10:
            logger.info(f"\n✅ Final detections: {len(boxes)}")
            for i in range(min(5, len(boxes))):
                class_name = self.class_names.get(class_ids[i], f"Class_{class_ids[i]}")
                logger.info(f"  Detection {i+1}: {class_name} (ID:{class_ids[i]}) conf={confidences[i]:.4f} box=({boxes[i][0]:.0f},{boxes[i][1]:.0f},{boxes[i][2]:.0f},{boxes[i][3]:.0f})")
        
        return sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids.astype(int),
            mask=None  # Skip masks for now to debug boxes
        )
    
    def _postprocess_rfdetr(self, outputs: List[np.ndarray], orig_w: int, orig_h: int) -> sv.Detections:
        """RF-DETR postprocessing"""
        if len(outputs) < 2:
            return sv.Detections.empty()
        
        bboxes = outputs[0][0] if len(outputs) > 0 else None
        logits = outputs[1][0] if len(outputs) > 1 else None
        
        if bboxes is None or logits is None:
            return sv.Detections.empty()
        
        # Get scores and class IDs
        scores = np.max(logits, axis=1)
        class_ids = np.argmax(logits, axis=1)
        
        # Apply threshold
        mask_idx = scores > self.conf_threshold
        
        if not np.any(mask_idx):
            return sv.Detections.empty()
        
        # Filter
        bboxes = bboxes[mask_idx]
        scores = scores[mask_idx]
        class_ids = class_ids[mask_idx]
        
        # Convert center to corner
        x_c, y_c, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        bboxes_xyxy = np.column_stack([
            x_c - (w / 2),
            y_c - (h / 2),
            x_c + (w / 2),
            y_c + (h / 2)
        ])
        
        # Scale
        scale = np.array([orig_w, orig_h, orig_w, orig_h])
        bboxes_scaled = bboxes_xyxy * scale
        
        # Clip
        bboxes_scaled[:, 0] = np.clip(bboxes_scaled[:, 0], 0, orig_w)
        bboxes_scaled[:, 1] = np.clip(bboxes_scaled[:, 1], 0, orig_h)
        bboxes_scaled[:, 2] = np.clip(bboxes_scaled[:, 2], 0, orig_w)
        bboxes_scaled[:, 3] = np.clip(bboxes_scaled[:, 3], 0, orig_h)
        
        return sv.Detections(
            xyxy=bboxes_scaled,
            confidence=scores,
            class_id=class_ids.astype(int),
            mask=None
        )
    
    def predict(self, source: str, show: bool = False, save: bool = False):
        """Predict on static image"""
        frame = cv2.imread(source)
        if frame is None:
            raise FileNotFoundError(f"Source not found: {source}")
        
        detections = self.predict_frame(frame)
        
        if show:
            annotated = self._draw_detections(frame, detections)
            cv2.imshow('Detection', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if save:
            output_path = Path(source).stem + "_output.jpg"
            annotated = self._draw_detections(frame, detections)
            cv2.imwrite(output_path, annotated)
            logger.info(f"Saved to {output_path}")
        
        yield {
            "path": source,
            "detections": detections,
            "raw": detections
        }
    
    def _draw_detections(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw detections on frame"""
        annotated = frame.copy()
        
        if detections.xyxy is not None and len(detections.xyxy) > 0:
            for box, class_id, conf in zip(detections.xyxy, detections.class_id, detections.confidence):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names.get(class_id, f"Class_{class_id}")
                
                # Different colors for different classes
                color = (0, 255, 0) if class_id == 0 else (0, 165, 255)
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def get_summary(self, result: dict) -> dict:
        """Get summary of detection results"""
        detections = result['detections']
        
        class_counts = {}
        if detections.class_id is not None:
            for class_id in detections.class_id:
                class_name = self.class_names.get(class_id, str(class_id))
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "file_name": Path(result['path']).name,
            "total_detections": len(detections),
            "class_counts": class_counts,
        }
    
    def get_stats(self) -> dict:
        """Get inference statistics"""
        return self.stats.get_avg() if self.enable_profiling else {}
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats.reset()


# Alias for backward compatibility
ONNXInferencer = OptimizedONNXInferencer
