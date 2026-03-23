import logging
from src.shared.registry import HOOKS

logger = logging.getLogger(__name__)

@HOOKS.register('IndustrialLogger')
class IndustrialLogger:
    """A clean, epoch-level logger for Industrial Computer Vision."""
    
    def before_train(self, trainer):
        # Print the Header once at the very start
        header = (
            f"\n{'Epoch':>8} {'GPU_mem':>10} {'box_loss':>10} {'seg_loss':>10} "
            f"{'cls_loss':>10} {'dfl_loss':>10} {'Instances':>10} {'Size':>8}"
        )
        print(header)
        print("-" * len(header))

    def after_epoch(self, trainer):
        # Extract metrics from the trainer
        # trainer.current_epoch is 0-indexed in some versions, adding 1 for display
        epoch_str = f"{trainer.current_epoch + 1}/{trainer.config.get('epochs', 30)}"
        
        # Get GPU memory (if available via torch)
        import torch
        gpu_mem = f"{torch.cuda.memory_reserved(0) / 1e9:.2f}G"
        
        # Extract losses from trainer (these are updated via the callback we wrote in yolo.py)
        # Note: In a full implementation, you'd pull box, seg, cls, dfl separately.
        # For now, we'll display the summarized current_loss in the box column
        # and placeholders for others to match your requested format.
        
        row = (
            f"{epoch_str:>8} {gpu_mem:>10} {trainer.current_loss:>10.4f} "
            f"{'--':>10} {'--':>10} {'--':>10} {'--':>10} {trainer.config.get('image_size', 640):>8}"
        )
        print(row)
        
    def after_train(self, trainer):
        print("-" * 80)
        logger.info(f"✅ Training Session Complete. Best weights saved to {trainer.output_dir}")
