#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# 1. System Path Setup
# This ensures Python can find your 'src' folder no matter where you run the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 2. Wake up the Registry
from src.shared.registry import TRAINERS
# 🛑 CRITICAL: We must import the trainer and hook modules so their @register decorators fire!
import src.training.trainers 
import src.training.hooks

# Setup standard industrial logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Ignition")

def main():
    # 3. CLI Argument Parser
    parser = argparse.ArgumentParser(description="isiDetector Industrial Training Pipeline")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/train.yaml',
        help="Path to the master training configuration YAML."
    )
    args = parser.parse_args()

    # 4. Load the Command Center (YAML)
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Updated fallback to match your project name
    project_name = config.get('project_name', 'isiDetector')
    model_type = config.get('model_type')

    logger.info(f"🚀 INITIALIZING PROJECT: {project_name.upper()}")
    logger.info(f"🔍 Requested Model Architecture: {model_type}")

    try:
        # 5. The Magic of the Registry
        # We ask the registry for the class, and instantly instantiate it with our config
        TrainerClass = TRAINERS.get(model_type)
        trainer = TrainerClass(config)
        
        # 6. Execute the Pipeline
        trainer.train()
        
        # 7. Post-Training Validation & Export
        trainer.evaluate()
        
        if config.get('export_model', True):
            trainer.export(format='onnx')

        logger.info("✅ PIPELINE EXECUTION COMPLETELY SUCCESSFUL.")

    except KeyError as e:
        logger.error(e)
        logger.error("💡 Did you spell the model_type correctly in train.yaml?")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"💥 FATAL ERROR during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
