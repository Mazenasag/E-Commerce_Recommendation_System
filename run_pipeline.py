"""
Main script to run the training pipeline
"""
import argparse
from src.pipeline.train_pipeline import run_training_pipeline
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the recommender system training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        run_training_pipeline(config_path=args.config)
        logger.info("🎉 Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

