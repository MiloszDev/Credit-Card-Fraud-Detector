import logging

from typing import List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

PROJECT_FILES: List[str] = [
    "docs/README.md",

    "main.py",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "configs/config.yaml",
    "configs/mlflow.yaml",

    "src/data/load_data.py",
    "src/data/preprocess.py",

    "src/envs/fraud_env.py",
    "src/models/dqn_agent.py",

    "src/training/train_rl.py",
    "src/training/train_ae.py",

    "src/evaluation/evaluate_agent.py",

    "src/utils/logger.py",
    "src/utils/config.py",

    "tests/test_env.py",

    "scripts/run_training.sh",
    "scripts/run_evaluation.sh",

    "models/.gitkeep",
    "logs/.gitkeep"
]

if __name__ == "__main__":
    for filepath in PROJECT_FILES:
        path = Path(filepath)
        directory = path.parent

        try:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            else:
                logging.info(f"Directory already exists: {directory}")
            
            if not path.exists():
                with open(path, "w") as f:
                    logging.info(f"Created file: {path}")
            else:
                logging.info(f"File already exists: {path}")
        
        except Exception as e:
            logging.error(f"Error creating {filepath}: {e}")