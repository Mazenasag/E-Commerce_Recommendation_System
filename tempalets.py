import os
from pathlib import Path

project_name = "E-Commerce_Recommendation_System"

# List of files and folders to create
list_file = [
    # GitHub workflow placeholder
    ".github/workflows/.gitkeep",

    # Source code
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluator.py",
    f"src/{project_name}/components/prediction.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    f"src/{project_name}/pipeline/predict_pipeline.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/exception.py",
    f"src/{project_name}/utils/helpers.py",

    f"src/{project_name}/config_loader.py",

    # Config files
    "config/config.yaml",
    "config/logging.yaml",
    "params.yaml",

    # Data folder structure
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",

    # Static and templates
    "static/index.html",
    "static/styles.css",
    "static/script.js",


    "templates/index.html",
    "templates/home.html",

    # Tests
    "tests/test_data_ingestion.py",
    "tests/test_data_transformation.py",
    "tests/test_model_trainer.py",

    # Project root files
    "requirements.txt",
    "setup.py",
    "README.md",
    "application.py",
    "run_pipeline.py",
    "locustfile.py",
    "dvc.yaml",
    ".gitignore",
]

# Create folders and empty files
for file in list_file:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w'):
            pass

print(f"✅ Base project structure for '{project_name}' created successfully!")
