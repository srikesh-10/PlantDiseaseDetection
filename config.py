"""Central project configuration for PlantDiseaseDetection.

This module contains paths and constants only. Environment-specific secrets
should be supplied through environment variables in future implementations.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"
TRAIN_DATASET_DIR = DATASET_DIR / "train"
VAL_DATASET_DIR = DATASET_DIR / "val"
TEST_DATASET_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
DATABASE_DIR = PROJECT_ROOT / "database"
KNOWLEDGEBASE_DIR = PROJECT_ROOT / "knowledgebase"

# Keras v3's native `.keras` format is used for all future saved models.
BEST_MODEL_PATH = SAVED_MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = SAVED_MODELS_DIR / "final_model.keras"
LABELS_PATH = SAVED_MODELS_DIR / "labels.json"
MODEL_PATH = FINAL_MODEL_PATH
TRAINING_LOG_PATH = LOGS_DIR / "training.log"
TRAINING_METRICS_PATH = DOCS_DIR / "training_metrics.json"
CLASSIFICATION_REPORT_PATH = DOCS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = DOCS_DIR / "confusion_matrix.png"
TRAINING_ACCURACY_PATH = DOCS_DIR / "training_accuracy.png"
TRAINING_LOSS_PATH = DOCS_DIR / "training_loss.png"
DATABASE_PATH = DATABASE_DIR / "prediction_history.sqlite3"
DISEASE_INFO_PATH = KNOWLEDGEBASE_DIR / "disease_info.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 20
LEARNING_RATE = 1e-3

CLASS_NAMES = (
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
)
