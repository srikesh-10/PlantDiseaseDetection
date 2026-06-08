# PlantDiseaseDetection

PlantDiseaseDetection is a production-oriented Python project scaffold for a
web-based plant disease detection system. The planned application will use
MobileNetV2 transfer learning to classify tomato and potato leaf images, then
present confidence scores, disease guidance, and prediction history.

> **Status:** Architecture and placeholders only. Model training, dataset
> processing, prediction logic, database behavior, and the Streamlit UI are not
> implemented yet.

## Supported Classes

- `Tomato_Healthy`
- `Tomato_Early_Blight`
- `Tomato_Late_Blight`
- `Potato_Healthy`
- `Potato_Early_Blight`
- `Potato_Late_Blight`

## Planned Features

- Leaf image disease classification using MobileNetV2 transfer learning
- Confidence scores for model predictions
- Symptoms, treatment recommendations, and prevention tips
- Streamlit-based web interface
- SQLite-backed prediction history
- Modular training and inference workflows
- JSON-based disease knowledge base

## Tech Stack

- Python
- TensorFlow and Keras
- OpenCV
- NumPy and Pandas
- Streamlit
- SQLite
- JSON
- Pytest
- Git and GitHub

## Folder Structure

```text
PlantDiseaseDetection/
|-- app/                    # Future Streamlit UI modules
|-- database/               # Runtime SQLite databases
|-- dataset/
|   |-- raw/                # Original source images
|   |-- processed/          # Future cleaned and transformed images
|   |-- train/              # Future training split
|   |-- val/                # Future validation split
|   `-- test/               # Future test split
|-- docs/                   # Architecture and project documentation
|-- knowledgebase/          # Disease information in JSON format
|-- logs/                   # Training, evaluation, and application logs
|-- models/
|   `-- saved_models/       # Future .keras model artifacts
|-- notebooks/              # Exploration and experimentation notebooks
|-- src/                    # Core preprocessing, model, training, and inference code
|-- tests/                  # Automated tests
|-- app.py                  # Streamlit application entry point
|-- config.py               # Shared paths and constants
|-- database.py             # Prediction-history persistence boundary
|-- predict.py              # Prediction CLI entry point
|-- train.py                # Training CLI entry point
|-- requirements.txt        # Python dependencies
`-- README.md               # Project documentation
```

## Installation

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repository-url>
   cd PlantDiseaseDetection
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Add original source images under `dataset/raw/`. Future processing workflows
   will populate `processed/`, `train/`, `val/`, and `test/`. Dataset contents
   are intentionally excluded from Git.

The current entry points raise `NotImplementedError` by design until their
respective workflows are implemented.

## Architecture

The scaffold separates user-interface, data, modeling, prediction, persistence,
and domain-knowledge concerns. Shared configuration lives in `config.py`;
reusable application logic belongs in `src/`; top-level scripts remain thin
entry points. This layout supports future CLI, web, testing, and deployment
workflows without coupling them to one another.

Future TensorFlow models will use the native Keras v3 format and be stored under
`models/saved_models/`, for example as `best_model.keras` and
`final_model.keras`. Training logs, evaluation reports, and application logs
belong under `logs/`.

## Future Enhancements

- Implement validated image preprocessing and augmentation pipelines
- Build and fine-tune the MobileNetV2 classifier
- Add evaluation metrics, experiment tracking, and model versioning
- Implement prediction services and calibrated confidence scores
- Build the Streamlit upload, results, and history views
- Add SQLite migrations and prediction-history queries
- Expand automated tests and continuous integration
- Add containerization, deployment configuration, and monitoring
- Add expert-reviewed localization and treatment guidance

## Responsible Use

Model output and knowledge-base guidance should be treated as decision support,
not a substitute for advice from a qualified agricultural professional. Future
implementations should validate recommendations for the user's crop, region,
and local regulations.
