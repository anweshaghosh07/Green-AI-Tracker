# Sustainability & Green-AI Usage Tracker

## 🚀 Project Objective
This project aims to develop a tool that trains standard machine learning models while simultaneously tracking their computational and environmental footprint using CodeCarbon. The entire workflow is managed using MLOps principles (MLflow, GitHub Actions) and deployed as an interactive web application with Streamlit.

## ✨ Core Features
- Train ML models on a given dataset.
- Track and log CO2 emissions, energy consumption, and training duration for each model run.
- Use MLflow to compare experiments based on both performance metrics (e.g., accuracy) and sustainability metrics (e.g., CO2eq).
- Provide a Streamlit dashboard to visualize and compare the results.

## 💡 Motivation

As AI models grow in complexity, their computational and energy demands skyrocket. This project aims to bring awareness to the environmental impact of ML by providing a simple, integrated dashboard to monitor the CO₂ emissions and energy usage alongside traditional performance metrics like accuracy.

## 🛠️ Tech Stack
- **Data & Model Versioning:** DVC
- **Experiment Tracking:** MLflow
- **Sustainability Tracking:** CodeCarbon
- **Automation/CI/CD & MLOps:** GitHub Actions
- **Frontend:** Streamlit
- **Core Libraries:** Scikit-learn, Pandas

## 🚀 Getting Started
Follow these steps to set up the project locally, retrieve the versioned data, and run the application.

### Prerequisites
- Python 3.9+
- Git
- DVC installed (`pip install dvc`)

### Installation and Reproduction Steps
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Green-AI-Tracker
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Pull data and models using DVC:**
    This command downloads all DVC-tracked files (datasets and trained models) from S3.
    ```bash
    dvc pull
    ```

### Running the Application
1.  **Start the Streamlit Dashboard:**
    ```bash
    streamlit run app.py
    ```
2.  **View Experiment History (Optional):**
    To explore runs in the MLflow UI, run the following command in a separate terminal:
    ```bash
    mlflow ui
    ```
    Then open `http://127.0.0.1:5000` in your browser.

## 📅 Initial Plan
- **Week 1:** Foundational research and project setup.
- **Week 2:** Core Development.
