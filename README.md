# 🌍 Sustainability & Green-AI Usage Tracker

**Live Demo:** http://13.233.94.174/

## 🚀 Project Objective
This project aims to develop a tool that trains standard machine learning models while simultaneously tracking their computational and environmental footprint using CodeCarbon. The entire workflow is managed using MLOps principles (MLflow, GitHub Actions) and deployed as an interactive web application with Streamlit and AWS.

## 💡 Motivation
As AI models grow in complexity, their computational and energy demands skyrocket. This project aims to bring awareness to the environmental impact of ML by providing a simple, integrated dashboard to monitor the CO₂ emissions and energy usage alongside traditional performance metrics like accuracy.

## ✨ Core Features & 🛠️ Tech Stack
-   **Multi-Model Training:** Train and evaluate various models (Logistic Regression, RandomForest, SVM, CNN).
-   **Sustainability Tracking:** Uses **CodeCarbon** to monitor CO₂ emissions and energy consumption during training.
-   **Experiment Management:** Uses **MLflow** to log and compare all performance and sustainability metrics.
-   **Data & Model Versioning:** Uses **DVC** to version control large datasets and model artifacts.
-   **Interactive Dashboard:** A **Streamlit** application to filter, visualize, and gain insights from experiments.
-   **Automated CI/CD & MLOps:** **GitHub Actions** for continuous integration (testing, linting) and continuous delivery (publishing Docker images).
-   **Cloud Deployment:** Containerized with **Docker** and deployed on **AWS EC2**.

## 🚀 Getting Started
Follow these steps to set up the project locally, retrieve the versioned data, and run the application.

### Prerequisites
- Python 3.11
- Git
- DVC 
- Docker

### Installation and Reproduction Steps
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Green-AI-Tracker
    ```
2.  **Create and activate a virtual environment:**
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
    ```bash
    mlflow ui
    ```

## 🐳 Running with Docker & Docker Compose
The easiest way to run the entire application stack locally is with Docker Compose.

### Steps
1.  **Build and start the services:**
    ```bash
    docker-compose up --build
    ```
2.  **Access the applications:**
    -   **Streamlit Dashboard:** `http://localhost:8501`
    -   **MLflow UI:** `http://localhost:5000`

## ☁️ Cloud Deployment (AWS EC2)
This application is deployed on an AWS EC2 instance using a Docker container pulled from the GitHub Container Registry (GHCR).

### Summary of Deployment Steps
1.  **Launch an EC2 Instance:** An Ubuntu Server `t2.micro` (Free Tier eligible) instance is launched on AWS. A key pair is created for SSH access.
2.  **Configure Security Group:** The instance's security group is configured to allow inbound traffic on:
    -   **Port 22 (SSH)** for secure remote access.
    -   **Port 80 (HTTP)** for public web access to the Streamlit app.
3.  **Install Dependencies on EC2:** `Docker` and `Git` are installed on the virtual server.
4.  **Pull & Run Container:** The latest Docker image is pulled from GHCR. The container is then run in detached mode, mapping the public port 80 of the EC2 instance to the internal port 8501 where Streamlit is running.
    ```bash
    # Example run command on EC2
    docker run -d -p 80:8501 ghcr.io/your-github-username/green-ai-tracker:latest
    ```
5.  **Verify Deployment:** Open http://`<ec2-public-ip>`/
Ensure historical metrics are displayed.    

## ♻️ Reproducing a Training Run
To execute the training pipeline and generate new metrics, use DVC.

-   **Locally:**
    ```bash
    dvc repro
    ```
-   **With Docker Compose:**
    ```bash
    docker-compose run app dvc repro
    ```
This command runs the `train` stage defined in `dvc.yaml`, which executes the training script, tracks emissions with CodeCarbon, and logs all results to MLflow.