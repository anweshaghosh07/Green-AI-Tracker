# Contributing Guidelines

Thank you for your interest in contributing to the Sustainability & Green-AI Usage Tracker! Whether you're fixing a bug, improving documentation, or proposing a new feature, your help is appreciated.

## How to Contribute

### Reporting Bugs or Suggesting Features

* Use the GitHub Issues tracker to report bugs or suggest new features.
* For bugs, please include steps to reproduce the issue, expected behavior, and actual results.
* For feature requests, clearly describe the proposed functionality and its potential value to the project.

### Development Workflow

We follow the standard GitHub "Fork & Pull Request" workflow.

1.  **Fork the Repository:** Create your own copy of the project on GitHub.
2.  **Clone Your Fork:** Clone your forked repository to your local machine.
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
3.  **Create a Feature Branch:** Create a new branch for your changes. Use a descriptive name (e.g., `feature/add-new-chart` or `fix/data-loading-error`).
    ```bash
    git checkout -b feature/my-new-feature
    ```
4.  **Make Changes:** Write your code and tests. Use black and flake8 for formatting/linting.

5.  **Commit Your Changes:** Commit your changes with a clear and concise message.
    ```bash
    git commit -m "feat: Add new visualization for energy consumption"
    ```
6.  **Testing:** Run pytest before pushing changes.Make sure Streamlit app (app.py) runs locally.   

7.  **Push to Your Fork:** Push your feature branch to your fork on GitHub.
    ```bash
    git push origin feature/my-new-feature
    ```
8.  **Open a Pull Request (PR):** Go to the original repository on GitHub and open a Pull Request from your feature branch to the `main` branch of the original repository. Provide a detailed description of the changes in the PR.
