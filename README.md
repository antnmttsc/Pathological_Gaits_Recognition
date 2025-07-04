# Pathological Gait Recognition

This repository addresses the task of pathological gait recognition using deep learning techniques, leveraging both unimodal and hybrid approaches.

## Overview

The project includes data preprocessing, model training with Leave-One-Subject-Out Cross-Validation (LOSO-CV), and a basic web application for inference.

We used Google Colab with a Tesla T4 GPU (Free Tier) for experimentation. The dataset is stored as a `.zip` file on Google Drive. A utility archive (`Utils.zip`) is manually uploaded via the Colab interface after initializing the virtual environment.

## Repository Structure

- `Utils.zip`  
  Contains reusable functions and constants for data processing and model training.

- `Preprocessing.ipynb`  
  Demonstrates each step of the preprocessing pipeline with visual outputs. The corresponding functions are also implemented in `Utils/Preprocessing.py`.

- `LOSO_CV.ipynb`  
  The main notebook for training and evaluating deep learning models. Uses a Leave-One-Subject-Out Cross-Validation strategy to assess model performance. The best-performing model is then trained and saved along with its trained weights.
  Due to file size limits allowed by github the outputs have been removed, for results refer to `Report.pdf`.

- `my_app.py`  
  A prototype Streamlit web application for local inference. Users can upload a `.csv` file, and the app will predict the gait label and display class probabilities. It relies on utility functions found in the `Utils` folder and is designed to be run locally (e.g., in VS Code).

- `Report.pdf`  
  A detailed report presenting methodology, experiments, and final results.

## Requirements

- Python 3.8+
- TensorFlow 2.18.0
- Keras 3.8.0
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Google Colab (optional, for GPU acceleration)

##

For any issues or questions, feel free to contact me at [antonio.mattesco01@gmail.com](mailto:antonio.mattesco01@gmail.com).
