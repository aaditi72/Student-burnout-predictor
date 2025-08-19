# 🎓 Student Burnout Prediction & Intervention System

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A data-driven web application built with Python and Streamlit that predicts the likelihood of student burnout and provides personalized, actionable advice to help students manage their well-being.

---

### 🚀 Live Demo

**You can access the live, interactive application here:**

**[➡️ Click here to launch the Burnout Predictor App](https://student-burnout-predictor-aditichavan.streamlit.app/)**


### 📸 Application Screenshot

![App Screenshot](https://storage.googleapis.com/gemini-prod/images/409d9491-d852-4416-8051-4043f114620f.png)
*(Feel free to replace this with your own screenshot after running the app)*

---

### ✨ Project Features

*   **Interactive Prediction:** An easy-to-use tabbed interface with sliders for inputting student data across 12 key features (Academic, Health, and Behavioral).
*   **Machine Learning Model:** Utilizes a highly accurate **XGBoost Classifier** (97% accuracy) trained on simulated data to predict burnout risk across three levels: **Low**, **Medium**, and **High**.
*   **Personalized Feedback:** Provides simple, rule-based reasoning and actionable advice based on the user's specific inputs to help them identify and address potential risk factors.
*   **Data-Driven Explainability:** The underlying Jupyter Notebook includes **SHAP** (SHapley Additive exPlanations) analysis to understand the key drivers behind the model's predictions.

---

### 🛠️ Tech Stack

*   **Backend & Machine Learning:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Joblib
*   **Frontend & Application:** Streamlit
*   **Development Environment:** Jupyter Notebook, VS Code
*   **Deployment:** Streamlit Community Cloud, Git & GitHub

---

### 📂 Project Structure

```
.
├── burnout_xgb_model.pkl      # Saved XGBoost model
├── scaler.pkl                 # Saved data scaler
├── risk_mapping.pkl           # Saved label encoder mapping
├── app.py                     # The Streamlit web application script
├── Burnout_Project.ipynb      # Jupyter notebook with the full ML workflow
├── requirements.txt           # Python dependencies
└── README.md                  # You are here!
```

---

### ⚙️ Setup and Installation

To run this project locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[Your-GitHub-Username]/Student-Burnout-predictor.git
    cd Student-Burnout-predictor
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### ▶️ How to Run

There are two main components to this project: the analysis notebook and the interactive web app.

1.  **To explore the full machine learning workflow:**
    Open and run the `Burnout_Project.ipynb` file in Jupyter Notebook or Jupyter Lab.

2.  **To launch the interactive web application:**
    Make sure you are in the project's root directory in your terminal and that the virtual environment is active. Then run:
    ```bash
    streamlit run app.py
    ```
    Your web browser will automatically open with the application running.

---

### 🧠 Machine Learning Workflow

The model was developed following a structured machine learning pipeline:

1.  **Data Simulation:** A dataset of 6,000 students was programmatically generated with 12 relevant features (e.g., GPA, Sleep Hours, Missed Deadlines).
2.  **Feature Engineering:** A `BurnoutRisk` target variable (Low, Medium, High) was created based on a scoring system derived from the input features.
3.  **Exploratory Data Analysis (EDA):** The data distribution and feature correlations were analyzed to gain insights before modeling.
4.  **Preprocessing:** The categorical target variable was label-encoded, and numerical features were standardized using `StandardScaler`.
5.  **Model Selection:** Four different models (Logistic Regression, Random Forest, SVM, and XGBoost) were trained and evaluated. **XGBoost** was selected as the final model due to its superior performance.
6.  **Model Evaluation:** The final XGBoost model was fine-tuned and achieved **~97% accuracy** on the unseen test set, validated with 5-fold cross-validation.
7.  **Explainability (XAI):** SHAP values were calculated in the notebook to identify which features were the most influential drivers of the model's predictions, ensuring the model's logic is transparent.

---

### 🚀 Future Improvements

*   **Integrate Real-World Data:** Replace the simulated dataset with anonymous data from a university survey for a more accurate and impactful model.
*   **Dynamic SHAP Explanations:** Integrate SHAP force plots directly into the Streamlit app to provide users with data-driven, visual feedback on their specific inputs.
*   **User Accounts:** Implement a simple user login to allow students to track their burnout risk over time.

---

### 👤 Author

**[Your Name Here]**

*   **LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
*   **GitHub:** [github.com/Your-GitHub-Username](https://github.com/Your-GitHub-Username)
