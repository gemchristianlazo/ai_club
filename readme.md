# 🎓 Fuzzy Logic Grading System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![Status](https://img.shields.io/badge/Status-Deployed-success)

An intelligent grading application built for the **AI Club** that utilizes **Fuzzy Logic** to evaluate student performance. Unlike traditional binary grading, this system processes linguistic variables (e.g., "Excellent", "Average", "Poor") to compute precise numerical grades based on complex logical rule sets.

## 🚀 Live Demo
*[Insert your Streamlit Cloud Link Here after deploying]*

## ⚡ Key Features

* **Linguistic Grading:** Input grades using natural language (5-point scale: *Excellent, Good, Average, Poor, Very Bad*) which are mapped to numerical vectors.
* **Fuzzy Inference Engine:** Uses Trapezoidal and Triangular membership functions to calculate partial truths.
* **3D Logic Visualization:** Interactive 3D surface plots (using Plotly) to visualize the decision-making landscape of the AI.
* **Batch Processing & Export:** Upload a CSV of an entire class, auto-grade everyone instantly, and **export/download** the results as a formatted CSV.
* **Dynamic Calibration:** Adjust the "strictness" of the grading algorithm in real-time using the sidebar sliders.

## 🧠 The Math (How it Works)

This project implements a **Mamdani Fuzzy Inference System**:

1.  **Fuzzification:** Crisp inputs (Accuracy, Writing, Timeliness) are converted into fuzzy sets using membership functions.
2.  **Rule Evaluation:** The system evaluates rules such as:
    > *IF Accuracy is High AND Writing is Good THEN Grade is Superior*
3.  **Aggregation:** The results of all triggered rules are combined.
4.  **Defuzzification:** The Center of Gravity (Centroid) method is used to convert the fuzzy result back into a crisp grade (0-100).

## 🛠️ Installation & Local Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/fuzzy-logic-grader.git](https://github.com/YOUR-USERNAME/fuzzy-logic-grader.git)
    cd fuzzy-logic-grader
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```bash
    streamlit run app.py
    ```

## 📊 CSV Batch Format

To use the batch grader, upload a CSV with the following columns. Values must use the **5-point linguistic scale**.

| Name      | Accuracy  | Writing   | Timeliness |
|-----------|-----------|-----------|------------|
| Student A | Excellent | Good      | Average    |
| Student B | Poor      | Very Bad  | Good       |
| Student C | Good      | Excellent | Excellent  |

## 📂 Project Structure

```text
fuzzy-logic-grader/
├── app.py                # Main application logic (Streamlit + Fuzzy Engine)
├── requirements.txt      # Dependencies for deployment
├── static/
│   └── profile_v1.png    # Project Logo
└── README.md             # Documentation


👨‍💻 Credits
Developer: Gem Christian O. Lazo

Professor: Jan Eilbert Lee

Organization: Artificial Intelligence Club