# Text Toxicity Classifier 🤬🛡️

A Natural Language Processing (NLP) machine learning system that analyzes text inputs and classifies them as **Toxic** or **Non-Toxic**. 

This project solves the problem of automated content moderation for online communities, social media platforms, and forums, allowing platforms to filter out abusive language and maintain safe online spaces.

## 📊 Dataset Overview
The dataset used for this project consists of **4,000 rows and 7 columns** of textual data. It contains online user comments labeled across degrees of toxicity.

### Key Features Analyzed:
* **Input Text:** The raw text comment or post submitted by a user.
* **Target Categories:** Binary classification identifying toxic content flags.

## 🛠️ Tech Stack & Advanced Workflow
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Natural Language Processing (NLP):** NLTK (`PorterStemmer`, `stopwords`)
* **Imbalanced Data Handling:** Imbalanced-Learn (`SMOTE`)
* **Machine Learning Framework:** Scikit-Learn

### Engineering & Preprocessing Highlights:
* **Text Cleansing Pipeline:** Utilized regular expressions (`re`) and NLTK to strip noise, remove English stopwords, and standardize tokens.
* **Feature Engineering Experimentation:** Evaluated both `WordNetLemmatizer` and `PorterStemmer`. Opted for Stemming as it yielded superior computational performance and better downstream alignment for this specific corpus.
* **Class Imbalance Resolution:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) and **Class Weighting** adjustments to prevent model bias toward the majority (non-toxic) class.

## 📈 Model Performance & Evaluation
I thoroughly experimented with **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear SVM** across 14 distinct optimization configurations. Performance was measured via Accuracy, Precision, Recall, F1-Score, and the full Confusion Matrix.

| Model Variant | Accuracy | Precision | Recall | F1-Score | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression Baseline | 0.7625 | 0.7419 | 0.1121 | 0.1949 | 587 | 8 | 182 | 23 |
| Logistic Regression Baseline Tuned | 0.7587 | 0.5517 | 0.3121 | 0.3987 | 543 | 52 | 141 | 64 |
| Naive Bayes Baseline | 0.7450 | 0.6666 | 0.0097 | 0.0192 | 594 | 1 | 203 | 2 |
| Naive Bayes Baseline Tuned | 0.7425 | 0.4941 | 0.2048 | 0.2896 | 552 | 43 | 163 | 42 |
| Logistic Regression Class Weight | 0.7425 | 0.4976 | 0.5170 | 0.5071 | 488 | 107 | 99 | 106 |
| Naive Bayes Fit Prior | 0.7425 | 0.4942 | 0.2097 | 0.2945 | 551 | 44 | 162 | 43 |
| **Logistic Regression SMOTE** | **0.7525** | **0.5174** | **0.5073** | **0.5123** | **498** | **97** | **101** | **104** |
| Logistic Regression SMOTE (Alt) | 0.7250 | 0.4712 | 0.6000 | 0.5278 | 457 | 138 | 82 | 123 |
| Logistic Regression SMOTE Tuned | 0.7112 | 0.4429 | 0.4926 | 0.4665 | 468 | 127 | 104 | 101 |
| Naive Bayes SMOTE | 0.7187 | 0.4615 | 0.5853 | 0.5161 | 455 | 140 | 85 | 120 |
| Naive Bayes SMOTE Tuned | 0.6987 | 0.4174 | 0.4439 | 0.4302 | 468 | 127 | 114 | 91 |
| Linear SVM | 0.7525 | 0.5271 | 0.3317 | 0.4071 | 534 | 61 | 137 | 68 |
| Linear SVM SMOTE | 0.7037 | 0.4279 | 0.4634 | 0.4449 | 468 | 127 | 110 | 95 |
| Linear SVM Class Weighted | 0.7237 | 0.4611 | 0.4634 | 0.4622 | 484 | 111 | 110 | 95 |

### Key Engineering Insights:
* **The Baseline Imbalance Trap:** Standard baselines (like Naive Bayes Baseline) showed artificially high accuracy (~74%) but effectively failed at the core task, catching only 2 out of 205 toxic statements (`Recall = 0.0097`).
* **The SMOTE Advantage:** Applying **SMOTE to Logistic Regression** dramatically fixed the sensitivity of the system. It successfully caught **104 toxic instances** compared to the baseline's 23, raising the F1-Score from `0.1949` up to a far more stable `0.5123`. 
* **Trade-off Analysis:** While balancing techniques slightly reduced global accuracy, they dramatically mitigated the dangerous risk of False Negatives (missing toxic language) in a live community moderation scenario.

## 💻 How to Run This Project Locally

1. **Clone or download** this repository to your computer.
2. Open your terminal or command prompt and install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn nltk
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the `.ipynb` file and execute the cells to observe text cleaning and the model evaluation matrix.

---
📩 **Let's Connect!**
* **GitHub:** [Peasy01](https://github.com)
