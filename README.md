# Understanding Uncertainty in Large Language Model Predictions of Early Death in the Emergency Department: A Conformal Prediction Approach

This repository contains the full pipeline to reproduce the results from our study on quantifying uncertainty in GPT-4o predictions of in-hospital mortality using conformal prediction. The work utilizes unstructured clinical notes from emergency department (ED) admissions and applies large language models (LLMs) for early prediction of in-hospital death, with uncertainty calibration through conformal prediction.

## 📂 Repository Structure

### 1. `Building_Cohort_Final_repo.ipynb`
- This Jupyter notebook contains the code used to build the final study cohort.
- Filters patients with emergency department admission and acute kidney failure diagnosis.
- Extracts and preprocesses 24-hour unstructured clinical notes for input to the language model.

### 2. `prompting_GPT_repo.py`
- A Python script that formats the extracted notes and sends them as zero-shot prompts to GPT-4o.
- Handles API communication, prompt construction, and output parsing for mortality prediction.


### 3. `Final_Conformal_Prediction_repo.ipynb`
- Implements conformal prediction to quantify uncertainty in GPT-4o predictions.
- Calculates nonconformity scores and generates calibrated prediction sets.
- Evaluates empirical coverage, class-specific performance, and analyzes overconfidence patterns.

## 🧪 Key Features

- **Zero-shot inference** using GPT-4o for binary classification: in-hospital death vs survival.
- **Uncertainty quantification** using conformal prediction to ensure statistically valid confidence guarantees.
- **Fine-grained analysis** of prediction behavior, stratified by note length, confidence, and outcome classes.

## 📝 Requirements

- Python 3.8+
- `openai`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`, `tqdm`, and `jupyter`
- OpenAI API key for GPT access
