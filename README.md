# 📄 Resume Screening System

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20App-Visit-blue?logo=streamlit)](https://resume-screeningapp.streamlit.app/#predicted-category)

A machine learning-powered web application that **screens resumes** and predicts the **most relevant job category**. Built using Python, scikit-learn, and Streamlit, the app also attempts to extract key resume sections like **Skills** and **Experience**.

---

## 🌐 Live Demo

**[Launch the App](https://resume-screeningapp.streamlit.app/#predicted-category)**  

---

## 🔍 Features

 Upload `.pdf` or `.txt` resumes  
 Predict job domain using ML (e.g., Python Dev, HR, Data Science, etc.)  
 Extract `Skills` and `Experience` sections (if available)  
 Clean and interactive UI using **Streamlit**  
 Trained using TF-IDF + classifier model from notebook

---

## Tech Stack

- **Frontend & UI:** Streamlit
- **ML Model:** scikit-learn
- **Preprocessing:** NLTK, Regex
- **Language:** Python
- **Deployment:** Streamlit Cloud

---

---

## 🧠 Model Info

The model is trained using the notebook: `ResumeClassifier.ipynb`.  
It uses:

- **TF-IDF Vectorizer** to convert text into numerical features
- **ML classifier** (e.g., Logistic Regression or RandomForest)
- **Custom resume cleaner** with regex and NLTK
- Resume dataset with labeled categories

You can retrain the model by modifying the notebook and re-exporting `clf.pkl` and `tfidf.pkl`.

---

## 💻 Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/shibadityadeb/Resume-Screening-Sample-Dataset-.git
cd Resume-Screening-System

