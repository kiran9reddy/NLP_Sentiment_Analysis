# NLP Sentiment Analysis Project

## Overview
This project demonstrates **fine-tuning DistilBERT** for **binary sentiment classification** on the IMDB dataset (positive/negative reviews). It includes both **training scripts** and a **demo notebook** for inference.

## Folder Structure



NLP_Sentiment_Analysis/
├── notebooks/
│   └── sentiment_demo.ipynb       # Demo notebook for inference and evaluation
├── src/
│   ├── train_trainer.py           # Training using HuggingFace Trainer
│   ├── train_manual.py            # Training using custom training loop
│   ├── predict.py                 # Prediction / inference script
│   └── model/                     # Folder to save trained model
├── README.md                      # Project overview
└── requirements.txt               # Dependencies



## Features
- Fine-tune DistilBERT for sentiment analysis
- Custom training loop or HuggingFace Trainer
- Model evaluation: Accuracy, F1-score, Confusion Matrix
- Demo notebook with ready-to-run examples
- Clean, modular folder structure suitable for GitHub portfolio

## Installation
```bash
git clone https://github.com/kiran9reddy/NLP_Sentiment_Analysis.git
cd NLP_Sentiment_Analysis
pip install -r requirements.txt

##Usage

Train the model

python src/train_trainer.py
# or
python src/train_manual.py

##RUN predictions

python src/predict.py


Run the demo notebook
Open notebooks/sentiment_demo.ipynb in Jupyter or Colab to see inference and evaluation.