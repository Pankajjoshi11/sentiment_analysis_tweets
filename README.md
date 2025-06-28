# 🗳️ Indian Election Forecast using Twitter Sentiment Analysis

This repository contains the code and methodology for the project **"A Data-Driven Forecast of Indian Elections Using Twitter Sentiment Analysis"**, which leverages data analytics and machine learning to predict public sentiment during the 2019 Indian Lok Sabha Elections using tweets.

---

## 📌 Project Overview

With the rise of social media platforms like Twitter, analyzing public sentiment has become a powerful tool for political forecasting. This project focuses on extracting, processing, and analyzing over 78,000 tweets related to major Indian political parties—**Bharatiya Janata Party (BJP)** and **Indian National Congress (INC)**. 

Using data analytics for preprocessing and exploratory analysis, combined with machine learning for sentiment classification, we aim to uncover public opinion trends.

We applied a **Convolutional Neural Network (CNN)** model for sentiment classification, with plans to explore **BERT** and **Linear Support Vector Classifier (Linear SVC)** for comparative analysis.

---

## 💡 Key Highlights

- ✅ Achieved **94.53% accuracy** using CNN, with **0.94 precision**, **0.95 recall**, and **0.95 F1 score**.
- 📊 Preprocessing included URL removal, hashtag/mention stripping, lowercasing, tokenization, and dataset balancing via resampling.
- 🔍 Evaluation Metrics: Accuracy, Precision, Recall, F1 Score.
- 🗃️ Dataset used: *Indian Election Tweets 2019 – Kaggle*.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- TensorFlow / Keras
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Matplotlib, Seaborn, WordCloud (for visualizations)

---

## 📁 Directory Structure

```bash
.

├── bjp_tweets.csv
├── congress_tweets.csv
├── research_practise2.ipynb
├── README.md
└── requirements.txt
````

---

## 📊 Model Performance

| Model      | Accuracy | Precision | Recall | F1 Score |
| ---------- | -------- | --------- | ------ | -------- |
| CNN        | 94.53%   | 0.94      | 0.95   | 0.95     |
| BERT       | 92.99%   | 0.92      | 0.96   | 0.94     |
| Linear SVC | 81.43%   | 0.81      | 0.80   | 0.81     |

---

## 🛠️ How to Run

### Clone the Repository:

```bash
git clone https://github.com/Pankajjoshi11/sentiment_analysis_tweets.git
cd indian-election-sentiment-analysis
```

### Set Up Environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Notebook:

```bash
jupyter notebook notebooks/research_practise2.ipynb
```

* Ensure `bjp_tweets.csv` and `congress_tweets.csv` are in the `data/` folder.
* Execute the notebook cells to preprocess data, train the CNN model, and generate visualizations.

---

## 📈 Results & Visualizations

The project includes the following visualizations:

* 📉 **Loss Curves**: Training and validation loss over 30 epochs (`visualizations/loss_plot.png`)
* 📊 **Sentiment Distribution**: Bar plot of positive vs. negative tweets
* ☁️ **Word Clouds**: Frequent words in positive and negative tweets
* 🔤 **Word Frequency**: Bar plots of top words in each sentiment class

> 📌 The notebook `research_practise2.ipynb` contains code for data preprocessing, model training, and visualizations.

---

## 📚 References

* Indian Election Tweets 2019 Dataset – Kaggle
* Batra et al., “Election Result Prediction Using Twitter Sentiments Analysis”
* Chollet, F., “Deep Learning with Python”

## 📬 Contact

Feel free to reach out via email for queries or contributions:
📧 [pankaj70451@gmail.com](mailto:pankaj70451@gmail.com)

```
