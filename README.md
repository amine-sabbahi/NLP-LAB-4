<center><h1>NLP Lab 4</h1></center>

## Overview
This repository contains the work done for NLP Lab 4 as part of the Master's program in Computer Engineering at Abdelmalek Essaadi University, Faculty of Science and Technology Tangier. The lab focuses on implementing various Natural Language Processing (NLP) techniques, including classification and regression using RNN models, text generation using GPT-2, and text classification using BERT.

## Lab Tasks

### Part 1: Classification and Regression with RNN Models
- **Data Collection and Preprocessing**: Scraped text data from Arabic websites and performed preprocessing (tokenization, stemming, lemmatization, stop words removal).
- **Model Training**: Trained RNN, Bidirectional RNN, GRU, and LSTM models to classify and predict the relevance of texts.
- **Evaluation**: Evaluated the models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared metrics. The Bidirectional RNN model performed the best.

### Part 2: Text Generation with GPT-2
- **Fine-Tuning GPT-2**: Fine-tuned the pre-trained GPT-2 model on a custom dataset of Midjourney image generation prompts.
- **Text Generation**: Generated text based on given prompts, demonstrating the model's ability to produce coherent and relevant text.

### Part 3: Text Classification with BERT
- **Data Preparation**: Loaded and preprocessed a JSON dataset containing Amazon Fashion reviews.
- **Model Training**: Fine-tuned the BERT model for classification to predict the overall rating of reviews.
- **Evaluation**: Achieved high accuracy (98.42%) and F1 score (98.40%).

## Results
### Key Metrics:
- **RNN Models**:
  - **Bidirectional RNN**: MSE: 5.4139, MAE: 2.0077, R-squared: 0.0255
  - **GRU**: MSE: 6.6321, MAE: 2.2827, R-squared: -0.1938
  - **LSTM**: MSE: 12.0846, MAE: 2.9831, R-squared: -1.1752

- **GPT-2 Text Generation**:
  - Example Generated Text: "Once upon a time, in a land far away, there was a..."

- **BERT Classification**:
  - **Evaluation Loss**: 0.0674
  - **Accuracy**: 98.42%
  - **F1 Score**: 98.40%

## Tools Used
- **Python**: Programming language for implementing the models.
- **PyTorch**: Deep learning framework for building and training the RNN, GRU, LSTM, and BERT models.
- **Transformers**: Hugging Face library for using pre-trained transformer models like GPT-2 and BERT.
- **NLTK**: Natural Language Toolkit for text preprocessing (tokenization, stemming, lemmatization, stop words removal).
- **Pandas**: Data manipulation and analysis library for loading and preprocessing datasets.
- **BeautifulSoup**: Library for web scraping to collect text data from Arabic websites.
- **Scrapy**: Another web scraping framework used for data collection.
- **Sklearn**: Scikit-learn library for evaluation metrics.

## Conclusion
The lab provided hands-on experience with different NLP models and techniques, enhancing our understanding of how to preprocess text data, fine-tune pre-trained models, and evaluate their performance. These skills are essential for various NLP applications, including sentiment analysis, document classification, and machine translation.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/amine-sabbahi//NLP-LAB-4.git
    cd NLP-LAB-4
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run on Jupyter lab :
    - **ipynb format**: `NLP_LAB_4.ipynb`


### Repository Structure Description
- **Datasets**: Contains the datasets used for the lab (AMAZON_FASHION_5.json, midjourney_prompts.csv).
- **Lab 4.pdf**: The PDF document of the lab instructions.
- **NLP_LAB_4.ipynb**: Jupyter notebook containing the code and analysis.
- **README.md**: This readme file.
- **Synthesis LAB 4 (NLP).pdf**: The synthesis document summarizing the lab work.
- **requirements.txt** Contains the necessary requirements used in the lab.

## Contact
For any questions or issues, please contact [me](https://github.com/amine-sabbahi/) at mohamedamine.sabbahi@etu.uae.ac.ma

---

**Abdelmalek Essaadi University** Faculty of Science and Technology Tangier (FSTT)
   - Department : Computer Engineering
   - Master : AIDS
   - Module : NLP
   - Sujet : LAB 4
   - Realized by : [SABBAHI MOHAMED AMINE](https://github.com/amine-sabbahi/) 
   - Framed by : Pr. ELAACHAK LOTFI