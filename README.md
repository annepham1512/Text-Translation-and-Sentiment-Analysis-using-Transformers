# Text Translation and Sentiment Analysis Using Transformers

## Overview

This project analyzes movie reviews and synopses in three languages—English, French, and Spanish—by performing the following tasks:

1. **Translation:** Convert all French and Spanish reviews and synopses into English using HuggingFace's pre-trained translation models.
2. **Data Consolidation:** Combine data into a single dataframe with uniform column names: `Title`, `Year`, `Synopsis`, `Review`, `Original Language`, and add a `Sentiment` column based on sentiment analysis.
3. **Sentiment Analysis:** Analyze the sentiment (positive or negative) of each review using a pre-trained sentiment analysis transformer model.
4. **Output:** Save the final dataframe as a CSV file (`reviews_with_sentiment.csv`) containing 30 rows (one for each movie).

---

## Workflow

### 1. **Data Loading and Preprocessing**

- Input CSV files:
  - `movie_reviews_eng.csv`
  - `movie_reviews_fr.csv`
  - `movie_reviews_sp.csv`
- Columns normalized to:
  - `Title`, `Year`, `Synopsis`, `Review`, `Original Language`.

### 2. **Text Translation**

Translation is performed using HuggingFace's MarianMT models:

- French to English: `Helsinki-NLP/opus-mt-fr-en`
- Spanish to English: `Helsinki-NLP/opus-mt-es-en`

### 3. **Sentiment Analysis**

Sentiment analysis is carried out using HuggingFace's `distilbert-base-uncased-finetuned-sst-2-english`. Reviews are classified as:

- **Positive**
- **Negative**

### 4. **Final Output**

The consolidated dataframe includes:

- **Title**: Movie title.
- **Year**: Year of release.
- **Synopsis**: Movie synopsis (translated into English if needed).
- **Review**: Movie review (translated into English if needed).
- **Original Language**: Language of the original review (EN, FR, SP).
- **Sentiment**: Sentiment of the review (Positive/Negative).

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `pandas`: Data handling and preprocessing.
  - `transformers`: Pre-trained transformer models for translation and sentiment analysis.
  - `os`: File system management.

---

## How to Run

1. **Install Dependencies:**

   ```bash
   pip install pandas transformers
   ```

2. **Prepare Input Data:**
   Place the CSV files in the `data/` directory:

   - `movie_reviews_eng.csv`
   - `movie_reviews_fr.csv`
   - `movie_reviews_sp.csv`

3. **Run the Script:**
   Execute the Python script to process the data:

   ```bash
   python script.py
   ```

4. **Output File:**
   The output CSV (`reviews_with_sentiment.csv`) will be saved in the `result/` directory.

---

## Example Usage

### Input Data Sample

| Title                    | Year | Synopsis                           | Review                                 | Original Language |
| ------------------------ | ---- | ---------------------------------- | -------------------------------------- | ----------------- |
| The Shawshank Redemption | 1994 | A banker is sentenced to life...   | An inspiring tale of hope and freedom. | EN                |
| Roma                     | 2018 | Cleo works for a wealthy family... | Roma is a beautiful and moving film... | SP                |

### Output Data Sample

| Title                    | Year | Synopsis                           | Review                                 | Original Language | Sentiment |
| ------------------------ | ---- | ---------------------------------- | -------------------------------------- | ----------------- | --------- |
| The Shawshank Redemption | 1994 | A banker is sentenced to life...   | An inspiring tale of hope and freedom. | EN                | Positive  |
| Roma                     | 2018 | Cleo works for a wealthy family... | Roma is a beautiful and moving film... | SP                | Positive  |

---

## Results

The final dataframe contains:

- **30 rows**, one for each movie.
- Sentiment analysis results for all reviews.
- Reviews and synopses translated into English where necessary.

For more information or improvements, feel free to reach out!
