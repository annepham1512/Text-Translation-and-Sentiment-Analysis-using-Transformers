# Text Translation and Sentiment Analysis using Transformers

## Project Overview:

The objective of this project is to analyze the sentiment of movie reviews in three different languages - English, French, and Spanish. We have been given 30 movies, 10 in each language, along with their reviews and synopses in separate CSV files named `movie_reviews_eng.csv`, `movie_reviews_fr.csv`, and `movie_reviews_sp.csv`.

- The first step of this project is to convert the French and Spanish reviews and synopses into English. This will allow us to analyze the sentiment of all reviews in the same language. We will be using pre-trained transformers from HuggingFace to achieve this task.

- Once the translations are complete, we will create a single dataframe that contains all the movies along with their reviews, synopses, and year of release in all three languages. This dataframe will be used to perform sentiment analysis on the reviews of each movie.

- Finally, we will use pretrained transformers from HuggingFace to analyze the sentiment of each review. The sentiment analysis results will be added to the dataframe. The final dataframe will have 30 rows


The output of the project will be a CSV file with a header row that includes column names such as **Title**, **Year**, **Synopsis**, **Review**, **Review Sentiment**, and **Original Language**. The **Original Language** column will indicate the language of the review and synopsis (*en/fr/sp*) before translation. The dataframe will consist of 30 rows, with each row corresponding to a movie.


```python
# imports
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline
```

    /opt/venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


### Get data from `.csv` files and then preprocess data


```python
df_eng = pd.read_csv("data/movie_reviews_eng.csv")
df_eng.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shawshank Redemption</td>
      <td>1994</td>
      <td>Andy Dufresne (Tim Robbins), a successful bank...</td>
      <td>"The Shawshank Redemption is an inspiring tale...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>Batman (Christian Bale) teams up with District...</td>
      <td>"The Dark Knight is a thrilling and intense su...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Forrest Gump</td>
      <td>1994</td>
      <td>Forrest Gump (Tom Hanks) is a simple man with ...</td>
      <td>"Forrest Gump is a heartwarming and inspiratio...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Godfather</td>
      <td>1972</td>
      <td>Don Vito Corleone (Marlon Brando) is the head ...</td>
      <td>"The Godfather is a classic movie that stands ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Inception</td>
      <td>2010</td>
      <td>Dom Cobb (Leonardo DiCaprio) is a skilled thie...</td>
      <td>"Inception is a mind-bending and visually stun...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_fr = pd.read_csv("data/movie_reviews_fr.csv")
df_fr.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Titre</th>
      <th>Année</th>
      <th>Synopsis</th>
      <th>Critiques</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>La La Land</td>
      <td>2016</td>
      <td>Cette comédie musicale raconte l'histoire d'un...</td>
      <td>"La La Land est un film absolument magnifique ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Intouchables</td>
      <td>2011</td>
      <td>Ce film raconte l'histoire de l'amitié improba...</td>
      <td>"Intouchables est un film incroyablement touch...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amélie</td>
      <td>2001</td>
      <td>Cette comédie romantique raconte l'histoire d'...</td>
      <td>"Amélie est un film absolument charmant qui vo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Les Choristes</td>
      <td>2004</td>
      <td>Ce film raconte l'histoire d'un professeur de ...</td>
      <td>"Les Choristes est un film magnifique qui vous...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Le Fabuleux Destin d'Amélie Poulain</td>
      <td>2001</td>
      <td>Cette comédie romantique raconte l'histoire d'...</td>
      <td>"Le Fabuleux Destin d'Amélie Poulain est un fi...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sp = pd.read_csv("data/movie_reviews_sp.csv")
df_sp.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Título</th>
      <th>Año</th>
      <th>Sinopsis</th>
      <th>Críticas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Roma</td>
      <td>2018</td>
      <td>Cleo (Yalitza Aparicio) es una joven empleada ...</td>
      <td>"Roma es una película hermosa y conmovedora qu...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>La Casa de Papel</td>
      <td>(2017-2021)</td>
      <td>Esta serie de televisión española sigue a un g...</td>
      <td>"La Casa de Papel es una serie emocionante y a...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y tu mamá también</td>
      <td>2001</td>
      <td>Dos amigos adolescentes (Gael García Bernal y ...</td>
      <td>"Y tu mamá también es una película que se qued...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>El Laberinto del Fauno</td>
      <td>2006</td>
      <td>Durante la posguerra española, Ofelia (Ivana B...</td>
      <td>"El Laberinto del Fauno es una película fascin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Amores perros</td>
      <td>2000</td>
      <td>Tres historias se entrelazan en esta película ...</td>
      <td>"Amores perros es una película intensa y conmo...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use the `pd.read_csv()` function to read the movie_review_*.csv files into 3 separate pandas dataframes

# Note: All the dataframes would have different column names. For testing purposes
# you should have the following column names/headers -> [Title, Year, Synopsis, Review]

def preprocess_data() -> pd.DataFrame:
    """
    Reads movie data from .csv files, map column names, add the "Original Language" column,
    and finally concatenate in one resultant dataframe called "df".
    """
    df_eng = pd.read_csv("data/movie_reviews_eng.csv")
    df_eng["Original Language"] = "EN"
    
    df_fr = pd.read_csv("data/movie_reviews_fr.csv")
    df_fr["Original Language"] = "FR"
    
    df_sp = pd.read_csv("data/movie_reviews_sp.csv")
    df_sp['Original Language'] = 'SP'
    
    # Normalize the column names of 3 dataframes
    df_fr.columns = df_eng.columns
    df_sp.columns = df_eng.columns
    
    # Concat 3 dataframes
    df = pd.concat([df_eng, df_fr, df_sp], ignore_index=True).reset_index(drop=True).sort_index()
    
    return df

df = preprocess_data()
```


```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shawshank Redemption</td>
      <td>1994</td>
      <td>Andy Dufresne (Tim Robbins), a successful bank...</td>
      <td>"The Shawshank Redemption is an inspiring tale...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>Batman (Christian Bale) teams up with District...</td>
      <td>"The Dark Knight is a thrilling and intense su...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Forrest Gump</td>
      <td>1994</td>
      <td>Forrest Gump (Tom Hanks) is a simple man with ...</td>
      <td>"Forrest Gump is a heartwarming and inspiratio...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Godfather</td>
      <td>1972</td>
      <td>Don Vito Corleone (Marlon Brando) is the head ...</td>
      <td>"The Godfather is a classic movie that stands ...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Inception</td>
      <td>2010</td>
      <td>Dom Cobb (Leonardo DiCaprio) is a skilled thie...</td>
      <td>"Inception is a mind-bending and visually stun...</td>
      <td>EN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>Esta serie de televisión española sigue las av...</td>
      <td>"Águila Roja es una serie aburrida y poco inte...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Toc Toc</td>
      <td>2017</td>
      <td>En esta comedia española, un grupo de personas...</td>
      <td>"Toc Toc es una película aburrida y poco origi...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>27</th>
      <td>El Bar</td>
      <td>2017</td>
      <td>Un grupo de personas quedan atrapadas en un ba...</td>
      <td>"El Bar es una película ridícula y sin sentido...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Torrente: El brazo tonto de la ley</td>
      <td>1998</td>
      <td>En esta comedia española, un policía corrupto ...</td>
      <td>"Torrente es una película vulgar y ofensiva qu...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>29</th>
      <td>El Incidente</td>
      <td>2014</td>
      <td>En esta película de terror mexicana, un grupo ...</td>
      <td>"El Incidente es una película aburrida y sin s...</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Roma</td>
      <td>2018</td>
      <td>Cleo (Yalitza Aparicio) es una joven empleada ...</td>
      <td>"Roma es una película hermosa y conmovedora qu...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>16</th>
      <td>La Tour Montparnasse Infernale</td>
      <td>2001</td>
      <td>Deux employés de bureau incompétents se retrou...</td>
      <td>"Je ne peux pas croire que j'ai perdu du temps...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Nice Guys</td>
      <td>2016</td>
      <td>In 1970s Los Angeles, a private eye (Ryan Gosl...</td>
      <td>"The Nice Guys tries too hard to be funny, and...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Les Visiteurs en Amérique</td>
      <td>2000</td>
      <td>Dans cette suite de la comédie française Les V...</td>
      <td>"Le film est une perte de temps totale. Les bl...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Babylon A.D.</td>
      <td>2008</td>
      <td>Dans un futur lointain, un mercenaire doit esc...</td>
      <td>"Ce film est un gâchis complet. Les personnage...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Amélie</td>
      <td>2001</td>
      <td>Cette comédie romantique raconte l'histoire d'...</td>
      <td>"Amélie est un film absolument charmant qui vo...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Le Dîner de Cons</td>
      <td>1998</td>
      <td>Le film suit l'histoire d'un groupe d'amis ric...</td>
      <td>"Je n'ai pas aimé ce film du tout. Le concept ...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>29</th>
      <td>El Incidente</td>
      <td>2014</td>
      <td>En esta película de terror mexicana, un grupo ...</td>
      <td>"El Incidente es una película aburrida y sin s...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Amores perros</td>
      <td>2000</td>
      <td>Tres historias se entrelazan en esta película ...</td>
      <td>"Amores perros es una película intensa y conmo...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Intouchables</td>
      <td>2011</td>
      <td>Ce film raconte l'histoire de l'amitié improba...</td>
      <td>"Intouchables est un film incroyablement touch...</td>
      <td>FR</td>
    </tr>
  </tbody>
</table>
</div>



### Text translation

Translate the **Review** and **Synopsis** column values to English.


```python
# load translation models and tokenizers
# TODO 2: Update the code below
fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
es_en_model_name = "Helsinki-NLP/opus-mt-es-en"

fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)
es_en_model = MarianMTModel.from_pretrained(es_en_model_name)

fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
es_en_tokenizer = MarianTokenizer.from_pretrained(es_en_model_name)

# TODO 3: Complete the function below
def translate(text: str, model, tokenizer) -> str:
    """
    function to translate a text using a model and tokenizer
    """
    # encode the text using the tokenizer
    inputs = tokenizer(text, return_tensors="pt")

    # generate the translation using the model
    outputs = model.generate(**inputs)

    # decode the generated output and return the translated text
    decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
    decoded = decoded[0]
    return decoded
```

    /opt/venv/lib/python3.10/site-packages/transformers/models/marian/tokenization_marian.py:194: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")



```python
fr_reviews = df[df["Original Language"] == "EN"]["Review"]
fr_reviews.head(5)
```




    0    "The Shawshank Redemption is an inspiring tale...
    1    "The Dark Knight is a thrilling and intense su...
    2    "Forrest Gump is a heartwarming and inspiratio...
    3    "The Godfather is a classic movie that stands ...
    4    "Inception is a mind-bending and visually stun...
    Name: Review, dtype: object




```python
# TODO 4: Update the code below

# filter reviews in French and translate to English
fr_reviews = df[df["Original Language"] == "FR"]["Review"]
fr_reviews_en = fr_reviews.apply(lambda x: translate(x, model=fr_en_model, tokenizer=fr_en_tokenizer))

# filter synopsis in French and translate to English
fr_synopsis = df[df["Original Language"] == "FR"]["Synopsis"]
fr_synopsis_en = fr_synopsis.apply(lambda x: translate(x, model=fr_en_model, tokenizer=fr_en_tokenizer))

# filter reviews in Spanish and translate to English
es_reviews = df[df["Original Language"] == "SP"]["Review"]
es_reviews_en = es_reviews.apply(lambda x: translate(x, model=es_en_model, tokenizer=es_en_tokenizer))

# filter synopsis in Spanish and translate to English
es_synopsis = df[df["Original Language"] == "SP"]["Synopsis"]
es_synopsis_en = es_synopsis.apply(lambda x: translate(x, model=es_en_model, tokenizer=es_en_tokenizer))

# update dataframe with translated text
# add the translated reviews and synopsis - you can overwrite the existing data
df.loc[df["Original Language"] == "FR", "Review"] = fr_reviews_en
df.loc[df["Original Language"] == "FR", "Synopsis"] = fr_synopsis_en
df.loc[df["Original Language"] == "SP", "Review"] = es_reviews_en
df.loc[df["Original Language"] == "SP", "Synopsis"] = es_synopsis_en
```

    /opt/venv/lib/python3.10/site-packages/transformers/generation/utils.py:1288: UserWarning: Using `max_length`'s default (512) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
      warnings.warn(



```python
df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>This Spanish television series follows the adv...</td>
      <td>"Red Eagle is a boring and uninteresting serie...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Toc Toc</td>
      <td>2017</td>
      <td>In this Spanish comedy, a group of people with...</td>
      <td>"Toc Toc is a boring and unoriginal film that ...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>27</th>
      <td>El Bar</td>
      <td>2017</td>
      <td>A group of people are caught in a bar after Ma...</td>
      <td>"The Bar is a ridiculous and meaningless film ...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Torrente: El brazo tonto de la ley</td>
      <td>1998</td>
      <td>In this Spanish comedy, a corrupt cop (played ...</td>
      <td>"Torrente is a vulgar and offensive film that ...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>29</th>
      <td>El Incidente</td>
      <td>2014</td>
      <td>In this Mexican horror film, a group of people...</td>
      <td>"The Incident is a bore and fairless film that...</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Amélie</td>
      <td>2001</td>
      <td>This romantic comedy tells the story of Amélie...</td>
      <td>"Amélie is an absolute charm film that will ma...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Amores perros</td>
      <td>2000</td>
      <td>Three stories intertwine in this Mexican film:...</td>
      <td>"Amores dogs is an intense and moving film tha...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Le Fabuleux Destin d'Amélie Poulain</td>
      <td>2001</td>
      <td>This romantic comedy tells the story of Amélie...</td>
      <td>"The Fabulous Destiny of Amélie Poulain is an ...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>This Spanish television series follows the adv...</td>
      <td>"Red Eagle is a boring and uninteresting serie...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>Batman (Christian Bale) teams up with District...</td>
      <td>"The Dark Knight is a thrilling and intense su...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>La Casa de Papel</td>
      <td>(2017-2021)</td>
      <td>This Spanish television series follows a group...</td>
      <td>"The Paper House is an exciting and addictive ...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>9</th>
      <td>The Island</td>
      <td>2005</td>
      <td>In a future where people are cloned for organ ...</td>
      <td>"The Island is a bland and forgettable sci-fi ...</td>
      <td>EN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Roma</td>
      <td>2018</td>
      <td>Cleo (Yalitza Aparicio) is a young domestic wo...</td>
      <td>"Rome is a beautiful and moving film that pays...</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Le Dîner de Cons</td>
      <td>1998</td>
      <td>The film follows the story of a group of rich ...</td>
      <td>"I didn't like this movie at all. The concept ...</td>
      <td>FR</td>
    </tr>
    <tr>
      <th>27</th>
      <td>El Bar</td>
      <td>2017</td>
      <td>A group of people are caught in a bar after Ma...</td>
      <td>"The Bar is a ridiculous and meaningless film ...</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
</div>



### Sentiment Analysis

Use HuggingFace pretrained model for sentiment analysis of the reviews. Store the sentiment result **Positive** or **Negative** in a new column titled **Sentiment** in the dataframe.


```python
# TODO 5: Update the code below
# load sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_classifier = pipeline("sentiment-analysis", model=model_name)

# TODO 6: Complete the function below
def analyze_sentiment(text, classifier):
    """
    function to perform sentiment analysis on a text using a model
    """
    result = classifier(text)[0]
    return "Positive" if result['label'] == 'POSITIVE' else "Negative"
```

    Downloading config.json: 100%|██████████| 629/629 [00:00<00:00, 3.11MB/s]
    Downloading pytorch_model.bin: 100%|██████████| 268M/268M [00:01<00:00, 151MB/s]  
    Downloading tokenizer_config.json: 100%|██████████| 48.0/48.0 [00:00<00:00, 250kB/s]
    Downloading vocab.txt: 100%|██████████| 232k/232k [00:00<00:00, 2.72MB/s]



```python
# TODO 7: Add code below for sentiment analysis
# perform sentiment analysis on reviews and store results in new column
df['Sentiment'] = df['Review'].apply(lambda x: analyze_sentiment(x, sentiment_classifier))
```


```python
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Movie / TV Series</th>
      <th>Year</th>
      <th>Synopsis</th>
      <th>Review</th>
      <th>Original Language</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>Astérix aux Jeux Olympiques</td>
      <td>2008</td>
      <td>In this film adaptation of the popular comic s...</td>
      <td>"This film is a complete surprise. The jokes a...</td>
      <td>FR</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Scott Pilgrim vs. the World</td>
      <td>2010</td>
      <td>Scott Pilgrim (Michael Cera) must defeat his n...</td>
      <td>"It was difficult to sit through the whole thi...</td>
      <td>EN</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Águila Roja</td>
      <td>(2009-2016)</td>
      <td>This Spanish television series follows the adv...</td>
      <td>"Red Eagle is a boring and uninteresting serie...</td>
      <td>SP</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Godfather</td>
      <td>1972</td>
      <td>Don Vito Corleone (Marlon Brando) is the head ...</td>
      <td>"The Godfather is a classic movie that stands ...</td>
      <td>EN</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>29</th>
      <td>El Incidente</td>
      <td>2014</td>
      <td>In this Mexican horror film, a group of people...</td>
      <td>"The Incident is a bore and fairless film that...</td>
      <td>SP</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Le Dîner de Cons</td>
      <td>1998</td>
      <td>The film follows the story of a group of rich ...</td>
      <td>"I didn't like this movie at all. The concept ...</td>
      <td>FR</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Intouchables</td>
      <td>2011</td>
      <td>This film tells the story of the unlikely frie...</td>
      <td>"Untouchables is an incredibly touching film w...</td>
      <td>FR</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Amores perros</td>
      <td>2000</td>
      <td>Three stories intertwine in this Mexican film:...</td>
      <td>"Amores dogs is an intense and moving film tha...</td>
      <td>SP</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>27</th>
      <td>El Bar</td>
      <td>2017</td>
      <td>A group of people are caught in a bar after Ma...</td>
      <td>"The Bar is a ridiculous and meaningless film ...</td>
      <td>SP</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Les Visiteurs en Amérique</td>
      <td>2000</td>
      <td>In this continuation of the French comedy The ...</td>
      <td>"The film is a total wast of time. The jokes a...</td>
      <td>FR</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# export the results to a .csv file
import os
os.makedirs("result", exist_ok=True)
df.to_csv("result/reviews_with_sentiment.csv", index=False)
```
