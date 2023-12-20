import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import syllables
import numpy as np
import os

nltk.download('punkt')
nltk.download('vader_lexicon')

# Create the directory to save the extracted data files
extracted_data_dir = "Extracted Data"
os.makedirs(extracted_data_dir, exist_ok=True)

# Read the input Excel file
input_file = "Input.xlsx"
df_input = pd.read_excel(input_file)

# Initialize lists to store the computed variables
positive_score_list = []
negative_score_list = []
polarity_score_list = []
subjectivity_score_list = []
avg_sentence_length_list = []
percentage_complex_words_list = []
fog_index_list = []
avg_words_per_sentence_list = []
complex_word_count_list = []
word_count_list = []
syllables_per_word_list = []
personal_pronouns_list = []
avg_word_length_list = []


# Define a function to calculate the average word length
def avg_word_length(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    avg_length = sum(word_lengths) / len(words)
    return avg_length


# Read positive words from Positive_words.txt
with open("positive-words.txt", "r") as positive_file:
    positive_words = positive_file.read().splitlines()

# Read negative words from Negative_words.txt
with open("negative-words.txt", "r") as negative_file:
    negative_words = negative_file.read().splitlines()

# Read stop words from each stop word text file
stop_words_files = [
    "StopWords_Auditor.txt",
    "StopWords_Currencies.txt",
    "StopWords_DatesandNumbers.txt",
    "StopWords_Generic.txt",
    "StopWords_GenericLong.txt",
    "StopWords_Geographic.txt",
    "StopWords_Names.txt"
]
stop_words = []
for file in stop_words_files:
    with open(file, "r") as stop_words_file:
        stop_words.extend(stop_words_file.read().splitlines())

# Iterate through each row in the input DataFrame
for index, row in df_input.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Extract article text from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_tag = soup.find('article')

    if article_tag is not None:
        article_text = article_tag.text.strip()

        # Save the extracted article text in a text file
        with open(os.path.join(extracted_data_dir, f'{url_id}.txt'), 'w', encoding='utf-8') as file:
            file.write(article_text)

        # Perform text analysis using NLTK
        sid = SentimentIntensityAnalyzer()
        sentences = sent_tokenize(article_text)
        blob = word_tokenize(article_text)
        positive_score = sum([1 for sentence in sentences if sid.polarity_scores(sentence)["compound"] > 0])
        negative_score = sum([1 for sentence in sentences if sid.polarity_scores(sentence)["compound"] < 0])
        polarity_score = sid.polarity_scores(article_text)["compound"]
        subjectivity_score = (positive_score + negative_score) / (len(blob) + 0.000001)
        avg_sentence_length = sum([len(word_tokenize(sentence)) for sentence in sentences]) / len(sentences)

        words = nltk.word_tokenize(article_text)
        word_count = len(words)
        complex_word_count = len([word for word in words if
                                  len(word) > 2 and re.match(r'^[a-zA-Z]+$', word) is not None and syllables.estimate(
                                      word) > 2])
        percentage_complex_words = (complex_word_count / word_count) * 100

        syllables_per_word = sum([syllables.estimate(word) for word in words]) / word_count
        personal_pronouns = len(
            [word for word in words if word.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']])
        avg_word_length_value = avg_word_length(article_text)

        # Append computed variables to the respective lists
        positive_score_list.append(positive_score)
        negative_score_list.append(negative_score)
        polarity_score_list.append(polarity_score)
        subjectivity_score_list.append(subjectivity_score)
        avg_sentence_length_list.append(avg_sentence_length)
        percentage_complex_words_list.append(percentage_complex_words)
        fog_index_list.append(0.4 * (avg_sentence_length + percentage_complex_words))
        avg_words_per_sentence_list.append(word_count / len(sentences))
        complex_word_count_list.append(complex_word_count)
        word_count_list.append(word_count)
        syllables_per_word_list.append(syllables_per_word)
        personal_pronouns_list.append(personal_pronouns)
        avg_word_length_list.append(avg_word_length_value)
    else:
        # Append placeholder values for missing article text
        positive_score_list.append(np.nan)
        negative_score_list.append(np.nan)
        polarity_score_list.append(np.nan)
        subjectivity_score_list.append(np.nan)
        avg_sentence_length_list.append(np.nan)
        percentage_complex_words_list.append(np.nan)
        fog_index_list.append(np.nan)
        avg_words_per_sentence_list.append(np.nan)
        complex_word_count_list.append(np.nan)
        word_count_list.append(np.nan)
        syllables_per_word_list.append(np.nan)
        personal_pronouns_list.append(np.nan)
        avg_word_length_list.append(np.nan)

# Create a DataFrame to store the output variables
output_df = pd.DataFrame({
    'URL_ID': df_input['URL_ID'],
    'URL': df_input['URL'],
    'POSITIVE SCORE': positive_score_list,
    'NEGATIVE SCORE': negative_score_list,
    'POLARITY SCORE': polarity_score_list,
    'SUBJECTIVITY SCORE': subjectivity_score_list,
    'AVG SENTENCE LENGTH': avg_sentence_length_list,
    'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words_list,
    'FOG INDEX': fog_index_list,
    'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence_list,
    'COMPLEX WORD COUNT': complex_word_count_list,
    'WORD COUNT': word_count_list,
    'SYLLABLE PER WORD': syllables_per_word_list,
    'PERSONAL PRONOUNS': personal_pronouns_list,
    'AVG WORD LENGTH': avg_word_length_list
})

# Save the output DataFrame to an Excel file
output_file = "Output Data Structure.xlsx"
output_df.to_excel(output_file, index=False)
