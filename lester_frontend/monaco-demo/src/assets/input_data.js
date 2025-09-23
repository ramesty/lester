export const messy_original_pipeline = `import os
from transformers import pipeline
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


target_countries = ['UK', 'DE', 'FR']
customer_data = {}

sentiment_predictor = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def matches_usecase(text):
    return "complaint" in text or "bank account" in text

def sanitize(text):
    return text.lower()

with open(".scratchspace/__intermediate.csv", 'w') as output_file:
    with open("data/customers.csv") as file:
        for line in file:
            parts = line.strip().split(',')
            customer_id, customer_email, bank, country, level = parts
            is_premium = (level == 'premium')
            if country in target_countries:
                customer_data[customer_email] = (bank, country, is_premium)

    with open("data/mails.csv") as file:
        for line in file:
            parts = line.strip().split(",")
            mail_id, email, raw_date, mail_subject, mail_text = parts
            mail_year = int(raw_date.split("-")[0])
            if mail_year >= 2022:
                    if email in customer_data:
                        bank, country, is_premium = customer_data[email]
                        title = sanitize(mail_subject)
                        text = sanitize(mail_text)
                        sentiment = sentiment_predictor(mail_text)[0]['label'].lower()
                        output_file.write(f"{title}\t{text}\t{bank}\t{country}\t{sentiment}\t{is_premium}\n")


import numpy as np
from sentence_transformers import SentenceTransformer

sentence_embedder = SentenceTransformer("all-mpnet-base-v2")

def count_words(text):
    return len(text.split(" "))

country_indices = {'DE': 0, 'FR': 1, 'UK': 2}

titles = []
title_lengths = []
texts = []
countries = []


with open(".scratchspace/__intermediate.csv") as file:
    for line in file:
        parts = line.strip().split("\t")
        title, text, bank, country, sentiment, is_premium = parts

        titles.append(title)
        title_lengths.append(len(title))
        texts.append(text)
        countries.append(country)

subject_embeddings = sentence_embedder.encode(titles)
text_embeddings = sentence_embedder.encode(texts)
title_lengths_column = np.array(title_lengths)
title_lengths_column = (title_lengths_column - np.mean(title_lengths_column)) / np.std(title_lengths_column)

country_onehot = np.zeros((len(countries), len(country_indices)))
for row, country in enumerate(countries):
    country_onehot[row, country_indices[country]] = 1.0


X = np.concatenate((
    subject_embeddings,
    text_embeddings,
    title_lengths_column.reshape(-1,1),
    country_onehot
), axis=1)


labels = []
with open(".scratchspace/__intermediate.csv") as file:
    for line in file:
        parts = line.strip().split("\t")
        title, text, bank, country, sentiment, is_premium = parts

        label = 0.0
        if sentiment == 'negative' and is_premium == 'True':
            label = 1.0
        labels.append(label)

y = np.array(labels)

num_features = X.shape[1]

from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss="log_loss", penalty=None)
model.fit(X, y)

np.save(f'.scratchspace/X_train.npy', X)
np.save(f'.scratchspace/y_train.npy', y)

import pickle
with open(f".scratchspace/__model.pkl", 'wb') as model_file:
    pickle.dump(model, model_file)`
export const dataprep_input_arg_names="[customers_file, mails_file]"
export const dataprep_input_schemas="[['customer_id', 'customer_email', 'bank', 'country', 'level'], ['mail_id', 'email', 'raw_date', 'mail_subject', 'mail_text']]"
export const dataprep_output_columns="['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']"
export const featurisation_input_schema="['title', 'text', 'bank', 'country', 'sentiment', 'is_premium']"

export const initialHighlightMap = {
    "1": "green",
    "2": "green",
    "3": "green",
    "4": "green",
    "5": "green",
    "6": "green",
    "7": "green",
    "8": "green",
    "9": "green",
    "10": "green",
    "11": "green",
    "12": "green",
    "13": "green",
    "14": "green",
    "15": "green",
    "16": "green",
    "17": "green",
    "18": "green",
    "19": "green",
    "20": "green",
    "21": "green",
    "22": "green",
    "23": "green",
    "24": "green",
    "25": "green",
    "26": "green",
    "27": "green",
    "28": "green",
    "29": "green",
    "30": "green",
    "31": "green",
    "32": "green",
    "33": "green",
    "34": "green",
    "35": "green",
    "36": "green",
    "37": "green",
    "38": "green",
    "39": "green",
    "40": "green",
    "41": "green",
    "44": "yellow",
    "45": "yellow",
    "46": "yellow",
    "47": "yellow",
    "48": "yellow",
    "49": "yellow",
    "50": "yellow",
    "51": "yellow",
    "52": "yellow",
    "53": "yellow",
    "54": "yellow",
    "55": "yellow",
    "56": "yellow",
    "57": "yellow",
    "58": "yellow",
    "59": "yellow",
    "60": "yellow",
    "61": "yellow",
    "62": "yellow",
    "63": "yellow",
    "64": "yellow",
    "65": "yellow",
    "66": "yellow",
    "67": "yellow",
    "68": "yellow",
    "69": "yellow",
    "70": "yellow",
    "71": "yellow",
    "72": "yellow",
    "73": "yellow",
    "74": "yellow",
    "75": "yellow",
    "76": "yellow",
    "77": "yellow",
    "78": "yellow",
    "79": "yellow",
    "80": "yellow",
    "81": "yellow",
    "82": "yellow",
    "83": "yellow",
    "84": "yellow",
    "85": "yellow",
    "86": "yellow",
    "103": "red",
    "104": "red"
}

