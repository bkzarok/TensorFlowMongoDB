from transformers import AutoConfig
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy
import numpy as np
from scipy.special import softmax
import tensorflow as tf
import pandas as pd

# distilbert model we are using
distilbert = "distilbert-base-uncased-finetuned-sst-2-english"


tokenizer = AutoTokenizer.from_pretrained(distilbert)
config = AutoConfig.from_pretrained(distilbert)
model = TFAutoModelForSequenceClassification.from_pretrained(distilbert)


def sentiment_finder(horoscope):
    input = tokenizer(horoscope, padding=True, truncation=True, max_length=512, return_tensors='tf')
    output = model(input)
    scores = output.logits[0].numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return config.id2label[ranking[0]]


# test and see if works before we try on our csv file
horoscope = "Things might get a bit confusing for you today, Capricorn. Don't feel like you need to make sense of it all. In fact, this task may be impossible. Just be yourself. Let your creative nature shine through. Other people are quite malleable, and you should feel free to take the lead in just about any situation. Make sure, however, that you consider other people's needs."
sentiment = sentiment_finder(horoscope)
print(f"Horoscope is {sentiment}")

def apply_sentiment(horoscope):
    sentiment = sentiment_finder(horoscope) 
    return 1 if sentiment == "POSITIVE" else 0

df = pd.read_csv("anaiya-six-months-horoscopes.csv")

df["sentiment"] =  df["horoscope"].apply(apply_sentiment)