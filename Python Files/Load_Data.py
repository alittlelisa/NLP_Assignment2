import pandas as pd
import numpy as np

# load dataset
data_URL = "https://raw.githubusercontent.com/cacoderquan/Sentiment-Analysis-on-the-Rotten-Tomatoes-movie-review-dataset/master/train.tsv"
data = pd.read_csv(data_URL, sep='\t')