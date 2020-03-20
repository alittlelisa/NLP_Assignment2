import random
import nltk
from Preprocess import Preprocess
from Load_Data import data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize
random.seed(2003)
documents = []

for phraseId in range(data.shape[0]):
  documents.append((word_tokenize(data['Phrase'].iloc[phraseId]), data['Sentiment'].iloc[phraseId]))

random.shuffle(documents)

porter = PorterStemmer()
lancaster=LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_en = stopwords.words("english")
punctuations="?:!.,;'\"-()"

remove_stopwords = True
useStemming = False
useLemma = True
removePuncs = True

documents = Preprocess(documents, remove_stopwords, removePuncs, useStemming, useLemma)