import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

reviews = pd.read('amazon_reviews.txt')
