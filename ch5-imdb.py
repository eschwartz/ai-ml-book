import string

import keras
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup

# Load imdb reviews training dataset
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))

# Clean up data

# Words to exclude
# see https://github.com/lmoroney/tfbook/blob/master/chapter5/imdb.ipynb
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

# remove punctuation
table = str.maketrans("", "", string.punctuation)

imdb_sentences = []
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())

    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()

    # put spaces around punctuation, so we don't create weird words
    # like "him/her" -> "himher"
    sentence = (sentence
                .replace(',', ' , ')
                .replace('.', ' . ')
                .replace('-', ' - ')
                .replace('/', ' / ')
                )

    words = sentence.split()
    filtered_sentence = ''
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "

    imdb_sentences.append(filtered_sentence)

# Tokenize reviews
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

print('done')

# Alternatively, can use the IMDb subwords dataset
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True,
)

encoder = info.features['text'].encoder
encoded_string = encoder.encode('Today is a sunny day')
print(f'encoded string: {encoded_string}')
print(f'decoded string: {encoder.decode(encoded_string)}')
