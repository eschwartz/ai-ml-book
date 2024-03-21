from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',  # punctuation is not tokenized
    'I really enjoyed walking in the snow today'
]

# Tokenizer encodes words as tokens
tokenizer = Tokenizer(
    num_words=100,
    # use for words that are not in the word_index
    oov_token="<OOV>",
)
tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index)
# {'<OOV>': 1, 'today': 2, 'is': 3, 'a': 4, 'sunny': 5, 'day': 6, 'rainy': 7, 'it': 8}

# convert sentences to lists of tokens (numbers)
sequences = tokenizer.texts_to_sequences(sentences + [
    "Today is a snowy day",
    "Will it be rainy tomorrow?"
])
print(sequences)
# [ [1, 2, 3, 4, 5], [1, 2, 3, 6, 5], [2, 7, 4, 1] ]

# Pad sequences so they're all the same length
padded = pad_sequences(sequences, maxlen=6)
print(padded)
