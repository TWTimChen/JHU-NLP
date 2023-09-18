from collections import *
from math import *
import pprint
from random import random
from nltk.tokenize import sent_tokenize, word_tokenize

# charlm.py: exmaple code for lab 2 in 605.646

# This lab is inspired from a similar lab of Chris Callison-Burch's,
# which was in turn inspired by a blot post by Yoav Goldberg:
# https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139
#
# We fixed the example code to pad each sentence/line with a start indicator instead
# of only padding the very first sentence.

# Convert counts to probabilities for successor chars in a given context
def normalize(counter):
        s = float(sum(counter.values()))
        return [(c,cnt/s) for c,cnt in counter.items()]

# Read a training file and produce a language model
def train_char_lm(fname, order=4):
    data = open(fname).read()
    sents = data.split('\n')
    lm = defaultdict(Counter)
    for s in sents:
        pad = "~" * order
        data = pad + s + '\n'
        for i in range(len(data)-order):
            history, char = data[i:i+order], data[i+order]
            lm[history][char]+=1
    outlm = {hist:normalize(chars) for hist, chars in lm.items()}
    return outlm

# Given a character LM, randomly choose a next character given this history and return it
def generate_letter(lm, history, order):
        history = history[-order:]
        dist = lm[history]
        x = random()
        for c,v in dist:
            x = x - v
            if x <= 0: return c

# Generate a random text by repeatedly calling generate_letter
def generate_text(lm, order, nletters=1000):
    history = "~" * order
    out = []
    for i in range(nletters):
        c = generate_letter(lm, history, order)
        history = history[-order:] + c
        out.append(c)
        if c == '\n':
            history = "~" * order
    return "".join(out)

# Print alternatives given this context
def print_probs(lm, history):
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)

def train_word_lm(fname, order=1):
    """Train a word-level language model using n-grams from a given file.

    Args:
    - fname (str): Name of the file containing training data.
    - order (int): The n-gram order for the language model.

    Returns:
    - dict: The trained language model.
    """

    # Open and read the file content
    with open(fname, 'r') as f:
        data = f.read()

    # Tokenize the data into sentences
    sentences = sent_tokenize(data)

    # Tokenize each sentence into words
    sents = [word_tokenize(sentence) for sentence in sentences]

    lm = defaultdict(Counter)
    
    # Use a special padding word "<PAD>" to pad the sentences.
    # This helps in managing the start of sentences.
    pad = ["<PAD>"] * order
    for sent in sents:
        # Add padding to the start and a special "<END>" token to the end of each sentence
        sent = pad + sent + ["<END>"]
        
        # Loop through each word in the sentence and create n-grams
        for i in range(len(sent) - order):
            history, word = tuple(sent[i:i+order]), sent[i+order]
            # Update the counts for this n-gram in the language model
            lm[history][word] += 1

    # Normalize the counts to get probabilities
    outlm = {hist: normalize(chars) for hist, chars in lm.items()}
    
    return outlm


def perplexity(text, lm, order=4, mode="char"):
    """Compute the perplexity of a given text using an input language model (LM).

    Args:
    - text (str): The input text for which perplexity is calculated.
    - lm (dict): The language model.
    - order (int): The n-gram order.
    - mode (str): Mode of operation - either 'char' for character-level or 'word' for word-level.

    Returns:
    - float: The perplexity of the input text.
    """

    # Choose padding based on mode (character or word)
    if mode == "char":
        pad = "~" * order
        data = pad + text
    elif mode == "word":
        pad = ["<PAD>"] * order
        data = pad + word_tokenize(text) + ["<END>"]
    else:
        raise ValueError("Invalid mode. Choose 'char' or 'word'.")

    log_prob = 0
    for i in range(len(data) - order):
        if mode == "char":
            history, char = data[i:i+order], data[i+order]
        elif mode == "word":
            history, char = tuple(data[i:i+order]), data[i+order]
        
        # Check if the character or word is in the LM for the given history
        if char in [ch for ch, _ in lm[history]]:
            prob = dict(lm[history])[char]
            log_prob += log(prob)
        else:
            # Return infinity if probability isn't found in the model
            return float("inf")
            
    return exp(-log_prob / len(data))


def smoothed_perplexity(text, lm, order=4, mode="char"):
    """Compute the smoothed perplexity of a given text using an input LM.

    Args:
    - text (str): The input text for which perplexity is calculated.
    - lm (dict): The language model.
    - order (int): The n-gram order.
    - mode (str): Mode of operation - either 'char' for character-level or 'word' for word-level.

    Returns:
    - float: The smoothed perplexity of the input text.
    """
    
    # Choose padding based on mode (character or word)
    if mode == "char":
        pad = "~" * order
        data = pad + text
    elif mode == "word":
        pad = ["<PAD>"] * order
        data = pad + word_tokenize(text) + ["<END>"]
    else:
        raise ValueError("Invalid mode. Choose 'char' or 'word'.")

    log_prob = 0
    for i in range(len(data) - order):
        if mode == "char":
            history, char = data[i:i+order], data[i+order]
        elif mode == "word":
            history, char = tuple(data[i:i+order]), data[i+order]
        
        try:
            prob = dict(lm[history])[char]
        except KeyError:
            prob = 1.0e-7
        log_prob += log(prob)
            
    return exp(-log_prob / len(data))

# end of file
