import nltk
import sys
import os
import re

from nltk import word_tokenize, punkt

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""   

NONTERMINALS = """
# Sentence
S -> NP VP | S CoordConj S
# Noun Phrase
NP -> N | Det N | AdjP N | Det N PrepP | N PrepP
# Verb Phrase
VP -> V | V NP | V NP PrepP | V PrepP | Adv VP | VP Adv
# Adjective Phrase
AdjP -> Adj | Adj AdjP
# Prepositional Phrase
PrepP -> P NP
# Coordinating Conjuction (excluding the comma)
CoordConj -> Conj S | Conj VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Convert to lowercase and remove words without at least one alphabetic character
    words = nltk.word_tokenize(sentence.lower())
    contents = [word for word in words if re.match('[a-z]', word)]

    # Return list of its words
    return contents


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []

    # Code adapted from: https://www.nltk.org/_modules/nltk/tree.html
    # Use .subtrees() for a list of subtrees within current tree; Check for additional NP subtrees by using .label()
    for s in tree.subtrees(lambda t: t.label() == 'NP' and t != tree):
        np_chunks.append(s)

    # Return a list of all noun phrase chunks in the sentence tree.
    return np_chunks


if __name__ == "__main__":
    main()
