# Sentiment Analysis

Classify sentences to be either 'Positve' or 'Negative'.

Initial intention was to use this analysis tool to analyse Twitter trends.

Data from [SenticNet](https://sentic.net/)

The tokenization problem I was having has been fixed due to the amazing `mwe tokenizer` from `nltk`.
This tokenizer is able to split a sentence in to single word tokens and multi-word-expressions according to its settings.

I feel like I've got this working as best I can. However, I'm coming to realize using that this tool to analyse tweets isn't really feasible.
The language used on Twitter is too informal and slang-heavy for this AI to be able to glean much useful information from each tweet, which makes it largely
unreliable.

With all that said, this tool is still pretty cool. The challenges for this project were:
- ## Data Extraction
    Converting the senticnet data in to a format that I could use was a little tricky and took some trial and error, but I got it working perfectely.
    The data contained over a 100,000 lines, so manual conversion was completely out of question. I don't have much experience working with files so this 
    was some good practice.
- ## Text tokenization
    The senticnet data contains a lot phrases like 'a little' and 'a lot of fat' that contain multiple words. Initially, I was just evaluating each word separately
    while losing out on the context of the phrase. Obviously, this wasn't ideal. 

    This is where the `nltk.tokenize.mwe` module came in. This module from `nltk` is able to extract these phrases from a list or set of single word tokens and combine them
    in to multi-word-expressions which can then be evaluated.
- ## Plurals and Tenses
    The data does not contain plural and tense variations of the words, for example: "abandon" is in data but "abandoned" is not. So, if the sentence contained "abandoned", the AI
    wouldn't be able to detect the sentiment.

    Again, `nltk` comes to the rescue! The `from nltk.stem.snowball` stemmer module can be return the stem of a word (stem referring to the base version of the word without prefixes or suffixes)
    So, for any word that can't be found in the data the AI will stem the word and try again.