## Project #4 of the Natural Language Processing Course

For the project I chose to elaborate the proposal number 4 as I have a personal
interest in the topic of the domain, being boardgames.

All proposals are present in the repository under ```resources/```

## How to run

To run the solution dependencies shall be installed. They are listed in the requirements. <br>
Also install:

> python -m spacy download en_core_web_md <br>
> python -m spacy download en_core_web_sm
> 
No script has been written as I believed notebooks to be better at guiding the thought process.

First run the ```main/dataset/bgg_corpus_service.ipynb``` or download the dataset directly from: TODO LINK

To run preprocessing go to ```main/dataset/pre_processing.ipynb```. This will generate
various pre-processed datasets based on the starting one.

For LDA simply refer to ```main/lda/final_model.ipynb``` to launch training on the
best found configuration of hyperparameters. The model is then created under ```\output```
in the same directory being an LdaMulticore instance of Gensim that can be reloaded.

For ABAE it is the same but in the abae folder. It creates more files being one
for the word embeddings model, one for the initialization of aspect weight matrix before training and
the keras instance model. To load and manipulate the model please refer to the
```ABAEManager``` class that holds methods based on what output is needed (if classify or loss evaluation).

Inference is left to be done by hand but using class #todo you can save it as part of the
model output definition to be reloaded and used with #todo class to infer correct labels

### References

My reference paper I think:
> Paper: https://aclanthology.org/P17-1036.pdf <br>
> Repo :https://github.com/ruidan/Unsupervised-Aspect-Extraction/blob/master/code/train.py

Another interesting useful reference for an indepth application:
> https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis <br>
> Explains well how to do all. Nice insight on Emojis and Unicode normalization

## Approach?

In an unsupervised paradigm for aspect extraction, you don't rely on labeled data. Instead, you can use clustering and
topic modeling techniques to identify and extract aspects. Here’s how you can approach it:

    Data Collection and Preprocessing:
        Collect Data: Gather a large corpus of text related to your domain.
        Preprocess Text: Tokenize the text, remove stop words, and perform other cleaning steps.

    Text Representation:
        Word Embeddings: Use pre-trained embeddings like Word2Vec, GloVe, or contextual embeddings like BERT embeddings.
        Document Embeddings: Represent each document as a vector, for instance by averaging word embeddings or using sentence embeddings from models like Sentence-BERT.

    Aspect Extraction Techniques: ABAE, LDA

## Project Setup and Instllation

> python -m spacy download en_core_web_trf
