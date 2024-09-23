By referring to the paper we want to make our own implementation.

- First thing we will be needing is a way to communicate with bgg to extract the comments
- We will be needing for an embedding model as they measure relative distance:
    - Do we want our own trained on the corpus of BGG? Do we want to fine-tune an existing one?
      Or do we simply want to use an exiting one without further steps?
    - **Probably the way to go is to fine-tune an existing one, but it could be a fun exercise to
      compare one trained only on BGG corpus and one fine-tuned.**
- Instead of going for Markov models or random fields we stay at the state of the art
  Unsupervised Neural Models. (We have no labeled data)

> Rather than using all available information, attention mechanism aims to focus
> on the most pertinent information for a task

We use attention

#### Final idea:

We make our own implementation of the ABAE and see if I can come up with another model
and run it as well to see how bad I am at modelling and how much thought goes to find a good architecture.

## Note

> For the ABAE model, we initialize the word embedding matrix E with word vectors trained by
> word2vec with negative sampling on each dataset,setting the embedding size to 200, window size to
> 10, and negative sample size to 5

Come pensavo. Ma vedi cosa sono window e negative sample <br>
Per embeddings guarda:
> https://stackoverflow.com/questions/64145666/fine-tuning-of-bert-word-embeddings


## Word2vec-like or BERT-like?
This question requires further research

## Going further
ABAE only recognizes Aspects, we have to give a sentiment to each one found in the comments to make
a final avarage or median on all the comments to classify a boardgame.

