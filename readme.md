> Todo restructure

### References
My reference paper I think:
> https://github.com/ruidan/Unsupervised-Aspect-Extraction/blob/master/code/train.py
>
> https://aclanthology.org/P17-1036.pdf

> https://seunghan96.github.io/nlp/absa/ABAE/

Another one could be:
> https://www.researchgate.net/publication/220816965_An_Unsupervised_Aspect-Sentiment_Model_for_Online_Reviews


>https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis <br>
> Explains well how to do all. Nice insight on Emojis and Unicode normalization

> https://stats.stackexchange.com/questions/523066/should-stemming-and-lemmatization-both-be-used-together-or-not-what-is-best-pra <br>
> Background- stemming and lemmatization are both ways to shrink the size of the vocabulary space. By turning "running", "runner" and "runs" all into the stem or lemma "run", you can curb sparsity in your dataset. This is important as overly sparse data can lead to overfit (ie memorizing findings, not learning generalizable patterns.) Stemming is much faster as it's a fairly simple case-based algorithm. Lemmatizing is much more expensive yet doesn't offer an improvement proportionate to the increased computation time.
In either case, (or with neither method) you still need to vectorize your text inputs in some manner. TF-IDF (term frequency inverse document frequency) is a common means. It essentially says how import is a given word to a given document when contrasted with the frequency of the word occurring across all documents. Ex: If "horrible" is uncommon in a dataset of Yelp reviews but occurs three times in a specific review, then this word is assumed to play a more important role in this review than in others.
The last development in NLP text-preprocessing is using word vectors (see GloVe, Word2Vec for more info.) These methods map vocabulary to high dimensional space. And when a sequence of words is treated as the aggregation of word vectors, you get an embedding that is sometimes better than TF-IDF representation.
Short answer- go with stemming when the vocab space is small and the documents are large. Conversely, go with word embeddings when the vocab space is large but the documents are small. However, don't use lemmatization as the increased performance to increased cost ratio is quite low.



## Project (P#) of the Natural Language Processing Course
For the project I chose to elaborate the proposal number # as I have a personal
interest in the topic of the domain, being boardgames.

All proposals are present in the repository under ```resources/```

## Approach
TBD

An idea would be to make many models each for a specific task?
How do I know if a text is of importance for my model?
What to do with short or non useful text?


## IDEA?
However, specialized topic modeling algorithms like Latent Dirichlet Allocation (LDA) are more commonly used because
they are specifically designed for discovering topics in large text corpora.

> Vector space models problems The approaches seen so far did not take
into consideration semantics of words. We understood that the vector space
models suffer from high dimensionality, the terms also could be synonyms, lead-
ing to ambiguity, in fact two words that means the same thing are two different
dimensions in the model
 
Altrimenti uso i metadati? Non suona fattibile: Magari complexity è alta ma alcuni commenti lo bashano come facile!
Posso usarlo per fare validazione sul consenso generale -> Mean on all. (Feedback loop)

> One vs Rest The goal here is to learn a binary classifier between a class and all the others.
One vs One The idea here is to train a classifier between every pair of classes.

>  Extrinsic evaluation : embed the system in a real applications and measure
improvement <br>
intrinsic evaluation : define a metric and evaluate the model independent
from any application, this requires a training set and a testing set, where
the second must be different but consistent with the language learned by
the mode

> Soft binary classification?

> Clustering? Ho paura vada a farlo per gioco a quel punto
> Ma almeno non richiede l'uso di label?    

>Using the notion of linear transformation as a building block, we can use
sequence learning for several different tasks
 

> Another embedding model: GloVe

> Word embedding is extremely useful for a variety of
applications:
- Text search and retrieval
- Measuring the semantic distance among words
- Feeding a neural network language model
- Text classi cation, either supervised or unsupervised

> VIT? Fuzzy logic? Two sides of spectrum for each category.

luck or alea: all those game elements independent of player intervention, introduced by game mechanics outside the control of the players.
-> Quantize the luck if there is any. (Extract comment that recall the luck/unbalance). Contrapposto ad agency

Bookeeping -> Quanto devo consultare elementi esterni?

downtime: unproductive waiting time between one player turn and the next. By unproductive we mean not
only having nothing (or little) to do, but also nothing (or little) to think about.

interaction: the degree of influence that one player's actions have on the actions of the other participants.   

bash the leader: when, to prevent the victory of whoever is first, the players are forced to take actions against
him, often to the detriment of their own advantage or in any case without gaining anything directly. At the
table, the unfortunate situation can arise whereby one or more must "sacrifice" themselves to curb the leader
and let the others benefit from this conduct.

complicated vs complex: A game is complicated the more the rules are quantitatively many and qualitatively
equipped with exceptions. Once you understand and learn all the variables, a game (that is only) complicated
is not difficult to master. In a complicated game, solving a problem leads to immediate, certain and predictable
results.
A game is as complex as the repercussions of one's actions are difficult to predict and master. Even once you
understand and learn all the variables, a complex game is still difficult to master. In a complex game, solving
one problem leads to other problems.

#### Hard to learn vs hard to master sarebbe complicated vs complex


## Approach?

In an unsupervised paradigm for aspect extraction, you don't rely on labeled data. Instead, you can use clustering and topic modeling techniques to identify and extract aspects. Here’s how you can approach it:

    Data Collection and Preprocessing:
        Collect Data: Gather a large corpus of text related to your domain.
        Preprocess Text: Tokenize the text, remove stop words, and perform other cleaning steps.

    Text Representation:
        Word Embeddings: Use pre-trained embeddings like Word2Vec, GloVe, or contextual embeddings like BERT embeddings.
        Document Embeddings: Represent each document as a vector, for instance by averaging word embeddings or using sentence embeddings from models like Sentence-BERT.

    Aspect Extraction Techniques:

Method 1: Clustering

    Clustering: Apply clustering algorithms like K-Means, DBSCAN, or hierarchical clustering on document embeddings to identify clusters of text that correspond to different aspects.
        Determine Number of Clusters: Use methods like the elbow method or silhouette score to decide the number of clusters.
        Analyze Clusters: Interpret each cluster by examining the most frequent or representative words within each cluster to identify the aspects.

python

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Reduce dimensionality for visualization (optional)
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X.toarray())

# Cluster the documents
num_clusters = 4  # Assuming we want 4 aspects
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Analyze clusters
for i in range(num_clusters):
cluster_center = kmeans.cluster_centers_[i]
terms = vectorizer.get_feature_names_out()
print(f"Cluster {i}:")
print(" ".join([terms[i] for i in cluster_center.argsort()[-10:]]))

Method 2: Topic Modeling

    Latent Dirichlet Allocation (LDA): Use LDA to identify topics within the text, which can correspond to aspects.

python

from sklearn.decomposition import LatentDirichletAllocation

# Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Fit LDA model
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X)

# Display the topics
terms = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
print(f"Topic {idx}:")
print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

    Evaluation and Interpretation:
        Human Evaluation: Evaluate the extracted aspects by manually checking if the identified clusters/topics make sense and correspond to the desired aspects (luck, complexity, interaction, bash the leader).
        Adjustments: If needed, refine the preprocessing steps, choose different parameters, or try alternative algorithms for better results.

By following these steps, you can perform unsupervised aspect extraction to identify the desired aspects in your text corpus.


## Project Setup and Instllation
> python -m spacy download en_core_web_trf
