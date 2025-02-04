I am doing a complete overhaul. Something is messed up somewhere.<br>
Is the metric not well-defined? What is it?

### Guarda da slides: Application example: use graph community detection to find aspects

> https://github.com/afflint/nlp/blob/main/nlp/word2vec.ipynb


> I will try out other approaches. <br> What we have done using ABAE will
> be documented in the report BUT we will se if it really was the path to pursuit.

## Main Tasks:

> Good news everyone! By looking the data on the restaurant coherence and our model

- Preprocess the dataset
    - Refer to the restaurant dataset provided by og. paper of ABAE as it is commonly used
      for aspect extraction tasks.
    - Should I remove reviews that do not cite my aspects? If so HOW?
- Define the CAt model and use its default configuration.
    - If it still is very disappointing we could conclude that our dataset has some serious issues.
- Run on a default LDA to confront
- Define a metric to evaluate the coherence for measuring our model results
    - This metric will later be used to study the best HP for the CAt model
- Tune the model + the LDA model
- Compare the two and make conclusions