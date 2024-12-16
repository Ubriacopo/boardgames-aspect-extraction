What should I do next?

# Plan: Next steps...

### 2 - Begin writing the report

We might not be done but that doesn't mean we can start working on the report. <br>

> The results must be documented in a short article of not less than 4 pages and no more than 8, composed
> according to the guidelines available here https://www.springernature.com/gp/authors/campaigns/latex-author-support

#### Structure of the paper:

1. Introduction
   Provides an overview of the project and a short discussion on the pertinent literature
2. Research question and methodology
   Provides a clear statement on the goals of the project, an overview of the proposed approach, and a formal
   definition of the problem
3. Experimental results
   Provides an overview of the dataset used for experiments, the metrics used for evaluating performances,
   and the experimental methodology. Presents experimental results as plots and/or tables
4. Concluding remarks
   Provides a critical discussion on the experimental results and some ideas for future work

### 3 - HP Tuning

Sulla base delle metriche che usiamo (distanza degli aspetti dalle parole, coerenza etc..) alleniamo
il modello K volte sul DS per vedere per quali parametri funziona meglio. Dovrei fare un proxy ds? Non credo non essendo
supervised.

#### HP (Provo qualche combinazione non esagerata, non cerchiamo perfezioni)

- max_seq_length (128, 256, 512, FULL)
- embedding_size () -> Cerca modo per sceglierle #todo
- aspect_emb_size -> Come per emb_size cerca
- batch_size
- SGD / adam params ? (nah)

Fare ricerca con kerastuner? #todo vedere se si puo

- (128, 128, 128, 32)
- (256, 128, 128, 32)
- (512, 128, 128, 32)
- (FULL, 128, 128, 32)
- (256, 200, 200, 32)

### 4 - Abae e Abae-

Togli layer di attention e vediamo se cambia qualcosa

### 5 - LDA per confronto con il modello scelto
https://www.kaggle.com/code/pranjalsoni17/topic-modelling-using-lda

### 6 - k-Means come metodo di base per confrontare tutto

### 5 - Delete and refactor

Remove stuff that is not needed anymore.
Refactor the code to make it cleaner.

### 6 - Piccolo dataset di test? ~ 1k records