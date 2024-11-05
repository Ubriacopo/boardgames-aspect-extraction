## IMPROTANTE

- Alla generazione del dataset ho sbagliato qualcosa. Parole che appaiono in generale una volta
  sola devono essere rimossi dal corpus (in particolare link e altre parole inutili) min in 5 docs

> If your model relies heavily on word co-occurrence and contextual embeddings (as in many transformer models), it’s
> generally better to keep these rare words in the vocabulary.
However, if memory efficiency is a critical issue or if the rare words significantly inflate the vocabulary, you might
consider selectively removing them based on criteria such as document frequency.

- Remove links and numbers?
## PENDING

- Usare le issue (Sarebbe comdoo)
- Prendo solo commenti da giochi che ne hanno un quantitativo minimo?
    - No non ne vedo il motivo, devo limitare probabilemtne la quantita ma poi verra
    - Faccio massimo 25 pagine per gioco, Sono 2.5k commenti.
    - Devo postprocessare (post download) i dati per rimuovere:
        - Duplicati
        - Rumorosi (Inutili come Kickstarter got e poco informativi)
        - Non inglesi?
        - Testi rotti
- Fare cosa di preprocessing per togliere commenti inutili

- Rifare pre-processing dei testi. Metto anche riga commento numeor per riferimento?
- Scrivere Modello
- Dictionary only on training set and not all dataset

## DONE

[x] Rimuovere punteggiatura da pre-processing

https://github.com/ruidan/Unsupervised-Aspect-Extraction/tree/master