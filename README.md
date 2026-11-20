# Swedish Grammaticality Scoring

I tested whether adding Swedish article-agreement examples improves
the AI Sweden starter model. The model uses ModernBERT embeddings
and support vector regression to predict scores between 0 and 1.

I added 100 examples with correct or swapped en/ett articles, bringing
the training set from 478 to 578 examples. On the same 120 public test
examples, Spearman increased from 0.528 to 0.552 and ROC-AUC from
0.783 to 0.788.

The notebook shows how I generated the extra examples, trained both
versions, evaluated them, and inspected prediction errors.

Starter code: https://github.com/nicher92/swedish_correctness_classifier
