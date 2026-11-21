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

## Model and training data

- Model: https://huggingface.co/nehatanzilaahsan/swedish-grammaticality-svr
- Training data: https://huggingface.co/datasets/nehatanzilaahsan/swedish-grammaticality-training

## Run prediction

From the repository root, install the dependencies and download the trained SVR:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
curl -fL https://huggingface.co/nehatanzilaahsan/swedish-grammaticality-svr/resolve/main/svr.joblib -o svr.joblib
python predict.py --input test.jsonl --output scores.jsonl
```

Each line of `test.jsonl` must contain an `id` and a `text` field.
The script downloads the ModernBERT encoder on first use and writes
one `id` and `score` per input line.

## Reproduce training

With the same dependencies installed, download the 578-example training
set to the path expected by `train.py`:

```bash
mkdir -p data
curl -fL https://huggingface.co/datasets/nehatanzilaahsan/swedish-grammaticality-training/resolve/main/train_augmented.jsonl -o data/train.jsonl
python train.py
```

Training uses seed 0 and saves the fitted regressor as `svr.joblib`.
