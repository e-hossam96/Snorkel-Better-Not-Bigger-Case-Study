# Snorkel Better not Bigger

Replication of the case study Better not Bigger from Snorkel [blog](https://snorkel.ai/better-not-bigger-how-to-get-gpt-3-quality-at-0-1-the-cost/).

## Labeling Functions

We used zero-shot classification models as labeling functions to label the [IMDb](https://huggingface.co/datasets/imdb) Dataset and set the confidence score to higher than 0.75 for all the models.

| #   | Zero-Shot Model                                                                              | Emp. Acc. |
| --- | -------------------------------------------------------------------------------------------- | --------- |
| 1   | [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)                           | 0.85      |
| 2   | [xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli)              | 0.68      |
| 3   | [mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | 0.92      |
| 4   | [camembert-base-xnli](https://huggingface.co/BaptisteDoyen/camembert-base-xnli)              | 0.46      |

The last labeling function of `camembert` will be dropped as it has lower than random accuracy.

The **Snorkel Labeling Model** when fitted to the data gave out 0.86 accuracy score.

More stats can be found in the relevant notebook [LF Analysis](./lf_analysis.ipynb)

## Results

| Method            | F1 Score | Accuracy |
| ----------------- | -------- | -------- |
| Supervised        | 0.94     | 0.94     |
| Weakly Supervised | 0.90     | 0.90     |

ML Model used: [BERT Base Uncased](https://huggingface.co/bert-base-uncased)
