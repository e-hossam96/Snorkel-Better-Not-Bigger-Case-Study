# Snorkel Better not Bigger

Replication of the case study Better not Bigger from Snorkel [blog](https://snorkel.ai/better-not-bigger-how-to-get-gpt-3-quality-at-0-1-the-cost/).

## Labeling Functions

We used zero-shot classification models as labeling functions to label the [IMDb](https://huggingface.co/datasets/imdb) Dataset and set the confidence score to 0.5 or higher for all the models.

| #   | Zero-Shot Model                                                                              | Emp. Acc. |
| --- | -------------------------------------------------------------------------------------------- | --------- |
| 1   | [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)                           | 0.62      |
| 2   | [xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli)              | 0.59      |
| 3   | [mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | 0.73      |
| 4   | [camembert-base-xnli](https://huggingface.co/BaptisteDoyen/camembert-base-xnli)              | 0.49      |

The last labeling function of `camembert` will be dropped as it has lower than random accuracy.

More stats can be found in the relevant notebook [LF Analysis](./lf_analysis.ipynb)

## Results

| Method            | F1 Score | Accuracy |
| ----------------- | -------- | -------- |
| Supervised        | 0.94     | 0.94     |
| Weakly Supervised | 0.73     | 0.74     |

ML Model used: [BERT Base Uncased](https://huggingface.co/bert-base-uncased)
