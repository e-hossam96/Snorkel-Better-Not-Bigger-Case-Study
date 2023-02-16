# snorkel-better-not-bigger
Replication of the case study Better not Bigger from Snorkel [blog](https://snorkel.ai/better-not-bigger-how-to-get-gpt-3-quality-at-0-1-the-cost/).

## Zero-Shot Models used
1. [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
2. [xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli)
3. [mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
4. [camembert-base-xnli](https://huggingface.co/BaptisteDoyen/camembert-base-xnli)


## Results
1. Snorkel Labeling Model: 0.84
2. ML Model on Gold Data: 0.86
3. ML on Noisy Data: 0.81

- ML Model used: TF-IDF followed by a Logistic Regression Model.