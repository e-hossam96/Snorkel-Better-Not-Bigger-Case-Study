from transformers import pipeline
from datasets import load_dataset
from snorkel.preprocess import preprocessor
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
import numpy as np


def main():
    
    ledgar = load_dataset('lex_glue', name='ledgar')
    labels = ledgar['train'].features['label'].names
    
    int2str = {i: j for i, j in enumerate(labels)}
    str2int = {j: i for i, j in enumerate(labels)}
    
    ledgar['train'] = ledgar['train'].shuffle(seed=42).select(range(1000))
    ledgar['validation'] = ledgar['validation'].shuffle(seed=42).select(range(200))
    train_df = ledgar['train'].to_pandas()
    valid_df = ledgar['validation'].to_pandas()
    
    dbert_cls = pipeline("zero-shot-classification",
                          model="typeform/distilbert-base-uncased-mnli", device=0)

    @preprocessor(memoize=True)
    def get_label_dbert(example):
        output = dbert_cls(example['text'], labels)
        label = output['labels'][0]
        score = output['scores'][0]
        example.label_ = label
        example.score_ = score
        return example


    @labeling_function(pre=[get_label_dbert])
    def label_dbert(example):
        if example.score_ > 0.5:
            return str2int[example.label_]
        else:
            return -1
        
    dbart_129_cls = pipeline("zero-shot-classification",
                          model="valhalla/distilbart-mnli-12-9", device=0)

    @preprocessor(memoize=True)
    def get_label_dbart_129(example):
        output = dbart_129_cls(example['text'], labels)
        label = output['labels'][0]
        score = output['scores'][0]
        example.label_ = label
        example.score_ = score
        return example


    @labeling_function(pre=[get_label_dbart_129])
    def label_dbart_129(example):
        if example.score_ > 0.5:
            return str2int[example.label_]
        else:
            return -1
    
    dbart_121_cls = pipeline("zero-shot-classification",
                          model="valhalla/distilbart-mnli-12-1", device=0)

    @preprocessor(memoize=True)
    def get_label_dbart_121(example):
        output = dbart_121_cls(example['text'], labels)
        label = output['labels'][0]
        score = output['scores'][0]
        example.label_ = label
        example.score_ = score
        return example


    @labeling_function(pre=[get_label_dbart_121])
    def label_dbart_121(example):
        if example.score_ > 0.5:
            return str2int[example.label_]
        else:
            return -1

    
    applier = PandasLFApplier([label_dbert, label_dbart_129, label_dbart_121])

    L_train = applier.apply(train_df)
    L_valid = applier.apply(valid_df)
    
    label_model = LabelModel(cardinality=100, verbose=True).to('cuda')
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=42)
    
    gold = np.array(ledgar['validation'][:]['label'])
    
    print(label_model.score(L_valid, gold))

    
if __name__ == '__main__':
    main()