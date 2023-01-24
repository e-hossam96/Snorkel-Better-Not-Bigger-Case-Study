from transformers import pipeline
from datasets import load_dataset


def main():
    
    ledgar = load_dataset('lex_glue', name='ledgar')
    labels = ledgar['train'].features['label'].names
    
    int2str = {i: j for i, j in enumerate(labels)}
    str2int = {j: i for i, j in enumerate(labels)}
    
    ledgar['train'] = ledgar['train'].shuffle(seed=42).select(range(10))
    ledgar['validation'] = ledgar['validation'].shuffle(seed=42).select(range(5))
    
    
    pipe_cls = pipeline("zero-shot-classification",
                          model="valhalla/distilbart-mnli-12-1", device=0)
    
    def get_label(example):
        output = pipe_cls(example['text'], labels)
        label = output['labels'][0]
        score = output['scores'][0]
        if score >= 0.5:
            return {'dbert': str2int[label]}
        else:
            return {'dbert': -1}
    
    ledgar['train'] = ledgar['train'].map(get_label)
    ledgar['validation'] = ledgar['validation'].map(get_label)
    
    ledgar['train'].save_to_disk('train')
    ledgar['validation'].save_to_disk('valid')

    
if __name__ == '__main__':
    main()