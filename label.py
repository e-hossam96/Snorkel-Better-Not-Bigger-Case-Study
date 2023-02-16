import argparse

from transformers import pipeline
from datasets import load_dataset


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, 
                        help="Zero-shot classification models from hugging face."
                        "Example: facebook/bart-large-mnli")
    
    args = parser.parse_args()
    return args


def main():
    
    args = parse()
    
    print(f'Arguments: {args}')
    
    imdb_data = load_dataset('imdb')
    labels = imdb_data['train'].features['label'].names
    
    int2str = {i: j for i, j in enumerate(labels)}
    str2int = {j: i for i, j in enumerate(labels)}
    
    imdb_data['train'] = imdb_data['train'].shuffle(seed=42)
    imdb_data['test'] = imdb_data['test'].shuffle(seed=42)
    
    
    pipe_cls = pipeline("zero-shot-classification",
                          model=args.model_name, device=0)
    
    def get_label(example):
        output = pipe_cls(example['text'], labels)
        label = output['labels'][0]
        score = output['scores'][0]
        if score > 0.50:
            return {args.model_name: str2int[label]}
        else:
            return {args.model_name: -1}
    
    imdb_data['train'] = imdb_data['train'].map(get_label)
    imdb_data['train'].to_json(f'imdb_data/{args.model_name}/train.json')
    
    
    imdb_data['test'] = imdb_data['test'].map(get_label)
    imdb_data['test'].to_json(f'imdb_data/{args.model_name}/test.json')

    
if __name__ == '__main__':
    main()