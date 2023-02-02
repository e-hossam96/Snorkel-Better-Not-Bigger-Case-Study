import argparse

from transformers import pipeline
from datasets import load_dataset


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, 
                        help="Zero-shot classification models from hugging face."
                        "Example: facebook/bart-large-mnli")
    
    parser.add_argument("--start_idx", default=0, type=int, required=True, 
                        help="Starting range of the operation.")
    
    parser.add_argument("--end_idx", default=None, type=int, required=True, 
                        help="Ending range of the operation.")
    
    args = parser.parse_args()
    return args


def main():
    
    args = parse()
    
    print(f'Arguments: {args}')
    
    ledgar = load_dataset('lex_glue', name='ledgar')
    labels = ledgar['train'].features['label'].names
    
    int2str = {i: j for i, j in enumerate(labels)}
    str2int = {j: i for i, j in enumerate(labels)}
    
    ledgar['train'] = ledgar['train'].select(range(args.start_idx, args.end_idx))
    ledgar['validation'] = ledgar['validation'].select(range(args.start_idx, args.end_idx))
    
    
    pipe_cls = pipeline("zero-shot-classification",
                          model=args.model_name, device=0)
    
    def get_label(example):
        output = pipe_cls(example['text'], labels)
        label = output['labels'][0]
        score = output['scores'][0]
        if score >= 0.5:
            return {args.model_name: str2int[label]}
        else:
            return {args.model_name: -1}
    
    ledgar['train'] = ledgar['train'].map(get_label)
    ledgar['train'].to_json(f'data/train_{args.start_idx}_{args.end_idx}.json')
    
    
    ledgar['validation'] = ledgar['validation'].map(get_label)
    ledgar['validation'].to_json(f'data/valid_{args.start_idx}_{args.end_idx}.json')

    
if __name__ == '__main__':
    main()