prompt = ('The possible labels are positive or negative. '
          'What is the label of the following movie review? '
          'Review: I enjoyed this movie!!! ' 
          'Label: ')

print(prompt)

from transformers import pipeline
generator = pipeline('text-generation', model = 'gpt2', cache_dir='./cache')

text = generator(prompt, max_length=40)

text_ = text[0]['generated_text']

print(text_)