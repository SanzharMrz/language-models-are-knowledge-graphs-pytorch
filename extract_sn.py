from process import parse_sentence
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import time
import pandas as pd
import en_core_web_sm

if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    selected_model = 'bert-base-cased'

    use_cuda = True

    tokenizer = AutoTokenizer.from_pretrained(selected_model)

    encoder = GPT2Model.from_pretrained(
        selected_model) if 'gpt' in selected_model.lower() else BertModel.from_pretrained(selected_model)
    encoder = encoder.cuda() if use_cuda else encoder.cpu()
    encoder.eval()

    data = pd.read_csv('texts.csv')
    elapsed_time, parsed_triplets = [], []
    
    for text in data.text.values:
        triplets = parse_sentence(text, tokenizer, encoder, nlp, use_cuda)
        start_time = time.time()
        parsed_triplets.append([triplet for triplet in triplets if len(triplet.get('r')[0]) > 1 and triplet.get('c') > 0.05])
        parsing_time = time.time() - start_time
        elapsed_time.append(parsing_time)
        
    data['time'] = elapsed_time
    data['parsing_results'] = parsed_triplets
    data.to_csv('texts_parsed.csv', index=False)

