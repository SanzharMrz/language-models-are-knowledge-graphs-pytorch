from process import parse_sentence
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import time
<<<<<<< HEAD
import pickle
=======
>>>>>>> 5cba80c4ad4fce227578e0bd849d71c7de998908
import pandas as pd
import en_core_web_sm
from tqdm import tqdm

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
    data['time_cuda'] = None
    triplets_dict = {}
    
    for ind, text in tqdm(enumerate(data.text.values), total=data.shape[0]):
        start_time = time.time()
        triplets = parse_sentence(text, tokenizer, encoder, nlp, use_cuda)
        triplets = [triplet for triplet in triplets if (len(triplet.get('r')[0]) > 1) and (triplet.get('c') > 0.05)]
        data.loc[ind, 'time_cuda'] = time.time() - start_time
        data.to_csv('texts_parsed.csv', index=False)
        triplets_dict[text] = triplets
        with open('triplets.pkl', 'wb') as file:
            pickle.dump(triplets_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
