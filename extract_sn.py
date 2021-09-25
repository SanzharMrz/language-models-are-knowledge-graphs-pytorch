from process import parse_sentence
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import en_core_web_sm

if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    selected_model = 'bert-base-cased'

    use_cuda = False

    tokenizer = AutoTokenizer.from_pretrained(selected_model)

    encoder = GPT2Model.from_pretrained(
        selected_model) if 'gpt' in selected_model.lower() else BertModel.from_pretrained(selected_model)
    encoder = encoder.cuda() if use_cuda else encoder.cpu()
    encoder.eval()

    sentence = """
    MIPT has an original emblem, which embodies its devotion to science. Every 5 years MIPT marks two anniversaries, 
    celebrating the creation of the Department of Physics and Technology at Moscow State 
    University on November 25, 1946 and the creation of Moscow Institute of Physics and Technology, 
    which took place five years later.
    """
    triplets = parse_sentence(sentence, tokenizer, encoder, nlp, use_cuda)
    for triplet in triplets:
        score = triplet.get('c')
        if len(triplet.get('r')[0]) > 1 and score > 0.05:
            print(triplet)
