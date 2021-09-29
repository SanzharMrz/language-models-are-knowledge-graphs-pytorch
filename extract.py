import sys, os
from mapper import Map, deduplication
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import en_core_web_md
from tqdm import tqdm
import json
import pickle
from deeppavlov import configs, build_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process lines of text corpus into knowledgraph')
parser.add_argument('--input_filename', type=str, help='text file as input')
parser.add_argument('--output_filename', type=str, help='output text file')
parser.add_argument('--language_model',default='bert-base-cased', 
                    choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                    help='which language model to use')
parser.add_argument('--use_cuda', default=True, 
                        type=str2bool, nargs='?',
                        help="Use cuda?")
parser.add_argument('--include_text_output', default=True, 
                        type=str2bool, nargs='?',
                        help="Include original sentence in output")
parser.add_argument('--threshold', default=0.003, 
                        type=float, help="Any attention score lower than this is removed")

args = parser.parse_args()

use_cuda = args.use_cuda
nlp = en_core_web_md.load()
el_model = build_model(configs.kbqa.entity_linking_eng, download=False)

'''Create
Tested language model:

1. bert-base-cased

2. gpt2-medium

Basically any model that belongs to this family should work

'''

language_model = args.language_model


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    if 'gpt2' in language_model:
        encoder = GPT2Model.from_pretrained(language_model)
    else:
        encoder = BertModel.from_pretrained(language_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()    
    input_filename = args.input_filename
    output_filename = args.output_filename
    include_sentence = args.include_text_output

    with open(output_filename, 'w') as g:
        with open(input_filename, "rb") as f:
            parsed_data = pickle.load(f)
            for sentence, triplets in parsed_data.items():
                sentence  = sentence.strip()
                if len(sentence):
                    if len(triplets) > 0:
                        # Map
                        entities, _, wiki_ids = el_model([sentence])
                        entity_wiki_dict = dict(zip(entities[0], wiki_ids[0]))
                        mapped_triplets = []
                        for triplet in triplets:
                            head = triplet['h']
                            tail = triplet['t']
                            relations = triplet['r']
                            conf = triplet['c']
                            if conf < args.threshold:
                                continue
                            mapped_triplet = Map(head, relations, tail, entity_wiki_dict)
                            if 'h' in mapped_triplet:
                                mapped_triplet['c'] = conf
                                mapped_triplets.append(mapped_triplet)
                        output = {'tri': deduplication(mapped_triplets)}

                        if include_sentence:
                            output['sent'] = sentence
                        if len(output['tri']) > 0:
                            g.write(json.dumps( output )+'\n')
