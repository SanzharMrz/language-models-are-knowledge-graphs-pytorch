from constant import invalid_relations_set


def Map(head, relations, tail, mapped_entities):
    if head == None or tail == None or relations == None:
        return {}
    head_wiki = mapped_entities.get(head.lower(), '')
    tail_wiki = mapped_entities.get(tail.lower(), '')
    valid_relations = [r for r in relations if r not in invalid_relations_set and r.isalpha() and len(r) > 1]
    if head_wiki or tail_wiki:
        if len(valid_relations) == 0:
            return {}
        return { 'h': head, 't': tail, 'r': '_'.join(valid_relations), 'hw': head_wiki, 'tw': tail_wiki  }
    else:
        return {}

def deduplication(triplets):
    unique_pairs = []
    pair_confidence = []
    for t in triplets:
        key = '{}\t{}\t{}\t{}\t{}'.format(t['h'], t['r'], t['t'],t['hw'],t['tw'])
        conf = t['c']
        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)
    
    unique_triplets = []
    for idx, unique_pair in enumerate(unique_pairs):
        h, r, t, hw, tw = unique_pair.split('\t')
        unique_triplets.append({ 'h': h, 'r': r, 't': t , 'c': pair_confidence[idx], 'hw': hw, 'tw':tw})

    return unique_triplets