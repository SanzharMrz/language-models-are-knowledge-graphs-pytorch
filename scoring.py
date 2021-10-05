import os
import pickle
import argparse

import pandas as pd



def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predict_path", default='triplets_filtered.pkl', help="path to some predict pkl file")
    parser.add_argument("--target_path", default='docred_triplets_filtered.csv', help="path to some target csv file")
    parser.add_argument("--out_path", default='', help="path to save folder")
    args = parser.parse_args()
    return args

  
def scoring(args):

    predict_path = args.predict_path
    target_path = args.target_path
    out_path = args.target_path

    with open(predict_path, "rb") as file:
        flt_trp_filtered = pickle.load(file)

    df = pd.read_csv(target_path)

    tp, fp, fn = 0, 0, 0
    fps_dict = {}
    for sentence, triplets_predicted in flt_trp_filtered.items():

        triplets = [
            (trip[0], trip[1])
            for trip in eval(df[df.text == sentence].triplets.values[0])
        ]
        fps = []
        triplets_predicted = [(trip["h"], trip["t"]) for trip in triplets_predicted]

        for trip in triplets_predicted:
            if trip in triplets:
                tp += 1
            else:
                fps.append(trip)
                fp += 1

        fps_dict[sentence] = fps

        for trip in triplets:
            if trip not in triplets_predicted:
                fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    
    if not os.path.exists(out_path):
      os.mkdir(out_path)
      
    fps_path = os.path.join(out_path, "fps.pkl")
    csv_path = os.path.join(out_path, "metrics.csv")
    
    with open(fps_path, "wb") as file:
        pickle.dump(fps_dict, file)

    pd.DataFrame(metrics.items(), columns=["metrics", "scores"], index=False).to_csv(
        csv_path
    )
    return 0
  
  
if __name__ == '__main__':
  args = get_args()
  scoring(args)
