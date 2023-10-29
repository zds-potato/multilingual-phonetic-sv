import argparse
import pandas as pd
import numpy as np
import random

def random_dataset(args):
    data = pd.read_csv(args.data_list_path)

    labels = data["utt_spk_int_labels"].values
    name = data["speaker_name"].values
    paths = data["utt_paths"].values
    #durations = data["durations"].values

    dict_name = {}
    dict_paths = {}
    #dict_durations = {}
    for idx, label in enumerate(labels):
        if label not in dict_paths:
            dict_name[label] = name[idx]
            dict_paths[label] = []
            #dict_durations[label] = []
        dict_paths[label].append(paths[idx])
        #dict_durations[label].append(durations[idx])


    candi_spk = []
    for label in range(max(labels) + 1):
        if args.utt_per_spk <= len(dict_paths[label]):
            candi_spk.append(label)
    
    random_num_spk = random.sample(candi_spk, args.num_spk)


    result_name = []
    result_paths = []
    #result_durations = []
    result_labels = []
    for label in random_num_spk:
        candi_utt = [i for i in range(len(dict_paths[label]))]
        random_utt_per_spk = random.sample(candi_utt, args.utt_per_spk)
        result_labels.extend([label] * args.utt_per_spk)
        for idx in random_utt_per_spk:  
            result_name.append(dict_name[label])
            result_paths.append(dict_paths[label][idx])  
    
    label_set = set(result_labels)
    table = {}
    for idx, s in enumerate(label_set):
        table[s] = idx
        
    new_labels = []
    for label in result_labels:
        new_labels.append(table[label])

    dic = {'speaker_name': result_name, 'utt_paths': result_paths, 'utt_spk_int_labels': new_labels}
    df = pd.DataFrame(dic)
    df.to_csv(args.tsne_set_save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list_path', type=str, default="data.csv")
    parser.add_argument('--tsne_set_save_path', type=str, default="tsne.csv")
    parser.add_argument('--num_spk', type=int, default=20)
    parser.add_argument('--utt_per_spk', type=int, default=200)
    args = parser.parse_args()

    random_dataset(args)
