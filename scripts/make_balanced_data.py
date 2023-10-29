import argparse
import pandas as pd
import random

def build_balance_data(args):
    data = pd.read_csv(args.data_path)

    labels = data["utt_spk_int_labels"].values
    name = data["speaker_name"].values
    paths = data["utt_paths"].values
    durations = data["durations"].values
    
    dict_name = {}
    dict_paths = {}
    dict_durations = {}
    for idx, label in enumerate(labels):
        if label not in dict_paths:
            dict_name[label] = name[idx]
            dict_paths[label] = []
            dict_durations[label] = []
        if abs(durations[idx] - 9) < 3:
            dict_paths[label].append(paths[idx])
            dict_durations[label].append(durations[idx])


    candi_spk = []
    for label in range(max(labels) + 1):
        if args.utt_per_spk <= len(dict_paths[label]):
            candi_spk.append(label)
    random_num_spk = random.sample(candi_spk, args.num_spk)


    result_name = []
    result_paths = []
    result_durations = []
    result_labels = []
    for label in random_num_spk:
        candi_utt = [i for i in range(len(dict_paths[label]))]
        random_utt_per_spk = random.sample(candi_utt, args.utt_per_spk)
        result_labels.extend([label] * args.utt_per_spk)
        for idx in random_utt_per_spk:  
            result_name.append(dict_name[label])
            result_paths.append(dict_paths[label][idx])
            result_durations.append(dict_durations[label][idx])

    table = {}
    for idx, label in enumerate(set(result_labels)):
        table[label] = idx

    labels = []
    for label in result_labels:
        labels.append(table[label])

    dic = {'speaker_name': result_name, 'utt_paths': result_paths, 'utt_spk_int_labels': labels, 'durations': result_durations}    
    df = pd.DataFrame(dic)
    df.to_csv(args.save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/train.csv")
    parser.add_argument('--save_path', type=str, default="balance.csv")
    parser.add_argument('--num_spk', type=int, default=1211)
    parser.add_argument('--utt_per_spk', type=int, default=122)
    args = parser.parse_args()

    build_balance_data(args)
