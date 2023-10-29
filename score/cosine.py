import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity
import torch

def cosine_score(trials, index_mapping, eval_vectors):
    labels = []
    scores = []
    eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
    for item in trials:
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        labels.append(int(item[0]))
        scores.append(score)
    return labels, scores


def np_topk(arr, k):
    indices = np.argsort(arr)
    topk_indices = indices[-k:]
    return arr[topk_indices], topk_indices


def as_norm(score, embedding_1, embedding_2, cohort_feats, topk):
    #https://github.com/TaoRuijie/ECAPA-TDNN/issues/38#issuecomment-1271779955
    
    #print("cohort_feats.shape: {}".format(cohort_feats.shape)) (3000, 256)
    #print("embedding_1.shape: {}".format(embedding_1.shape)) (256,)
    score_1 = cosine_similarity(cohort_feats, embedding_1.reshape(1, -1))[:, 0]
    score_1 = np_topk(score_1, topk)[0]
    mean_1 = np.mean(score_1, axis=0)
    std_1 = np.std(score_1, axis=0)
    score_2 = cosine_similarity(cohort_feats, embedding_2.reshape(1, -1))[:, 0]
    score_2 = np_topk(score_2, topk)[0]
    mean_2 = np.mean(score_2, axis=0)
    std_2 = np.std(score_2, axis=0)
    score = 0.5 * (score - mean_1) / std_1 + 0.5 * (score - mean_2) / std_2
    return score


def calculate_mean_std(cohort_feats, embeddings, eval_path, topk):
    cohort_feats = torch.from_numpy(cohort_feats)
    embeddings = torch.from_numpy(embeddings)
    if torch.cuda.is_available():
        cohort_feats = cohort_feats.cuda()
        embeddings = embeddings.cuda()

    mean_std = {}
    for i in range(embeddings.shape[0]):
        score = cosine_similarity(cohort_feats, embeddings[i], dim=1)
        score = torch.topk(score, topk, dim = 0)[0]
        mean = torch.mean(score, dim = 0)
        std = torch.std(score, dim = 0)
        mean_std[eval_path[i]] = (mean.cpu().numpy(), std.cpu().numpy())
    return mean_std


def cosine_score_asnorm(trials, index_mapping, vectors, cohort_path, topk):
    cohort_feats = []
    for item in cohort_path:
        cohort_feats.append(vectors[index_mapping[item]])
    cohort_feats = np.array(cohort_feats)
    #cohort_feats = cohort_feats / np.linalg.norm(cohort_feats, axis=1, keepdims=True)
    
    labels = []
    scores = []
    eval_vectors = []
    eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
    for item in eval_path:
        eval_vectors.append(vectors[index_mapping[item]])
    eval_vectors = np.array(eval_vectors)
    eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)

    for i in range(eval_vectors.shape[0]):
        vectors[index_mapping[eval_path[i]]] = eval_vectors[i]

    mean_std = calculate_mean_std(cohort_feats, eval_vectors, eval_path, topk)

    for item in trials:
        enroll_vector = vectors[index_mapping[item[1]]]
        test_vector = vectors[index_mapping[item[2]]]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        # asnorm
        mean_1 = mean_std[item[1]][0]
        std_1 = mean_std[item[1]][1]
        mean_2 = mean_std[item[2]][0]
        std_2 = mean_std[item[2]][1]
        score = 0.5 * (score - mean_1) / std_1 + 0.5 * (score - mean_2) / std_2
        labels.append(int(item[0]))
        scores.append(score)
    return labels, scores
