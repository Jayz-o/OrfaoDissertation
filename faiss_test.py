import faiss
import numpy as np
from sklearn.metrics import silhouette_score

from DatasetProcessing.KeyFrameExtraction.KFHelpers.keyframe_helper import load_path_feature_dic

if __name__ == '__main__':

    try:

        path_feature_dic = load_path_feature_dic("/home/jarred/Documents/Datasets/SIW_KF/train/real/165/165-2-1-2-2")
        feature_array = np.stack(list(path_feature_dic.values()))
        k = 1000
        feature_array

        max_iter = 100
        n_init = 3
        print(k)

        import torch
        from kmeans_pytorch import kmeans

        print(k)
        # labels, cluster_centers = kmeans(torch.from_numpy(feature_array), num_clusters=k, device=torch.device('cuda:0'))
        # score = silhouette_score(feature_array, labels, metric='euclidean')

        kmeans = faiss.Kmeans(d=feature_array.shape[1], k=k, gpu=True, verbose=True)
        kmeans.train(feature_array)
        cluster_centers = kmeans.centroids
        D, I = kmeans.index.search(x=feature_array, k=1)
        I = I.reshape(-1)
        score = silhouette_score(feature_array, I, metric='euclidean')

        del kmeans
        # del labels
    except Exception as e:
        print(e)