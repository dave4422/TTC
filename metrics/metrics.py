from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import torch


def entropy(input):

    _, counts = np.unique(input, return_counts=True)

    probabilities = counts / counts.sum()

    return -np.sum(probabilities * np.log2(probabilities))


def generate_sphere_points(dim, num_points, radius=1):


    points = np.random.randn(num_points, dim)


    norms = np.linalg.norm(points, axis=1)
    points_normalized = points / norms[:, np.newaxis]

    # Scale points to the desired radius
    points_on_sphere = points_normalized * radius

    return points_on_sphere


def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean()


def calculate_sample_alignment_distance(similarities, n_samples, labels):

    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]

    dist_to_positives = np.zeros(n_samples)

    for i in range(n_samples):
        indices_pos = [
            i + (j+1)*n_samples for j in range((similarities.shape[0] // n_samples)-1)]
        dist_to_positives[i] = similarities[i, indices_pos].mean()

    # Compute mean distance for each class based on original label arrangement
    sad_0 = dist_to_positives[indices_0]
    sad_1 = dist_to_positives[indices_1]

    return sad_0, sad_1

def calculate_sample_alignment_distance_new(similarities, n_samples, labels):
    sad_0, sad_1 = calculate_sample_alignment_distance(similarities, n_samples, labels)
    return np.concatenate([sad_0,sad_1], axis=0)


def calculate_sample_alignment_accuracy(sim, n_samples, labels,all_labels):
    index_top1_nn = np.argmin(sim, axis=1)
    acc = 0.0
    acc_1_cls0 = 0.0
    acc_1_cls1 = 0.0

    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]

    acc_1_array_cls0 = np.zeros(len(index_top1_nn))
    acc_1_array_cls1 = np.zeros(len(index_top1_nn))
    for i, idx in enumerate(index_top1_nn):
        if i % n_samples == idx % n_samples:
            acc += 1
            if labels[i % n_samples] == 0:
                acc_1_cls0 += 1
                acc_1_array_cls0[i % n_samples] += 1
            else:
                acc_1_cls1 += 1
                acc_1_array_cls1[i % n_samples] += 1

    acc_1_cls0 = (acc_1_cls0 / indices_0.shape[0])*100.0
    acc_1_cls1 = (acc_1_cls1 / indices_1.shape[0])*100.0

    return acc_1_cls0, acc_1_cls1

def calculate_sample_alignment_accuracy_new(sim, n_samples, labels,all_labels):
    acc_1_cls0, acc_1_cls1 = calculate_sample_alignment_accuracy(sim, n_samples, labels,all_labels)
    return np.concatenate([acc_1_cls0,acc_1_cls1], axis=0)



def _compute_local_neighborhood_accuracies(sim, inidices_0, inidices_1, all_embeddings, all_labels, r=0.05):

    # avg dist med
    med_n = int(all_embeddings.shape[0] * r)
    avg_dist_med_0 = []
    avg_dist_med_1 = []
    avg_dist_med_all_lst = []

    index_sorted = np.argsort(sim, axis=1)

    sim_0 = sim[inidices_0, :][:, inidices_0]
    sim_1 = sim[inidices_1, :][:, inidices_1]

    index_sorted_0 = np.argsort(sim_0, axis=1)
    index_sorted_1 = np.argsort(sim_1, axis=1)

    acc_med = []

    for i, idx in enumerate(index_sorted):

        avg_dist_med_all_lst.append(
            np.mean(sim[i, :][index_sorted[i, :med_n]]))
        acc_med.append(all_labels[index_sorted[i, :med_n]])

    acc_med = np.array(acc_med)

    for i, idx in enumerate(index_sorted_0):
        avg_dist_med_0.append(np.mean(sim_0[i, :][index_sorted_0[i, :med_n]]))

    for i, idx in enumerate(index_sorted_1):
        avg_dist_med_1.append(np.mean(sim_1[i, :][index_sorted_1[i, :med_n]]))

    return np.array(avg_dist_med_0), np.array(avg_dist_med_1), acc_med


def calculate_class_alignment_distance(sim, all_embeddings, all_labels):

    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]
    return _compute_local_neighborhood_accuracies(sim, indices_0, indices_1, all_embeddings, all_labels, r=1.0)[:2]

def calculate_class_alignment_distance_new(sim, all_embeddings, all_labels):
    sad_0, sad_1 = calculate_class_alignment_distance(sim, all_embeddings, all_labels)
    return np.concatenate([sad_0,sad_1], axis=0)

def calculate_class_alignment_consistency(sim, all_embeddings, all_labels):

    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]
    med_n = int(all_embeddings.shape[0] * 0.05)

    acc_med = _compute_local_neighborhood_accuracies(
        sim, indices_0, indices_1, all_embeddings, all_labels, r=0.05)[-1]

    acc_med = np.array(acc_med)
    acc_med_0 = acc_med[indices_0].sum(axis=1) / med_n
    acc_med_0 = 1.0 - acc_med_0
    acc_med_1 = acc_med[indices_1].sum(axis=1) / med_n

    return acc_med_0*100.0, acc_med_1*100.0


def calculate_gaussian_potential_uniformity(all_embeddings, all_labels):

    # Convert numpy arrays to PyTorch tensors. Necessary if lunif works with PyTorch tensors.

    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]

    return lunif(torch.from_numpy(all_embeddings[indices_0])).item(), lunif(torch.from_numpy(all_embeddings[indices_1])).item()


def calculate_probabilistic_entropy_uniformity(all_embeddings, all_labels, radius=1):

    indices_0 = np.where(all_labels == 0)[0]
    indices_1 = np.where(all_labels == 1)[0]
    uniformity_entropies = []
    uniformity_entropies_cls0 = []
    uniformity_entropies_cls1 = []

    for _ in range(5):
        points = generate_sphere_points(
            128, 10 * all_embeddings.shape[0], radius)

        total_ps = points.shape[0]

        sim_points = pairwise_distances(all_embeddings, points, metric="l2")

        closest_point = np.argmin(sim_points, axis=1)
        closest_point_min = closest_point[indices_0]
        closest_point_maj = closest_point[indices_1]

        entropy_all = entropy(closest_point) / \
            entropy(np.arange(closest_point.shape[0]))
        entropy_cls0 = entropy(closest_point_min) / \
            entropy(np.arange(closest_point_min.shape[0]))
        entropy_cls1 = entropy(closest_point_maj) / \
            entropy(np.arange(closest_point_maj.shape[0]))

      
        uniformity_entropies.append(entropy_all)
        uniformity_entropies_cls0.append(entropy_cls0)
        uniformity_entropies_cls1.append(entropy_cls1)

    # Calculate mean and std of the metrics
    mean_entropy_all = np.mean(uniformity_entropies)
    std_entropy_all = np.std(uniformity_entropies)

    mean_entropy_cls0 = np.mean(uniformity_entropies_cls0)
    std_entropy_cls0 = np.std(uniformity_entropies_cls0)

    mean_entropy_cls1 = np.mean(uniformity_entropies_cls1)
    std_entropy_cls1 = np.std(uniformity_entropies_cls1)

    return 100*np.array(uniformity_entropies_cls0), 100*np.array(uniformity_entropies_cls1)
