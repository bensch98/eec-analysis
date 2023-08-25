import igl
import numpy as np 
import open3d as o3d
import ot
from scipy.stats import wasserstein_distance

EPSILON = 1e-8

def cotangent_weights(vertices, faces): return igl.cotmatrix(vertices, faces)

def laplace_beltrami_operator(vertices, faces, cotangent_weights):
  n_vertices = len(vertices)
  L = np.zeros((n_vertices, n_vertices), dtype=np.float64)
  for face in faces:
    for i in range(3):
      vi, vj = face[i], face[(i + 1) % 3]
      L[vi, vj] -= cotangent_weights[(vi, vj)]
      L[vj, vi] -= cotangent_weights[(vj, vi)]
      L[vi, vi] += cotangent_weights[(vi, vj)]
      L[vj, vj] += cotangent_weights[(vj, vi)]
  return L

def eigen(L):
  eigvals, eigvecs = np.linalg.eigh(L)
  return np.clip(eigvals, a_min=0, a_max=None), eigvecs

def compute_hks(vertices, faces, time_scales):
  cotmatrix = cotangent_weights(vertices, faces)
  L = laplace_beltrami_operator(vertices, faces, cotmatrix)
  eigvals, eigvecs = eigen(L)
  hks_matrix = np.zeros((len(vertices), len(time_scales)), dtype=np.float64)
  for t_idx, t in enumerate(time_scales):
    k_t = eigvecs @ np.diag(np.exp(-eigvals * t)) @ eigvecs.T
    hks = np.diag(k_t)
    hks_matrix[:, t_idx] = hks
  return hks_matrix

def dynamic_time_warping(seq1, seq2):
  n, m = len(seq1), len(seq2)
  dtw_matrix = np.zeros((n+1, m+1))
  dtw_matrix[0, 1:] = float("inf")
  dtw_matrix[1:, 0] = float("inf")
  cost_matrix = np.abs(seq1[:, None] - seq2[None, :])
  max_possible_cost = cost_matrix.sum()
  for i in range(1, n+1):
    min_previous = np.min([dtw_matrix[i-1, 1:], dtw_matrix[i, :-1], dtw_matrix[i-1, :-1]])
    dtw_matrix[i, 1:] = cost_matrix[i-1] + min_previous
  normalized_dtw_distance = dtw_matrix[-1, -1] / max_possible_cost
  return dtw_matrix[-1, -1], normalized_dtw_distance

def compare_hks(hks1, hks2, n_time_scales):
  total_distance, total_normalized_distance = 0, 0
  print("Dynamic Time Warping")
  for t in range(n_time_scales):
    hks1_t = hks1[:, t]
    hks2_t = hks2[:, t]
    distance_t, normalized_distance_t = dynamic_time_warping(hks1_t, hks2_t)
    total_distance += distance_t
    total_normalized_distance += normalized_distance_t
    print("Dynamic Time Warping: iteration")
  return total_distance / n_time_scales, total_normalized_distance / n_time_scales

def pairwise_distances(X): return np.linalg.norm(X[:, None] - X[None, :], axis=-1)
def shape_distribution_distance(x, y):
  n_samples = len(x)
  dist_x = pairwise_distances(x)
  dist_y = pairwise_distances(y)
  samples_x = np.random.choice(dist_x.flatten(), n_samples)
  samples_y = np.random.choice(dist_y.flatten(), n_samples)
  return wasserstein_distance(samples_x, samples_y)

def optimal_transport_distance(pcd1, pcd2):
  weights_shape1 = np.ones(len(pcd1)) / len(pcd1)
  weights_shape2 = np.ones(len(pcd2)) / len(pcd2)
  distance_matrix = ot.dist(pcd1, pcd2)
  return ot.emd2(weights_shape1, weights_shape2, distance_matrix, numItermax=1000000)

def arr2samples_distance(v1, v2):
  num_point1, _ = v1.shape
  expanded_v1 = np.expand_dims(v1, axis=1)
  differences = expanded_v1 - v2
  squared_differences = differences ** 2
  squared_distances = np.sum(squared_differences, axis=-1)
  distances = np.sqrt(squared_distances)
  min_distances = np.min(distances, axis=1)
  return np.mean(min_distances)

def chamfer_distance(v1, v2):
  av_dist1 = arr2samples_distance(v1, v2)
  av_dist2 = arr2samples_distance(v2, v1)
  return (av_dist1 + av_dist2) / 2