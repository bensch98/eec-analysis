import igl
import numpy as np 
import open3d as o3d
import torch

EPSILON = 1e-8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cotangent_weights(vertices, faces): return igl.cotmatrix(vertices, faces)

def laplace_beltrami_operator(vertices, faces, cotangent_weights):
  n_vertices = len(vertices)
  L = torch.zeros((n_vertices, n_vertices), dtype=torch.float64).to(device)
  for face in faces:
    for i in range(3):
      vi, vj = face[i], face[(i + 1) % 3]
      L[vi, vj] -= cotangent_weights[(vi, vj)]
      L[vj, vi] -= cotangent_weights[(vj, vi)]
      L[vi, vi] += cotangent_weights[(vi, vj)]
      L[vj, vj] += cotangent_weights[(vj, vi)]
  return L

def eigen(L):
  eigvals, eigvecs = torch.linalg.eigh(L)
  return torch.clamp(eigvals, min=0).to(device), eigvecs.to(device)

def compute_hks(vertices, faces, time_scales):
  cotmatrix = cotangent_weights(vertices, faces)
  L = laplace_beltrami_operator(vertices, faces, cotmatrix)
  eigvals, eigvecs = eigen(L)
  eigvals = eigvals.to(device)
  eigvecs = eigvecs.to(device)
  hks_matrix = torch.zeros((len(vertices), len(time_scales)), dtype=torch.float64).to(device)
  for t_idx, t in enumerate(time_scales):
    k_t = eigvecs @ torch.diag(torch.exp(-eigvals * t)) @ eigvecs.T
    hks = torch.diag(k_t)
    hks_matrix[:, t_idx] = hks
  return hks_matrix

def dynamic_time_warping(seq1, seq2):
  n, m = len(seq1), len(seq2)
  dtw_matrix = torch.zeros((n+1, m+1)).to(device)
  dtw_matrix[0, 1:] = float("inf")
  dtw_matrix[1:, 0] = float("inf")
  cost_matrix = torch.abs(seq1[:,None] - seq2[None,:]).to(device)
  max_possible_cost = cost_matrix.sum()
  for i in range(1, n+1):
    min_previous = torch.min(torch.min(dtw_matrix[i-1, 1:], dtw_matrix[i, :-1]), dtw_matrix[i-1, :-1]).to(device)
    dtw_matrix[i, 1:] = cost_matrix[i-1] + min_previous
  normalized_dtw_distance = dtw_matrix[-1, -1] / max_possible_cost
  return dtw_matrix[-1, -1], normalized_dtw_distance

def compare_hks(hks1, hks2, n_time_scales):
  total_distance, total_normalized_distance = 0, 0
  for t in range(n_time_scales):
    hks1_t = hks1[:, t]
    hks2_t = hks2[:, t]
    distance_t, normalized_distance_t = dynamic_time_warping(hks1_t, hks2_t)
    total_distance += distance_t
    total_normalized_distance += normalized_distance_t
  return total_distance / n_time_scales, total_normalized_distance / n_time_scales

if __name__ == "__main__":
  fname = "2940207"
  off = o3d.io.read_triangle_mesh(str("data/train/off/" + fname + ".off")).compute_vertex_normals()
  stl = o3d.io.read_triangle_mesh(str("data/train/stl/" + fname + ".stl")).compute_vertex_normals()
  print(f"OFF - n_vertices: {len(off.vertices)}")
  print(f"STL - n_vertices: {len(stl.vertices)}")

  n_time_scales = 16
  time_scales = torch.logspace(-2, 0., steps=n_time_scales).cuda()
  hks_off = compute_hks(np.asarray(off.vertices, dtype=np.float32), np.asarray(off.triangles, dtype=int), time_scales)
  hks_stl = compute_hks(np.asarray(stl.vertices, dtype=np.float32), np.asarray(stl.triangles, dtype=int), time_scales)

  dist, normalized_dist = compare_hks(hks_off, hks_stl, n_time_scales)
  print(dist, normalized_dist)