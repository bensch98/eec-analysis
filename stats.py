from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import shutil

from irrCAC.raw import CAC
import numpy as np
import open3d as o3d
import pandas as pd
from scipy import stats
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import kurtosis, skew, ttest_ind, mode
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from statsmodels.stats.inter_rater import fleiss_kappa
import torch

import geometry
from features import Component

# global options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', shutil.get_terminal_size().columns)

DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]
NP_DESCRIPTIVE_STATS = ["mean", "median", "std", "var", "min", "max"]
SCIPY_DESCRIPTIVE_STATS = ["kurtosis", "skew"]
FORMATS = ["off", "obj", "stl"]
o3d.utility.random.seed(420)

class Labels:
  scores = ["accuracy", "precision", "recall", "f1_score", "jaccard"]

  def __init__(self, d, n_labels):
    self.d = d
    self.n_raters = len([i for i in (d / "test_raters").iterdir() if i.is_dir()])
    self.n_labels = n_labels
    self.tr_df, self.val_df, self.te_df = self.init_df()
  
  @property
  def dfs(self): return [self.tr_df, self.val_df, self.te_df]

  def init_df(self):
    tr_fnames = [i.stem for i in (self.d / "train" / "txt").glob("*.txt")]
    val_fnames = [i.stem for i in (self.d / "val" / "txt").glob("*.txt")]
    te_fnames = [i.stem for i in (self.d / "test" / "txt").glob("*.txt")]
    self.tr_df = pd.DataFrame({"id": tr_fnames})
    self.val_df = pd.DataFrame({"id": val_fnames})
    self.te_df = pd.DataFrame({"id": te_fnames})
    return self.tr_df, self.val_df, self.te_df

  def _load_txt(self, f): return np.loadtxt(f, dtype=int)

  def load_raters(self):
    self.rater_dfs = {}
    for i in range(self.n_raters):
      df = self.te_df.copy()
      df["labels"] = df.apply(lambda row: self._load_txt(self.d / f"test_raters/{i}/{row['id']}.txt"), axis=1)
      self.rater_dfs[i] = df
  
  def load(self):
    self.tr_df["labels"] = self.tr_df.apply(lambda row: self._load_txt(self.d / f"train/txt/{row['id']}.txt"), axis=1)
    self.val_df["labels"] = self.val_df.apply(lambda row: self._load_txt(self.d / f"val/txt/{row['id']}.txt"), axis=1)

  def _maj_vote(self, row):
    votes = np.array([df.loc[row.name, "labels"] for df in self.rater_dfs.values()])
    mode, _ = stats.mode(votes, axis=0, keepdims=True)
    return mode[0]

  def majority_vote(self):
    self.te_df["labels"] = self.te_df.apply(self._maj_vote, axis=1)
  
  @staticmethod
  def extract_mode(x): return mode(x, keepdims=False).mode
  
  def _compute_descriptive_statistics_row(self, df):
    # descriptive stats
    df["mode"] = df["labels"].apply(Labels.extract_mode)
    df["median"] = df["labels"].apply(np.median).astype(int)
    df["std"] = df["labels"].apply(np.std)
    df["var"] = df["labels"].apply(np.var)
    df["min"] = df["labels"].apply(np.min)
    df["max"] = df["labels"].apply(np.max)
    df['unique'] = df["labels"].apply(np.unique)
    df["kurtosis"] = df["labels"].apply(kurtosis)
    df["skew"] = df["labels"].apply(skew)
    # counts, frequencies
    labels = list(range(self.n_labels))
    for label in labels:
      df[f"count_{label}"] = df["labels"].apply(lambda x: np.count_nonzero(x == label))
    for label in labels:
      df[f"frequency_{label}"] = df[f"count_{label}"] / df["labels"].apply(len)
    return df
  
  def compute_stats(self):
    self.rater_stats_dfs = {rater: self._compute_descriptive_statistics_row(df) for rater, df in self.rater_dfs.items()}
    for df in self.dfs:
      df = self._compute_descriptive_statistics_row(df)

  def _compute_descriptive_stats_all(self, df, src):
    # descriptive stats
    arr = np.concatenate(df["labels"].values)
    stats_dict = {
      "source": src,
      "mode": mode(arr, keepdims=False).mode,
      "median": np.median(arr),
      "std": arr.std(),
      "var": arr.var(),
      "min": arr.min(),
      "max": arr.max(),
      "unique": np.unique(arr),
      "kurtosis": kurtosis(arr),
      "skew": skew(arr)
    }
    # counts, frequencies
    labels = list(range(self.n_labels))
    counts = np.bincount(arr, minlength=len(labels))
    for label, count in enumerate(counts):
      stats_dict[f"count_{label}"] = count
    for label, count in enumerate(counts):
      stats_dict[f"frequency_{label}"] = count / counts.sum()
    return stats_dict

  def compute_individual_statistics(self):
    stats_list = []
    src_default = SPLITS
    src_rater = [f"rater_{i}" for i in range(self.n_raters)]
    # stats for each rater
    for df, src in zip(self.rater_dfs.values(), src_rater):
      stats_list.append(self._compute_descriptive_stats_all(df, src=src))
    # stats for tr/val/te
    for df, src in zip(self.dfs, src_default):
      stats_list.append(self._compute_descriptive_stats_all(df, src=src))
    # total stats for tr/val/te
    stats_list.append(self._compute_descriptive_stats_all(pd.concat(self.dfs, ignore_index=True), src="total"))
    self.stats_df = pd.DataFrame(stats_list)
  
  def compare_rater2gt(self):
    mv_labels = self.te_df["labels"]
    for _, df in self.rater_dfs.items():
      # accuracy, precision, recall, f1-score, jaccard
      df["accuracy"], df["precision"], df["recall"], df["f1_score"], df["jaccard"] = 0, 0, 0, 0, 0
      for idx, row in df.iterrows():
        rater = row["labels"]
        mv = mv_labels[idx]
        # metrics
        df.at[idx, "accuracy"] = accuracy_score(mv, rater)
        df.at[idx, "precision"] = precision_score(mv, rater, average="macro", zero_division=0)
        df.at[idx, "recall"] = recall_score(mv, rater, average="macro", zero_division=0)
        df.at[idx, "f1_score"] = f1_score(mv, rater, average="macro")
        df.at[idx, "jaccard"] = jaccard_score(mv, rater, average="macro")
  
  def compare_mv2avg(self):
    metrics = {metric: [] for metric in self.scores}
    for rater_df in self.rater_dfs.values():
      for score in self.scores:
        metrics[score].append(rater_df[score])
    for score in self.scores:  
      self.te_df[f"avg_{score}"] = np.mean(metrics[score], axis=0)
    
  def top5(self, df, metric): return df.nlargest(5, metric)
  def bottom5(self, df, metric): return df.nsmallest(5, metric)
  def _compute_ratings_matrix(self, arr): return np.apply_along_axis(lambda x: np.bincount(x, minlength=self.n_labels), axis=1, arr=arr)

  def compute_interrater_reliability(self):
    combs = list(combinations(range(self.n_raters), 2))
    index = pd.MultiIndex.from_tuples(combs, names=("rater1", "rater2"))
    df = pd.DataFrame(index=index)
    df[:] = None
    # ttests
    for score in self.scores:
      for comb in combs:
        t_stat, p_val = ttest_ind(self.rater_dfs[comb[0]][score].values, self.rater_dfs[comb[1]][score].values)
        df.loc[comb, f"{score}_t_stat"] = t_stat
        df.loc[comb, f"{score}_p_val"] = p_val

    # cohens kappa
    arrs = [np.concatenate(df["labels"]) for df in self.rater_dfs.values()]
    cohen_kappas = [cohen_kappa_score(arrs[comb[0]], arrs[comb[1]]) for comb in combs]
    df["cohen_kappa"] = cohen_kappas 
    self.interr_df = df

    # fleiss kappa
    self.ratings_arr = np.array(arrs).T
    self.ratings_matrix = self._compute_ratings_matrix(self.ratings_arr)
    fl_kappa = fleiss_kappa(self.ratings_matrix, method="fleiss")
    randolphs_kappa = fleiss_kappa(self.ratings_matrix, method="randolph")
    self.interr_df["fleiss_kappa"] = fl_kappa
    self.interr_df["randolphs_kappa"] = randolphs_kappa
  
  def chance_corrected_agreement_coefficients(self):
    column_names = [i for i in range(self.ratings_arr.shape[1])]
    self.ratings_df = pd.DataFrame(self.ratings_arr, columns=column_names)
    cac = CAC(self.ratings_df, categories=list(range(5)))
    cac.confidence_level = 0.95
    cacs = {}
    cacs["bp"] = cac.bp()
    cacs["conger"] = cac.conger()
    cacs["fleiss"] = cac.fleiss()
    cacs["gwet"] = cac.gwet()
    cacs["krippendorff"] = cac.krippendorff()

    # transform dict into df
    data = []
    for val in cacs.values():
      v = val["est"]
      coeff_name = v["coefficient_name"]
      coeff_value = v["coefficient_value"]
      conf_interval = v["confidence_interval"]
      p_value = v["p_value"]
      pa = v["pa"]
      pe = v["pe"]
      data.append([coeff_name, coeff_value, conf_interval, p_value, pa, pe])
    self.cacs = pd.DataFrame(data, columns=["coefficient", "value", "confidence interval", "p-value", "observed agreement (pa)", "expected agreement (pe)"])
  
  def save_stats(self):
    p = self.d / Path("dataframes/labels")
    if not p.exists():
      p.mkdir()
    [df.drop(columns=["labels"]).to_csv(p / f"{SPLITS[i]}.csv", index=False) for i, df in enumerate(self.dfs)] # train/val/test
    [df.drop(columns=["labels"]).to_csv(p / f"rater_{i}.csv", index=False) for i, df in self.rater_dfs.items()] # rater stats
    self.stats_df.to_csv(p / "stats.csv", index=False) # general stats
    self.interr_df.to_csv(p / "irr.csv", index=False) # inter-rater reliability
    self.cacs.to_csv(p / "cac.csv", index=False) # chance-corrected agreement coefficients
  
  def print_stats(self):
    print("Train/Val/Test:")
    [print(df) for df in self.dfs]
    print("Rater:")
    [print(df) for df in self.rater_dfs.values()]
    print("Top 5: Accuracy")
    print(self.top5(self.te_df, "avg_accuracy"))
    print("Bottom 5: Accuracy")
    print(self.bottom5(self.te_df, "avg_accuracy"))
    print("Stats")
    print(self.stats_df.T)
    print("Interrater-Reliability Analysis")
    print(self.interr_df)
    print("Chance-Corrected Agreement Coefficients")
    print(self.cacs)

class Meshes:
  formats = ["off", "obj", "stl"]
  def __init__(self, d):
    self.d = d
    self.formats = len([i for i in (d / "test_raters").iterdir() if i.is_dir()])
    self.mesh_df = self.init_df()
  
  def init_df(self):
    self.fnames = {}
    self.fnames["train"] = [i.stem for i in (self.d / "train" / "off").glob(f"*.off")]
    self.fnames["val"] = [i.stem for i in (self.d / "val" / "off").glob(f"*.off")]
    self.fnames["test"] = [i.stem for i in (self.d / "test" / "off").glob(f"*.off")]
    #self.fnames["train"] = self.fnames["train"][:1]
    #self.fnames["val"] = self.fnames["val"][:1]
    #self.fnames["test"] = self.fnames["test"][:1]
    mesh_df = pd.DataFrame({
      "id": self.fnames["train"] + self.fnames["val"] + self.fnames["test"],
      "split": ["train"] * len(self.fnames["train"]) + ["val"] * len(self.fnames["val"]) + ["test"] * len(self.fnames["test"])
    })
    return mesh_df
  
  def _load_mesh(self, fpath): return o3d.io.read_triangle_mesh(str(fpath))
  def load(self, fmt):
    self.mesh_df["mesh"] = self.mesh_df.apply(lambda row: pd.Series(self._load_mesh(self.d / f"{row['split']}/{fmt}/{row['id']}.{fmt}")), axis=1)

  @staticmethod
  def vf(mesh): return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int32)

  def _triangle_area(self, v1, v2, v3): return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1))
  def _mesh_surface_area(self, vertices, faces): return sum(self._triangle_area(vertices[f[0]], vertices[f[1]], vertices[f[2]]) for f in faces)
  def _mesh_volume(self, vertices, faces):
    reference_point = np.mean(vertices, axis=0)
    volume = 0
    for face in faces:
      v1, v2, v3 = vertices[face]
      tetrahedron_volume = np.dot(reference_point - v1, np.cross(v2 - v1, v3 - v1)) / 6
      volume += tetrahedron_volume
    return abs(volume)
  
  def _compute_descriptive_stats(self, row):
    vertices, faces = Meshes.vf(row["mesh"])
    # number of vertices, faces
    n_vertices, n_faces = vertices.shape[0], faces.shape[0]
    # mean, std, min, max coordinates
    mean_coord = vertices.mean(axis=0)
    std_coord = vertices.std(axis=0)
    min_coord = vertices.min(axis=0)
    max_coord = vertices.max(axis=0)
    # surface area, volume
    surface_area = self._mesh_surface_area(vertices, faces)
    volume = self._mesh_volume(vertices, faces)

    return pd.Series({
      "n_vertices": n_vertices,
      "n_faces": n_faces,
      "mean_x": mean_coord[0], "mean_y": mean_coord[1], "mean_z": mean_coord[2],
      "std_x": std_coord[0], "std_y": std_coord[1], "std_z": std_coord[2],
      "min_x": min_coord[0], "min_y": min_coord[1], "min_z": min_coord[2],
      "max_x": max_coord[0], "max_y": max_coord[1], "max_z": max_coord[2],
      "width": max_coord[0] - min_coord[0],
      "height": max_coord[1] - min_coord[1],
      "depth": max_coord[2] - min_coord[2],
      "surface_area": surface_area,
      "volume": volume,
    })
  
  def compute_stats(self):
    stats_df = self.mesh_df.apply(self._compute_descriptive_stats, axis=1)
    self.mesh_df = pd.concat([self.mesh_df, stats_df], axis=1)
    self.mesh_df["n_vertices"] = self.mesh_df["n_vertices"].astype(int)
    self.mesh_df["n_faces"] = self.mesh_df["n_faces"].astype(int)
  
  def save_stats(self):
    p = self.d / Path("dataframes/meshes")
    if not p.exists():
      p.mkdir()
    self.mesh_df.drop(columns=["mesh"]).to_csv(p / "mesh.csv", index=False)

  def print_stats(self): print(self.mesh_df)

class FeatureComparison:
  def __init__(self, d, df_meshes, df_raters, n_classes=5):
    self.d = d
    self.df_meshes = df_meshes[df_meshes["split"] == "test"]
    self.n_classes = n_classes
    self.df_raters = {k: pd.merge(self.df_meshes, df, on="id") for k, df in df_raters.items()}
    self.df_raters = {k: df[["id", "split", "mesh", "labels"]] for k, df in self.df_raters.items()}
  
  def _extract_features(self, row, fmt):
    split, mesh_id = row["split"], row["id"]
    comp = Component(self.d, split, fmt, mesh_id, 5)
    comp_features = comp.cluster_components()
    for val in comp_features.values():
      if not val: continue
      for compfeat in val:
        if len(compfeat._mesh.vertices) > 0:
          compfeat.compute_metrics()
    return comp_features

  def extract_features(self, fmt):
    for df in self.df_raters.values():
      df["features"] = df.apply(lambda row: self._extract_features(row, fmt), axis=1)
  
  def convert(self):
    id_values, class_values, feature_values = [], [], []
    for id, features_dict in self.df_raters[0][["id", "features"]].iterrows():
      for class_id, features in features_dict.items():
        for i, feature in enumerate(features):
          id_values.append(id)
          class_values.append(class_id)
          feature_values.append(i)

    id_values, class_values, feature_values = [], [], []
    multi_idx = pd.MultiIndex.from_tuples(
      list(zip(id_values, class_values, feature_values)),
      names=["id", "class", "feature"]
    )
    self.df = pd.DataFrame(index=multi_idx)
  
  def print_stats(self):
    [print(df) for df in self.df_raters.values()]
    print(self.df)

class ShapeComparison:
  def __init__(self, d, df1, df2, fmts=("off", "stl"), device="cpu"):
    self.d = d
    self.df1 = df1
    self.df2 = df2
    self.df_fmts = fmts
    self.df = pd.DataFrame(self.df1["id"])
    self.df.reset_index(drop=True, inplace=True)
    self.device = device

  def compute_hausdorff_distance(self):
    # directed hausdorff distance
    hausdorff_distances, max_indices, formats = [], [], []
    for (idx1, row1), (idx2, row2) in zip(self.df1.iterrows(), self.df2.iterrows()):
      v1, _ = Meshes.vf(row1["mesh"])
      v2, _ = Meshes.vf(row2["mesh"])
      dist1, idx1, _ = directed_hausdorff(v1, v2)
      dist2, idx2, _ = directed_hausdorff(v2, v1)
      if dist1 > dist2:
        max_dist = dist1
        max_idx = idx1
        fmt = self.df_fmts[0]
      else:
        max_dist = dist2
        max_idx = idx2
        fmt = self.df_fmts[0]
      hausdorff_distances.append(max_dist)
      max_indices.append(max_idx)
      formats.append(fmt)
    self.df["hausdorff_distance"] = hausdorff_distances
    self.df["hausdorff_max_index"] = max_indices
    self.df["hausdorff_fmt"] = formats
  
  def compute_optimal_transport_distance(self):
    optimal_transport_distances = []
    for (_, row1), (_, row2) in zip(self.df1.iterrows(), self.df2.iterrows()):
      n = min(row1["n_vertices"], row2["n_vertices"])
      v1 = np.asarray(row1["mesh"].sample_points_poisson_disk(n).points, dtype=np.float32)
      v2 = np.asarray(row2["mesh"].sample_points_poisson_disk(n).points, dtype=int)
      dist = geometry.optimal_transport_distance(v1, v2)
      optimal_transport_distances.append(dist)
    self.df["optimal_transport_distance"] = optimal_transport_distances
  
  def compute_shape_distribution_distance(self):
    shape_distribution_distances = []
    for (_, row1), (_, row2) in zip(self.df1.iterrows(), self.df2.iterrows()):
      v1, _ = Meshes.vf(row1["mesh"])
      v2, _ = Meshes.vf(row2["mesh"])
      dist = geometry.shape_distribution_distance(v1, v2)
      shape_distribution_distances.append(dist)
    self.df["shape_distribution_distance"] = shape_distribution_distances
  
  def compute_chamfer_distance(self):
    chamfer_distances = []
    for (_, row1), (_, row2) in zip(self.df1.iterrows(), self.df2.iterrows()):
      v1, _ = Meshes.vf(row1["mesh"])
      v2, _ = Meshes.vf(row2["mesh"])
      dist = geometry.chamfer_distance(v1, v2)
      chamfer_distances.append(dist)
    self.df["chamfer_distance"] = chamfer_distances

  def compute_heat_kernel_signature(self, n_eig=16):
    def compute_operators_row(row):
      verts = torch.tensor(np.asarray(row["mesh"].vertices, dtype=np.float32))
      faces = torch.tensor(np.asarray(row["mesh"].triangles, dtype=int))
      verts = geometry.normalize_positions(verts)
      _, _, _, evals, evecs, _, _ = geometry.get_operators(verts, faces, op_cache_dir="cache/")
      evals, evecs = evals.to(self.device), evecs.to(self.device)
      return evals, evecs

    hks, hks_normalized = [], []
    for (_, row1), (_, row2) in zip(self.df1.iterrows(), self.df2.iterrows()):
      print(row1["id"])
      evals1, evecs1 = compute_operators_row(row1)
      hks1 = geometry.compute_hks_autoscale(evals1, evecs1, n_eig)
      evals2, evecs2 = compute_operators_row(row2)
      hks2 = geometry.compute_hks_autoscale(evals2, evecs2, n_eig)
      hks_cmp, hks_cmp_normalized = geometry.compare_hks(hks1, hks2, n_eig)
      hks.append(hks_cmp)
      hks_normalized.append(hks_cmp_normalized)
      print(hks_cmp)
      print(hks_cmp_normalized)
    self.df["heat_kernel_signature"] = hks
    self.df["heat_kernel_signature_normalized"] = hks_cmp_normalized
  
  def save_stats(self):
    p = self.d / Path("dataframes/shape_comparison")
    if not p.exists():
      p.mkdir()
    self.df.to_csv(p / "geometric_similarity.csv", index=False)

  def print_stats(self): print(self.df)

def helper(df):
  for _, row in df.iterrows():
    fname = row["id"]
    max = row["hausdorff_max_index"]
    off = o3d.io.read_triangle_mesh(str(DATA_DIR / "train" / "off" / (fname+".off"))).compute_vertex_normals()
    stl = o3d.io.read_triangle_mesh(str(DATA_DIR / "train" / "stl" / (fname+".stl"))).compute_vertex_normals()
    off_ls = o3d.cuda.pybind.geometry.LineSet().create_from_triangle_mesh(off)
    stl_ls = o3d.cuda.pybind.geometry.LineSet().create_from_triangle_mesh(stl.translate((30, 0, 0)))
    pcd = o3d.cuda.pybind.geometry.PointCloud()
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(stl.vertices)[max, np.newaxis])
    o3d.visualization.draw_geometries([off_ls, stl_ls, pcd])
    return

if __name__ == "__main__":
  if True:
    # labels analysis
    lbls = Labels(d=DATA_DIR, n_labels=5)
    lbls.load_raters()
    lbls.load()
    lbls.majority_vote()
    lbls.compute_stats()
    lbls.compute_individual_statistics()
    lbls.compare_rater2gt()
    lbls.compare_mv2avg()
    lbls.compute_interrater_reliability()
    lbls.chance_corrected_agreement_coefficients()
    # save / print
    lbls.save_stats()
    lbls.print_stats()

  if True:
    # meshes analysis
    off_meshes = Meshes(d=DATA_DIR)
    off_meshes.load("off")
    off_meshes.compute_stats()
    off_meshes.print_stats()
    off_meshes.save_stats()
  if True:
    stl_meshes = Meshes(d=DATA_DIR)
    stl_meshes.load("stl")
    stl_meshes.compute_stats()
    stl_meshes.print_stats()
    stl_meshes.save_stats()

  if True:
    feature_compare = FeatureComparison(DATA_DIR, off_meshes.mesh_df, lbls.rater_dfs, n_classes=5)
    feature_compare.extract_features("off")
    feature_compare.convert()
    feature_compare.print_stats()

  if True:
    print("\nShape Comparison:")
    sc = ShapeComparison(DATA_DIR, off_meshes.mesh_df, stl_meshes.mesh_df)
    print("Hausdorff Distance")
    sc.compute_hausdorff_distance()
    #print("Optimal Transport Distance")
    #sc.compute_optimal_transport_distance()
    #print("Chamfer Distance")
    #sc.compute_chamfer_distance()
    #print("Shape Distribution Distance")
    #sc.compute_shape_distribution_distance()
    print("HKS")
    sc.compute_heat_kernel_signature()
    sc.print_stats()
    sc.save_stats()
    # helper
    # helper(sc.df)