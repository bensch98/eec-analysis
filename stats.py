from itertools import combinations
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew, ttest_rel
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from statsmodels.stats.inter_rater import fleiss_kappa

from irrCAC.raw import CAC

# global options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', shutil.get_terminal_size().columns)

DATA_DIR = Path("data")
SPLITS = ["train", "val", "test"]
NP_DESCRIPTIVE_STATS = ["mean", "median", "std", "var", "min", "max"]
SCIPY_DESCRIPTIVE_STATS = ["kurtosis", "skew"]
FORMATS = ["off", "obj", "stl"]

class Labels:
  scores = ["accuracy", "precision", "recall", "f1_score", "jaccard"]

  def __init__(self, d, n_labels):
    self.d = d
    self.n_raters = len([i for i in (d / "test_raters").iterdir() if i.is_dir()])
    self.n_labels = n_labels
    self.tr_df, self.val_df, self.te_df = self.init_df()
  
  @property
  def dfs(self):
    return [self.tr_df, self.val_df, self.te_df]

  def init_df(self):
    tr_fnames = [i.stem for i in (self.d / "train" / "txt").glob("*.txt")]
    val_fnames = [i.stem for i in (self.d / "val" / "txt").glob("*.txt")]
    te_fnames = [i.stem for i in (self.d / "test" / "txt").glob("*.txt")]
    self.tr_df = pd.DataFrame({"id": tr_fnames})
    self.val_df = pd.DataFrame({"id": val_fnames})
    self.te_df = pd.DataFrame({"id": te_fnames})
    return self.tr_df, self.val_df, self.te_df

  def _load_txt(self, f, fullpath): return np.loadtxt(fullpath, dtype=int)

  def load_raters(self):
    self.rater_dfs = {}
    for i in range(self.n_raters):
      df = self.te_df.copy()
      df["labels"] = df.apply(lambda row: self._load_txt(row["id"], self.d / f"test_raters/{i}/{row['id']}.txt"), axis=1)
      self.rater_dfs[i] = df
  
  def load(self):
    self.tr_df["labels"] = self.tr_df.apply(lambda row: self._load_txt(row["id"], self.d / f"train/txt/{row['id']}.txt"), axis=1)
    self.val_df["labels"] = self.val_df.apply(lambda row: self._load_txt(row["id"], self.d / f"val/txt/{row['id']}.txt"), axis=1)

  def _maj_vote(self, row):
    votes = np.array([df.loc[row.name, "labels"] for df in self.rater_dfs.values()])
    mode, _ = stats.mode(votes, axis=0, keepdims=True)
    return mode[0]

  def majority_vote(self):
    self.te_df["labels"] = self.te_df.apply(self._maj_vote, axis=1)
  
  def _compute_descriptive_statistics_row(self, df):
    # descriptive stats
    df["mean"] = df["labels"].apply(np.mean)
    df["median"] = df["labels"].apply(np.median)
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
      "mean": arr.mean(),
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
    df = pd.DataFrame(index=index, columns=self.scores)
    df[:] = None

    # ttests
    for score in self.scores:
      for comb in combs:
        t_stat, p_val = ttest_rel(self.rater_dfs[comb[0]][score].values, self.rater_dfs[comb[1]][score].values)
        df.loc[comb, score] = (t_stat, p_val)

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
    mesh_df = pd.DataFrame({
      "id": self.fnames["train"] + self.fnames["val"] + self.fnames["test"],
      "split": ["train"] * len(self.fnames["train"]) + ["val"] * len(self.fnames["val"]) + ["test"] * len(self.fnames["test"])
    })
    return mesh_df
  
  def _load_mesh(self, fpath):
    mesh = o3d.io.read_triangle_mesh(str(fpath))
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int32)

  def load(self, fmt):
    self.mesh_df[["vertices", "faces"]] = self.mesh_df.apply(lambda row: pd.Series(self._load_mesh(self.d / f"{row['split']}/{fmt}/{row['id']}.{fmt}")), axis=1)

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
    vertices, faces = row["vertices"], row["faces"]
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
      "surface_area": surface_area,
      "volume": volume,
    })
  
  def compute_stats(self):
    stats_df = self.mesh_df.apply(self._compute_descriptive_stats, axis=1)
    self.mesh_df = pd.concat([self.mesh_df, stats_df], axis=1)
  
  def print_stats(self):
    print(self.mesh_df)

if __name__ == "__main__":
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
  # print
  lbls.print_stats()
  del lbls

  # meshes analysis
  off_meshes = Meshes(d=DATA_DIR)
  off_meshes.load("off")
  off_meshes.compute_stats()
  off_meshes.print_stats()
  #stl_meshes = Meshes(d=DATA_DIR)
  #stl_meshes.load("stl")
  #stl_meshes.compute_stats()
  #stl_meshes.print_stats()

  # TODO:
  # shape comparison:
  # - hausdorff
  # - spectral analysis via eigenvectors/values, laplacian

  # feature comparison:
  # - crop out feature vectors for each rater
  # - compare variance etc. in their centroids