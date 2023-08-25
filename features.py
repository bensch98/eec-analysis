import copy
import json
import os
from pathlib import Path
import subprocess
from tqdm import tqdm

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

class ComponentFeature:
  colors = [[4, 30, 60], [106, 138, 34], [220, 30, 38], [255, 203, 0], [245, 130, 31]]
  colors = [[j / 255 for j in i] for i in colors]

  def __init__(self, mesh):
    self._mesh = mesh
    self.convex_hull = None
    # hole center
    self._hc = None
    # surface center of gravity
    self._scog = None
    # volume center of gravity
    self._vcog = None
    # cluster center
    self._cc = None
    # volume
    self._v = None
    # surface area
    self._sa = None

  @staticmethod
  def find_boundary_vertices(mesh):
    def face2edges(f):
      l = [f[0], f[1], f[2]]
      return zip(l, l[1:] + l[:1])
    faces = np.asarray(mesh.triangles)
    edges = [(t1, t2) if t1 < t2 else (t2, t1) for face in faces for t1, t2 in face2edges(face)]
    verts = pd.Series(edges).value_counts().where(lambda x: x == 1).dropna().keys().tolist()
    return list(set([i for t in verts for i in t]))

  def hole_center(self):
    verts = ComponentFeature.find_boundary_vertices(self._mesh)
    pcd = o3d.cuda.pybind.geometry.PointCloud()
    pcd.points = o3d.cuda.pybind.utility.Vector3dVector(np.asarray(self._mesh.vertices)[verts])
    self._hc = pcd.get_center()
    return self._hc

  def surface_center_of_gravity(self):
    tmp = copy.deepcopy(self._mesh)
    tmp = tmp.subdivide_loop(1)
    i = 0
    while True:
      i += 1
      verts = ComponentFeature.find_boundary_vertices(self._mesh)
      if len(verts) >= len(tmp.vertices) or i == 100:
        break
      inner = list(set(list(range(len(tmp.vertices)))) - set(verts))
      tmp = tmp.select_by_index(inner)
    self._scog = tmp.get_center()
    return self._scog

  def volume_center_of_gravity(self):
    pcd = self._mesh.sample_points_poisson_disk(10 * len(self._mesh.vertices))
    self._vcog = pcd.get_center()
    return self._vcog

  def cluster_center(self):
    self._cc = self._mesh.get_center()
    return self._cc

  def volume(self):
    if len(self._mesh.vertices) < 4:
      return 0
    self.convex_hull, _ = self._mesh.compute_convex_hull()
    self._v = self.convex_hull.get_volume()
    return self._v

  def surface_area(self):
    self._sa = self._mesh.get_surface_area()
    return self._sa

  def normal(self):
    if self._hc is None or self._scog is None:
      raise Exception(
        "Hole center and surface center of gravity must be calculated before function call.")
    v = self._hc - self._scog
    i = np.argmax(abs(v))
    bbox = self._mesh.get_axis_aligned_bounding_box()
    bbox.color = [0, 0, 0]
    x = (bbox.min_bound + bbox.max_bound) / 2
    x1 = np.copy(x)
    x1[i] = bbox.min_bound[i]
    x2 = np.copy(x)
    x2[i] = bbox.max_bound[i]
    self._n = np.stack((x1, x2))
    return self._n

  def compute_metrics(self):
    self._hc = self.hole_center()
    self._scog = self.surface_center_of_gravity()
    self._vcog = self.volume_center_of_gravity()
    self._cc = self.cluster_center()
    self._v = self.volume()
    self._sa = self.surface_area()
    self._n = self.normal()
    return self._hc, self._scog, self._vcog, self._cc, self._v, self._sa, self._n

  def visualize(self, gridview=True, centroids=True, convex_hull=False, bounding_box=False, coord_frame=False, get_vis=False):
    def attach(*args):
      for color, x in args:
        x = x[np.newaxis, ...]
        pcd = o3d.cuda.pybind.geometry.PointCloud()
        pcd.points = o3d.cuda.pybind.utility.Vector3dVector(x)
        pcd.paint_uniform_color(color)
        metrics.append(pcd)

    metrics = []
    if centroids:
      if self._hc is not None:
        attach((ComponentFeature.colors[0], self._hc))
      if self._scog is not None:
        attach((ComponentFeature.colors[1], self._scog))
      if self._vcog is not None:
        attach((ComponentFeature.colors[2], self._vcog))
      if self._cc is not None:
        attach((ComponentFeature.colors[3], self._cc))

      if self._n is not None:
        pcd = o3d.cuda.pybind.geometry.PointCloud()
        pcd.points = o3d.cuda.pybind.utility.Vector3dVector(self._n)
        pcd.paint_uniform_color(ComponentFeature.colors[4])
        metrics.append(pcd)

    if convex_hull:
      ch = o3d.cuda.pybind.geometry.LineSet().create_from_triangle_mesh(self.convex_hull)
      ch.paint_uniform_color([0] * 3)
      metrics.append(ch)

    if bounding_box:
      bbox = self._mesh.get_axis_aligned_bounding_box()
      bbox.color = [0, 0, 0]
      metrics.append(bbox)

    if gridview:
      ls = o3d.cuda.pybind.geometry.LineSet().create_from_triangle_mesh(self._mesh)
      ls.paint_uniform_color([0] * 3)
      metrics.append(ls)
    else:
      metrics.append(self._mesh)
    
    if coord_frame:
      cf = o3d.cuda.pybind.geometry.TriangleMesh().create_coordinate_frame(size=10.0, origin=np.array([0., 0., 0.]))
      metrics.append(cf)

    if get_vis: return metrics
    o3d.visualization.draw_geometries(metrics)
  
  def write(self, fname, write_mesh=True):
    data = {
      "hc": self._hc.tolist(),
      "scog": self._scog.tolist(),
      "vcog": self._vcog.tolist(),
      "cc": self._cc.tolist(),
      "v": self._v,
      "sa": self._sa,
      "n": self._n.tolist(),
    }
    if write_mesh:
      o3d.io.write_triangle_mesh(f"{fname}.off", self._mesh, write_ascii=True)
    with open(f"{fname}.json", "w", encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=4)

  @staticmethod
  def eliminate_borders(faces, y):
    lbl_update_indices = []
    for face in faces:
      lbl = -1
      for i, v in enumerate(face):
        if i == 0:
          lbl = y[v]
        elif y[v] != lbl:
          [lbl_update_indices.append(x) for x in face]
          break
    lbl_update_indices = np.unique(np.array(lbl_update_indices))
    return lbl_update_indices

class Component:
  colors = [[4, 30, 60], [106, 138, 34], [220, 30, 38], [255, 203, 0], [245, 130, 31]]
  colors = [[j / 255 for j in i] for i in colors]

  def __init__(self, datadir: str, split: str, fmt: str, mesh_id: str, n_classes: int = 5):
    self.fmesh = f"{datadir}/{split}/{fmt}/{mesh_id}.{fmt}"
    self.ftxt = f"{datadir}/{split}/txt/{mesh_id}.txt"
    self.n_classes = n_classes
    self.compfeatures = {k: [] for k in range(5)}

  def cluster_components(self):
    mesh = o3d.io.read_triangle_mesh(self.fmesh)
    lbls = np.loadtxt(self.ftxt)
    faces = np.asarray(mesh.triangles)
    lbl_update_indices = ComponentFeature.eliminate_borders(faces, lbls)
    y2 = np.copy(lbls)
    y2[lbl_update_indices] = 0

    for i in range(1, self.n_classes):
      tmpmesh = copy.deepcopy(mesh)
      class_indices = np.nonzero(y2 != i)[0]
      tmpmesh.remove_vertices_by_index(class_indices.tolist())

      segmesh = mesh.select_by_index(np.where(lbls == i)[0])
      segmesh.compute_vertex_normals()
      clidxi, nfacesi, _ = segmesh.cluster_connected_triangles()
      mask = np.asarray(clidxi)

      for j, x in enumerate(nfacesi):
        featmesh = copy.deepcopy(segmesh)
        featmesh.compute_vertex_normals()
        featmesh.remove_triangles_by_index(np.where(mask != j)[0])
        featmesh = featmesh.remove_unreferenced_vertices()
        self.compfeatures[i].append(ComponentFeature(featmesh))
    return self.compfeatures