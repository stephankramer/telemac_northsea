"""
Compare Thetis and TELEMAC-2D solutions on the (discontinuous) Thetis output
mesh.

For each output time this reads the Thetis Elevation2d/Velocity2d vtu files
(discontinuous: points are duplicated per element) and the corresponding
TELEMAC legacy VTK file (continuous P1, from `converter.py srf2vtk`),
"interpolates" the TELEMAC solution onto the discontinuous points and writes a
vtu per time step (plus a .pvd collection) containing:

    elev_thetis, uv_thetis        Thetis solution
    elev_telemac, uv_telemac      TELEMAC solution (velocity as a vector)
    elev_diff, uv_diff            thetis - telemac

When both models use the same mesh (as here) each discontinuous point
coincides with a TELEMAC node and the value is simply copied from that node.
Otherwise the TELEMAC P1 field is evaluated at the points with vtkProbeFilter.

Usage (in an environment with the vtk python package, e.g. ~/fdr):

    python compare_vtu.py
    python compare_vtu.py --thetis-dir ../examples/north_sea/outputs \\
        --telemac-vtk 'vtk_run/r2d_run*.vtk' --output-dir compare_run
"""
import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

pwd = os.path.abspath(os.path.dirname(__file__))


def read_pvd(pvd_file):
    """Return list of (timestep, vtu file) from a pvd collection."""
    root = ET.parse(pvd_file).getroot()
    base = os.path.dirname(pvd_file)
    return [(float(ds.get("timestep")), os.path.join(base, ds.get("file")))
            for ds in root.iter("DataSet")]


def read_vtu(fname):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader.GetOutput()


def read_legacy_vtk(fname):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(fname)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    return reader.GetOutput()


def point_array(grid, name=None, prefix=None):
    pd = grid.GetPointData()
    if prefix is not None:
        names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        matches = [n for n in names if n.startswith(prefix)]
        if len(matches) != 1:
            raise KeyError(f"no unique array starting with {prefix!r} in {names}")
        name = matches[0]
    if name is None:
        name = pd.GetArrayName(0)
    return vtk_to_numpy(pd.GetArray(name))


def telemac_fields(grid):
    """Elevation and velocity (as a 3-vector) from a TELEMAC vtk file."""
    elev = point_array(grid, prefix="FREE_SURFACE")
    u = point_array(grid, prefix="VELOCITY_U")
    v = point_array(grid, prefix="VELOCITY_V")
    return elev, np.column_stack([u, v, np.zeros_like(u)])


class Interpolator:
    """Maps TELEMAC P1 nodal values onto the points of the Thetis mesh."""

    def __init__(self, thetis_grid, telemac_grid, tol):
        self.points = vtk_to_numpy(thetis_grid.GetPoints().GetData())[:, :2]
        tel_points = vtk_to_numpy(telemac_grid.GetPoints().GetData())[:, :2]
        dist, self.index = cKDTree(tel_points).query(self.points)
        self.matching = dist.max() <= tol
        if self.matching:
            print(f"Meshes coincide (max distance {dist.max():.3g} m): copying nodal values")
        else:
            print(f"Meshes differ (max distance {dist.max():.3g} m): using vtkProbeFilter")
            self.probe = vtk.vtkProbeFilter()
            self.probe.SetInputData(thetis_grid)

    def __call__(self, telemac_grid):
        elev, uv = telemac_fields(telemac_grid)
        if self.matching:
            return elev[self.index], uv[self.index]
        # general case: evaluate the P1 field at the thetis points; points
        # that fall just outside the telemac mesh take the nearest nodal value
        src = vtk.vtkUnstructuredGrid()
        src.ShallowCopy(telemac_grid)
        src.GetPointData().Initialize()
        src.GetPointData().AddArray(named_array(elev, "elev"))
        src.GetPointData().AddArray(named_array(uv, "uv"))
        self.probe.SetSourceData(src)
        self.probe.Update()
        out = self.probe.GetOutput().GetPointData()
        valid = vtk_to_numpy(out.GetArray(self.probe.GetValidPointMaskArrayName())).astype(bool)
        elev_i = vtk_to_numpy(out.GetArray("elev")).copy()
        uv_i = vtk_to_numpy(out.GetArray("uv")).copy()
        elev_i[~valid] = elev[self.index[~valid]]
        uv_i[~valid] = uv[self.index[~valid]]
        return elev_i, uv_i


def named_array(values, name):
    arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    arr.SetName(name)
    return arr


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thetis-dir", default=os.path.join(pwd, os.pardir, "examples", "north_sea", "outputs"),
                        help="thetis output directory containing Elevation2d/ and Velocity2d/")
    parser.add_argument("--telemac-vtk", default=os.path.join(pwd, "vtk_run", "r2d_run*.vtk"),
                        help="glob pattern for the TELEMAC vtk series (sorted by record number)")
    parser.add_argument("--output-dir", default=os.path.join(pwd, "compare_run"))
    parser.add_argument("--telemac-offset", type=int, default=0,
                        help="TELEMAC record index corresponding to thetis export 0")
    parser.add_argument("--telemac-stride", type=int, default=1,
                        help="number of TELEMAC records per thetis export")
    parser.add_argument("--tol", type=float, default=1.0,
                        help="distance (m) below which mesh points are considered identical")
    args = parser.parse_args()

    elev_series = read_pvd(os.path.join(args.thetis_dir, "Elevation2d", "Elevation2d.pvd"))
    uv_series = read_pvd(os.path.join(args.thetis_dir, "Velocity2d", "Velocity2d.pvd"))
    assert len(elev_series) == len(uv_series), "elevation and velocity series differ in length"
    tel_files = sorted(glob.glob(args.telemac_vtk), key=natural_key)
    if not tel_files:
        raise IOError(f"no files matching {args.telemac_vtk}")

    n = len(elev_series)
    last_needed = args.telemac_offset + (n - 1) * args.telemac_stride
    if last_needed >= len(tel_files):
        n = (len(tel_files) - 1 - args.telemac_offset) // args.telemac_stride + 1
        print(f"WARNING: only {len(tel_files)} TELEMAC files; comparing the first {n} thetis exports")
    elif last_needed < len(tel_files) - 1:
        print(f"WARNING: {len(tel_files)} TELEMAC files, only using records "
              f"{args.telemac_offset}..{last_needed} (stride {args.telemac_stride})")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    interp = None
    pvd_entries = []
    for i in range(n):
        t, elev_file = elev_series[i]
        t_uv, uv_file = uv_series[i]
        assert t == t_uv
        tel_file = tel_files[args.telemac_offset + i * args.telemac_stride]

        elev_grid = read_vtu(elev_file)
        elev_th = point_array(elev_grid)
        uv_th = point_array(read_vtu(uv_file))
        if uv_th.shape[1] == 2:
            uv_th = np.column_stack([uv_th, np.zeros(len(uv_th))])
        tel_grid = read_legacy_vtk(tel_file)
        if interp is None:
            interp = Interpolator(elev_grid, tel_grid, args.tol)
        elev_tel, uv_tel = interp(tel_grid)

        out = vtk.vtkUnstructuredGrid()
        out.CopyStructure(elev_grid)
        pd = out.GetPointData()
        for values, name in ((elev_th, "elev_thetis"), (uv_th, "uv_thetis"),
                             (elev_tel, "elev_telemac"), (uv_tel, "uv_telemac"),
                             (elev_th - elev_tel, "elev_diff"), (uv_th - uv_tel, "uv_diff")):
            pd.AddArray(named_array(values, name))
        pd.SetActiveScalars("elev_diff")
        pd.SetActiveVectors("uv_diff")

        fname = f"compare_{i}.vtu"
        writer.SetFileName(os.path.join(args.output_dir, fname))
        writer.SetInputData(out)
        writer.Write()
        pvd_entries.append((t, fname))
        print(f"{fname}: thetis {os.path.basename(elev_file)} vs telemac {os.path.basename(tel_file)}"
              f"  max|elev diff|={np.abs(elev_th - elev_tel).max():.3f} m"
              f"  max|uv diff|={np.linalg.norm(uv_th - uv_tel, axis=1).max():.3f} m/s")

    with open(os.path.join(args.output_dir, "compare.pvd"), "w") as f:
        f.write('<?xml version="1.0" ?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for t, fname in pvd_entries:
            f.write(f'<DataSet timestep="{t:g}" file="{fname}" />\n')
        f.write("</Collection>\n</VTKFile>\n")
    print(f"Written {os.path.join(args.output_dir, 'compare.pvd')}")


if __name__ == "__main__":
    main()
