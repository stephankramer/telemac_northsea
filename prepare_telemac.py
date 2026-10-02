"""
Generate TELEMAC-2D input files that replicate the Thetis North Sea example
(examples/north_sea).

Run with a working Firedrake/Thetis environment from this directory:

    python prepare_telemac.py

Produces:
  geo_north_sea.slf    geometry (SERAFIND): mesh, BOTTOM and CORIOLIS fields
  geo_north_sea.cli    boundary conditions file (open boundary 100 -> 5 4 4,
                       coast 200 -> 2 2 2)
  liq_spinup.txt       node-by-node tidal elevation 2022-01-01 -> 2022-01-15
                       (incl. the linear spin-up ramp)
  liq_run.txt          node-by-node tidal elevation 2022-01-15 -> 2022-01-18
  stations_spinup.txt  time series coordinates files for the tide gauges
  stations_run.txt
"""
from thetis import *
import thetis.forcing as forcing
import numpy as np
import struct
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
example_dir = os.path.join(pwd, os.pardir, "examples", "north_sea")
sys.path.insert(0, example_dir)
from model_config import coord_system, read_station_data, sim_tz  # noqa: E402

# Same settings as examples/north_sea/model_config.py, spinup.py and run.py
omega = 7.292e-05
forcing_constituents = ["Q1", "O1", "P1", "K1", "N2", "M2", "S2", "K2"]
spinup_start = datetime.datetime(2022, 1, 1, tzinfo=sim_tz)
run_start = datetime.datetime(2022, 1, 15, tzinfo=sim_tz)
run_end = datetime.datetime(2022, 1, 18, tzinfo=sim_tz)
spinup_export = 24 * 3600.0
run_export = 3600.0
# sampling interval of the boundary forcing time series (TELEMAC interpolates
# linearly in time between samples)
forcing_dt = 300.0

data_dir = os.path.join(os.environ.get("DATA", os.path.join(example_dir, "data")), "tpxo")

# ---------------------------------------------------------------------------
# mesh and bathymetry
# ---------------------------------------------------------------------------
h5_file = os.path.join(example_dir, "north_sea_bathymetry.h5")
mesh2d = read_mesh_from_checkpoint(h5_file)
with CheckpointFile(h5_file, "r") as f:
    bathymetry_2d = f.load_function(mesh2d, "Bathymetry")
P1 = bathymetry_2d.function_space()
assert P1.ufl_element().degree() == 1

xy = Function(VectorFunctionSpace(mesh2d, "CG", 1)).interpolate(SpatialCoordinate(mesh2d))
x, y = xy.dat.data_ro[:, 0].copy(), xy.dat.data_ro[:, 1].copy()
npoin = len(x)
ikle = P1.cell_node_map().values.copy()
# TELEMAC requires counter-clockwise elements
xe, ye = x[ikle], y[ikle]
area2 = (xe[:, 1] - xe[:, 0]) * (ye[:, 2] - ye[:, 0]) - (xe[:, 2] - xe[:, 0]) * (ye[:, 1] - ye[:, 0])
cw = area2 < 0
ikle[cw] = ikle[cw][:, [0, 2, 1]]
print(f"{npoin} nodes, {len(ikle)} elements ({cw.sum()} reoriented)")

bathy = bathymetry_2d.dat.data_ro.copy()

# Coriolis parameter, exactly as in model_config.py
lon, lat = coord_system.to_lonlat(x, y)
fcor = 2 * omega * np.sin(np.radians(lat))

# ---------------------------------------------------------------------------
# boundary: exterior edges with their markers
# ---------------------------------------------------------------------------
ef = mesh2d.exterior_facets
cell_nodes = P1.exterior_facet_node_map().values
local_facet = ef.local_facet_dat.data_ro.ravel()
markers = np.zeros(len(local_facet), dtype=int)
for mark in ef.unique_markers:
    markers[ef.subset(int(mark)).indices] = mark
assert np.all(markers > 0)
edges = []
for nodes, lf, mark in zip(cell_nodes, local_facet, markers):
    # facet i of a triangle is opposite vertex i
    e = [nodes[j] for j in range(3) if j != lf]
    edges.append((e[0], e[1], mark))

# orient each boundary edge such that the domain is on its left
ccw_ikle = {}
for tri in ikle:
    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
        ccw_ikle[(a, b)] = True
next_node = {}
edge_marker = {}
for a, b, mark in edges:
    if (a, b) not in ccw_ikle:
        a, b = b, a
    assert (a, b) in ccw_ikle
    assert a not in next_node, "boundary node with two outgoing edges"
    next_node[a] = b
    edge_marker[a] = mark  # marker of edge (a, next_node[a])

# split into closed loops; outer loop is the one with largest (positive) area
loops = []
remaining = set(next_node)
while remaining:
    start = min(remaining)
    loop = [start]
    n = next_node[start]
    while n != start:
        loop.append(n)
        n = next_node[n]
    remaining -= set(loop)
    loops.append(loop)


def signed_area(loop):
    lx, ly = x[loop], y[loop]
    return 0.5 * np.sum(lx * np.roll(ly, -1) - np.roll(lx, -1) * ly)


loops.sort(key=signed_area, reverse=True)
assert signed_area(loops[0]) > 0
assert all(signed_area(lp) < 0 for lp in loops[1:])
# start outer boundary at the south-west most node (TELEMAC convention)
outer = loops[0]
i0 = int(np.argmin(x[outer] + y[outer]))
loops[0] = outer[i0:] + outer[:i0]
nbor = np.array([n for lp in loops for n in lp])
nptfr = len(nbor)
print(f"{nptfr} boundary nodes in {len(loops)} loops")

prev_node = {b: a for a, b in next_node.items()}
# a node is liquid only if both adjacent boundary edges are open (100);
# nodes at the junction of open boundary and coast are treated as solid
liquid = np.array([edge_marker[n] == 100 and edge_marker[prev_node[n]] == 100 for n in nbor])
print(f"{liquid.sum()} liquid boundary nodes")

ipobo = np.zeros(npoin, dtype=np.int32)
ipobo[nbor] = np.arange(1, nptfr + 1)


# ---------------------------------------------------------------------------
# write SERAFIND geometry and the boundary conditions file
# ---------------------------------------------------------------------------
def write_serafin(fname, title, varnames, values, date=None, double=True):
    """Minimal (big endian) SERAFIN(D) writer for a single time record."""
    ft, fs = ("d", 8) if double else ("f", 4)
    fmt = b"SERAFIND" if double else b"SERAFIN "

    def rec(fmtstr, *vals):
        payload = struct.pack(">" + fmtstr, *vals)
        return struct.pack(">i", len(payload)) + payload + struct.pack(">i", len(payload))

    with open(fname, "wb") as f:
        f.write(rec("80s", title.ljust(72)[:72].encode() + fmt))
        f.write(rec("2i", len(varnames), 0))
        for name, unit in varnames:
            f.write(rec("32s", (name.ljust(16) + unit.ljust(16)).encode()))
        iparam = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if date is not None:
            iparam[9] = 1
        f.write(rec("10i", *iparam))
        if date is not None:
            f.write(rec("6i", *date))
        f.write(rec("4i", len(ikle), npoin, 3, 1))
        f.write(rec(f"{ikle.size}i", *(ikle.ravel() + 1)))
        f.write(rec(f"{npoin}i", *ipobo))
        f.write(rec(f"{npoin}{ft}", *x))
        f.write(rec(f"{npoin}{ft}", *y))
        f.write(rec(ft, 0.0))
        for v in values:
            f.write(rec(f"{npoin}{ft}", *v))


write_serafin(
    os.path.join(pwd, "geo_north_sea.slf"), "NORTH SEA (THETIS MESH)",
    [("BOTTOM", "M"), ("CORIOLIS", "1/S")], [-bathy, fcor],
    date=(2022, 1, 1, 0, 0, 0),
)

with open(os.path.join(pwd, "geo_north_sea.cli"), "w") as f:
    for k, (n, liq) in enumerate(zip(nbor, liquid)):
        codes = "5 4 4" if liq else "2 2 2"
        litbor = 5 if liq else 2
        f.write(f"{codes} 0.000 0.000 0.000 0.000 {litbor} 0.000 0.000 0.000 {n+1} {k+1}\n")

# ---------------------------------------------------------------------------
# tidal boundary forcing, evaluated with the same Thetis/uptide code
# ---------------------------------------------------------------------------
liq_nodes = nbor[liquid]


def write_liquid_file(fname, start_date, t_end, ramp_t=None):
    elev_tide_2d = Function(P1, name="Tidal elevation")
    tbnd = forcing.TPXOTidalBoundaryForcing(
        elev_tide_2d, start_date, coord_system, data_dir=data_dir,
        constituents=forcing_constituents, boundary_ids=[100],
    )
    times = np.arange(0.0, t_end + 0.5 * forcing_dt, forcing_dt)
    with open(fname, "w") as f:
        f.write(f"# Tidal elevation at open boundary nodes, t=0 is {start_date.isoformat()}\n")
        f.write("# generated by prepare_telemac.py from TPXO9v5a (uptide) with constituents "
                + " ".join(forcing_constituents) + "\n")
        if ramp_t is not None:
            f.write(f"# includes linear ramp over {ramp_t} s\n")
        f.write("T " + " ".join(f"SL({n+1})" for n in liq_nodes) + "\n")
        f.write("s " + " ".join("m" for n in liq_nodes) + "\n")
        for t in times:
            tbnd.set_tidal_field(t)
            vals = elev_tide_2d.dat.data_ro[liq_nodes]
            if ramp_t is not None:
                vals = vals * min(t / ramp_t, 1.0)
            f.write(f"{t:.1f} " + " ".join(f"{v:.6f}" for v in vals) + "\n")
    print(f"Written {fname}: {len(times)} records")


spinup_t_end = (run_start - spinup_start).total_seconds()
run_t_end = (run_end - run_start).total_seconds()
write_liquid_file(os.path.join(pwd, "liq_spinup.txt"), spinup_start, spinup_t_end, ramp_t=spinup_t_end)
write_liquid_file(os.path.join(pwd, "liq_run.txt"), run_start, run_t_end)

# ---------------------------------------------------------------------------
# tide gauge locations
# ---------------------------------------------------------------------------
stations = read_station_data()
for fname, t_end, t_export in (("stations_spinup.txt", spinup_t_end, spinup_export),
                               ("stations_run.txt", run_t_end, run_export)):
    with open(os.path.join(pwd, fname), "w") as f:
        f.write(f"1 {len(stations)}\n")
        f.write(f"0. {t_end:.1f} {t_export:.1f}\n")
        for i, (name, data) in enumerate(stations.items()):
            sta_lat, sta_lon = data["latlon"]
            sta_x, sta_y = coord_system.to_xy(sta_lon, sta_lat)
            f.write(f"{sta_x:.4f} {sta_y:.4f} {i+1} {name}\n")
