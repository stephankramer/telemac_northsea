"""
Diagnostic: rerun the Thetis North Sea example (spin-up + run) with the open
boundary condition {'elev': tide} instead of {'elev': tide, 'uv': 0}.

With 'uv': 0 the external state of the Riemann solver has zero velocity, so
the boundary elevation becomes approximately eta_tide + sqrt(H/g)*u_n rather
than eta_tide (a Flather-type condition with zero reference velocity).

Outputs go to thetis_elev_only/outputs_spinup and thetis_elev_only/outputs_dt<dt>*.

Options:
  --run-only      skip the spin-up and restart from the original example
                  spin-up (examples/north_sea/outputs_spinup, run with 'uv': 0)
  --dt=<seconds>  time step for the run (default 3600)
  --orig-bc       keep the original {'elev', 'uv': 0} boundary condition (to
                  separate time step effects from boundary condition effects)
"""
from thetis import *
import os
import sys
import time as time_mod

pwd = os.path.abspath(os.path.dirname(__file__))
example_dir = os.path.join(pwd, os.pardir, "examples", "north_sea")
out_dir = os.path.join(pwd, "thetis_elev_only")
sys.path.insert(0, example_dir)
os.chdir(example_dir)  # model_config expects ./data and the bathymetry file here
from model_config import construct_solver, sim_tz  # noqa: E402


orig_bc = "--orig-bc" in sys.argv


def elev_only(solver_obj):
    bnd = solver_obj.bnd_functions["shallow_water"][100]
    if not orig_bc:
        bnd.pop("uv")
    print_output(f"Open boundary conditions: {list(bnd)}")


run_only = "--run-only" in sys.argv
dt = 3600.0
for arg in sys.argv[1:]:
    if arg.startswith("--dt="):
        dt = float(arg[5:])
run_name = f"outputs_dt{dt:g}" + ("_from_orig_spinup" if run_only else "") + ("_orig_bc" if orig_bc else "")
spinup_dir = os.path.join(example_dir, "outputs_spinup") if run_only else os.path.join(out_dir, "outputs_spinup")

mesh2d = read_mesh_from_checkpoint("north_sea_bathymetry.h5")
if not run_only:
    solver_obj, start_time, update_forcings = construct_solver(
        mesh2d,
        output_directory=os.path.join(out_dir, "outputs_spinup"),
        spinup=True,
        start_date=datetime.datetime(2022, 1, 1, tzinfo=sim_tz),
        end_date=datetime.datetime(2022, 1, 15, tzinfo=sim_tz),
        fields_to_export=[],
        fields_to_export_hdf5=["elev_2d", "uv_2d"],
        simulation_export_time=24 * 3600.0,
    )
    elev_only(solver_obj)
    solver_obj.assign_initial_conditions()
    update_forcings(0.0)
    tic = time_mod.perf_counter()
    solver_obj.iterate(update_forcings=update_forcings)
    print_output(f"Spin-up duration: {time_mod.perf_counter()-tic:.2f} seconds")

solver_obj, start_time, update_forcings = construct_solver(
    mesh2d,
    output_directory=os.path.join(out_dir, run_name),
    timestep=dt,
    start_date=datetime.datetime(2022, 1, 15, tzinfo=sim_tz),
    end_date=datetime.datetime(2022, 1, 18, tzinfo=sim_tz),
)
elev_only(solver_obj)
solver_obj.load_state(
    14, outputdir=spinup_dir, t=0, iteration=0, i_export=0
)
update_forcings(0.0)
tic = time_mod.perf_counter()
solver_obj.iterate(update_forcings=update_forcings)
print_output(f"Run duration: {time_mod.perf_counter()-tic:.2f} seconds")
