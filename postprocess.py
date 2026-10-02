"""
Convert TELEMAC-2D tide gauge time series (TIME SERIES FILE 1) into the
HDF5 format written by Thetis' TimeSeriesCallback2D, i.e.
    <outdir>/diagnostic_timeseries_<station>_elev.hdf5
and optionally plot them against the Thetis results and the observations.

Usage:
    python postprocess.py                    # ts_run.slf -> outputs/
    python postprocess.py ts_spinup.slf outputs_spinup
    python postprocess.py --plot             # also compare with thetis
    python postprocess.py --plot --rmse      # ... with RMS errors vs observations
"""
import argparse
import datetime
import os
import struct

import h5py
import numpy as np

pwd = os.path.abspath(os.path.dirname(__file__))
example_dir = os.path.join(pwd, os.pardir, "examples", "north_sea")


def read_serafin(fname):
    """Minimal SERAFIN/SERAFIND reader (big endian)."""
    with open(fname, "rb") as f:
        data = f.read()
    pos = 0

    def rec():
        nonlocal pos
        n, = struct.unpack(">i", data[pos:pos + 4])
        payload = data[pos + 4:pos + 4 + n]
        pos += n + 8
        return payload

    title = rec().decode()
    double = title[72:80] == "SERAFIND"
    ft, fs = ("d", 8) if double else ("f", 4)
    nbv1, nbv2 = struct.unpack(">2i", rec())
    names = [rec().decode()[:16].strip() for _ in range(nbv1 + nbv2)]
    iparam = struct.unpack(">10i", rec())
    date = struct.unpack(">6i", rec()) if iparam[9] == 1 else None
    nelem, npoin, ndp, _ = struct.unpack(">4i", rec())
    rec()  # ikle
    rec()  # ipobo
    x = np.frombuffer(rec(), dtype=">" + ft)
    y = np.frombuffer(rec(), dtype=">" + ft)
    times, values = [], []
    while pos < len(data):
        times.append(struct.unpack(">" + ft, rec())[0])
        values.append([np.frombuffer(rec(), dtype=">" + ft) for _ in names])
    return dict(names=names, date=date, x=x, y=y,
                times=np.array(times), values=np.array(values))


def read_stations(coord_file):
    stations = []
    with open(coord_file) as f:
        nt, npts = map(int, f.readline().split())
        for _ in range(nt):
            f.readline()
        for _ in range(npts):
            sx, sy, _, name = f.readline().split()[:4]
            stations.append((name, float(sx), float(sy)))
    return stations


def convert(ts_file, coord_file, outdir):
    ts = read_serafin(ts_file)
    ivar = ts["names"].index("FREE SURFACE")
    start = datetime.datetime(*ts["date"], tzinfo=datetime.timezone.utc)
    units = "seconds since " + start.isoformat()
    os.makedirs(outdir, exist_ok=True)
    for name, sx, sy in read_stations(coord_file):
        # match station by location
        i = int(np.argmin(np.hypot(ts["x"] - sx, ts["y"] - sy)))
        assert np.hypot(ts["x"][i] - sx, ts["y"][i] - sy) < 1.0, name
        fname = os.path.join(outdir, f"diagnostic_timeseries_{name}_elev.hdf5")
        with h5py.File(fname, "w") as h:
            h.attrs["location_name"] = name
            h.attrs["simulation_initial_date"] = start.isoformat()
            h.attrs["x"] = sx
            h.attrs["y"] = sy
            h.create_dataset("time", data=ts["times"][:, None]).attrs["units"] = units
            h.create_dataset("elev", data=ts["values"][:, ivar, i][:, None])
        print(f"Written {fname}")


def rms_error(t_model, v_model, t_obs, v_obs):
    """
    RMS error of a model time series against observations, evaluated at the
    observation times (model interpolated linearly), after removing the mean
    of each series over the common time window.
    """
    ts = np.array([t.timestamp() for t in t_model])
    to = np.array([t.timestamp() for t in t_obs])
    obs = np.ma.filled(np.ma.asarray(v_obs, dtype=float), np.nan)
    ix = (to >= ts[0]) & (to <= ts[-1]) & np.isfinite(obs)
    model = np.interp(to[ix], ts, v_model)
    obs = obs[ix]
    return np.sqrt(np.mean(((model - model.mean()) - (obs - obs.mean()))**2))


def plot(telemac_dir, thetis_dir, imgfile, rmse=False):
    import cftime
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import netCDF4
    import csv

    def load(f):
        with h5py.File(f, "r") as h:
            t = cftime.num2pydate(h["time"][:].ravel(), h["time"].attrs["units"])
            return t, h["elev"][:].ravel()

    errors = {}
    fig = plt.figure(figsize=(30, 10))
    axes = fig.subplots(ncols=4, nrows=2)
    with open(os.path.join(example_dir, "stations_elev.csv")) as f:
        stations = list(csv.DictReader(f, skipinitialspace=True))
    for i, data in enumerate(stations):
        sta, region = data["name"], data["region"]
        ax = axes[i // 4, i % 4]
        ax.set_title(sta)
        suffix = "TG" if region == "NO" else ""
        o = os.path.join(example_dir, "observations", f"{region}_TS_TG_{sta}{suffix}_202201.nc")
        time_tel, vals_tel = load(os.path.join(telemac_dir, f"diagnostic_timeseries_{sta}_elev.hdf5"))
        series = [("TELEMAC", time_tel, vals_tel, "b--")]
        f = os.path.join(thetis_dir, f"diagnostic_timeseries_{sta}_elev.hdf5")
        if os.path.exists(f):
            series.insert(0, ("Thetis", *load(f), "r:"))
        if os.path.exists(o):
            with netCDF4.Dataset(o, "r") as nc:
                t_obs = cftime.num2pydate(nc.variables["TIME"][:], nc.variables["TIME"].units)
                v_obs = nc.variables["SLEV"][:]
                v_obs = v_obs.reshape(len(t_obs), -1)[:, 0]
            ix = (t_obs >= time_tel[0]) * (t_obs <= time_tel[-1])
            if rmse:
                for j, (label, t, v, style) in enumerate(series):
                    err = rms_error(t, v, t_obs, v_obs)
                    errors.setdefault(label, {})[sta] = err
                    series[j] = (f"{label} (RMSE {err:.3f} m)", t, v, style)
            series.insert(0, ("Observation", t_obs[ix], v_obs[ix], "k-"))
        for label, t, v, style in series:
            ax.plot(t, v - np.mean(v), style, label=label, lw=1.5)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.set_xlim([time_tel[0], time_tel[-1]])
        if i % 4 == 0:
            ax.set_ylabel("Elevation (m)")
        ax.grid(True)
        ax.legend(ncol=3 if not rmse else 1)
    if errors:
        models = list(errors)
        print("RMS error vs observations (m), means removed:")
        print(f"{'station':12s}" + "".join(f"{m:>10s}" for m in models))
        for sta in errors[models[0]]:
            print(f"{sta:12s}" + "".join(f"{errors[m].get(sta, np.nan):10.3f}" for m in models))
        print(f"{'mean':12s}" + "".join(f"{np.mean(list(errors[m].values())):10.3f}" for m in models))
    print(f"Saving to {imgfile}")
    plt.savefig(imgfile, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ts_file", nargs="?", default="ts_run.slf")
    parser.add_argument("outdir", nargs="?", default="outputs")
    parser.add_argument("--coords", default=None,
                        help="time series coordinates file (default: derived from ts_file)")
    parser.add_argument("--plot", action="store_true",
                        help="plot against thetis output and observations")
    parser.add_argument("--rmse", action="store_true",
                        help="compute RMS errors against observations, show them in the "
                        "plot legends and print a table (implies --plot)")
    parser.add_argument("--thetis-dir", default=os.path.join(example_dir, "outputs"))
    args = parser.parse_args()
    coords = args.coords or args.ts_file.replace("ts_", "stations_").replace(".slf", ".txt")
    convert(args.ts_file, coords, args.outdir)
    if args.plot or args.rmse:
        plot(args.outdir, args.thetis_dir, "north_sea_elev_ts_telemac.png", rmse=args.rmse)
