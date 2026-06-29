"""
ERA5 monthly reanalysis — West Africa cocoa belt (Ivory Coast / Ghana)
Variables : total_precipitation + 2m_temperature
Period    : 1950 to present (monthly means)
Output    : era5_wa_monthly.parquet
"""

import os
import cdsapi
import numpy as np
import xarray as xr
import pandas as pd
import calendar
from datetime import date

BASE   = os.path.dirname(os.path.abspath(__file__))
TMP_NC = os.path.join(BASE, "_era5_wa_tmp.nc")
OUT_PQ = os.path.join(BASE, "era5_wa_monthly.parquet")

# Cocoa belt: Ivory Coast + Ghana
# North=10, West=-8, South=4, East=2
AREA = [10, -8, 4, 2]

START_YEAR = 1950
END_YEAR   = date.today().year


def download():
    client = cdsapi.Client(quiet=True)
    years  = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]
    print(f"  Requesting {START_YEAR}–{END_YEAR} ({len(years)*12} months) ...")
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     ["total_precipitation", "2m_temperature"],
            "year":         years,
            "month":        months,
            "time":         "00:00",
            "area":         AREA,
            "data_format":  "netcdf",
        },
        TMP_NC,
    )
    print(f"  Downloaded -> {os.path.basename(TMP_NC)}")


def _open_nc(path):
    """Open netCDF, handling zip archives from the new CDS API (may contain one file per variable)."""
    import zipfile, tempfile, shutil
    if zipfile.is_zipfile(path):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmpdir)
        nc_files = sorted([os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".nc")])
        if not nc_files:
            raise FileNotFoundError("No .nc found inside zip archive")
        print(f"  Found {len(nc_files)} file(s) in archive: {[os.path.basename(f) for f in nc_files]}")
        if len(nc_files) == 1:
            ds = xr.load_dataset(nc_files[0])
        else:
            # Merge all files (one per variable) into one dataset
            datasets = [xr.load_dataset(f) for f in nc_files]
            ds = xr.merge(datasets)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return ds, None
    return xr.open_dataset(path), None


def process():
    ds, tmpdir = _open_nc(TMP_NC)

    # Locate time coordinate (varies by API version)
    time_coord = "valid_time" if "valid_time" in ds else "time"
    times = pd.to_datetime(ds[time_coord].values)

    tp  = ds["tp"].mean(dim=["latitude", "longitude"])   # m/day mean
    t2m = ds["t2m"].mean(dim=["latitude", "longitude"])  # K

    # Days per month for unit conversion
    days = np.array([calendar.monthrange(t.year, t.month)[1] for t in times])

    df = pd.DataFrame({
        "precip_mm": tp.values * 1000 * days,   # m/day -> mm/month
        "temp_c":    t2m.values - 273.15,        # K -> °C
    }, index=times)
    df.index.name = "date"
    df = df[~df.index.duplicated()].sort_index()

    ds.close()
    os.remove(TMP_NC)
    if tmpdir and os.path.exists(tmpdir):
        import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    df.to_parquet(OUT_PQ)
    print(f"  Saved {len(df)} rows -> {os.path.basename(OUT_PQ)}")
    print(f"  Range : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Precip: {df['precip_mm'].min():.1f} – {df['precip_mm'].max():.1f} mm/month")
    print(f"  Temp  : {df['temp_c'].min():.1f} – {df['temp_c'].max():.1f} °C")
    return df


def main():
    print("\nERA5 West Africa Monthly Ingest")
    print("=" * 40)
    download()
    process()
    print("Done.\n")


if __name__ == "__main__":
    main()
