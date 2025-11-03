"""
script/data_utils.py
--------------------------
This helper script provides utility functions for preprocessing and harmonizing
geospatial datasets using the xarray library. It includes tools for regridding 
datasets onto a common spatial grid and aligning multiple datasets along a shared 
time dimension — tasks that are often essential for multi-source climate, ocean, 
and environmental data analysis.

Functions:
-----------
1. regrid_to_target(source_ds, target_ds, method='nearest'):
    Regrids the source dataset to match the longitude and latitude grid of a 
    target dataset using xarray interpolation methods (default: 'nearest').
    Useful for ensuring spatial consistency before data comparison or merging.

2. align_datasets(datasets):
    Aligns multiple xarray datasets based on their common time coordinates.
    It verifies that each dataset has a 'time' dimension, identifies the 
    overlapping time range, and returns a list of time-synchronized datasets.

Dependencies:
    - xarray

Author: Farshid Daryabor
Date: July 2025
"""


import xarray as xr

def regrid_to_target(source_ds, target_ds, method='nearest'):
    """Regrid source dataset to the grid of target dataset."""
    regridded = source_ds.interp(
        lon=target_ds.lon,
        lat=target_ds.lat,
        method=method
    )
    return regridded

def align_datasets(datasets):
    """Align multiple datasets by their shared time coordinate."""
    # Ensure all datasets use the same time resolution
    for i in range(len(datasets)):
        if 'time' not in datasets[i].dims:
            raise ValueError("Dataset does not have 'time' dimension.")

    # Find common time indices
    common_time = datasets[0].time
    for ds in datasets[1:]:
        common_time = common_time.intersection(ds.time)
    
    aligned = [ds.sel(time=common_time) for ds in datasets]
    return aligned

