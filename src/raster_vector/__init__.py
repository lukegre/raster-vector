import rioxarray as rxr  # noqa: F401

from . import accessors, projection, raster, utils, vector
from .conversion import (
    polygon_to_raster_bool,
    polygons_to_raster_int,
    raster_bool_to_vector,
    raster_int_to_vector,
)

__all__ = [
    "accessors",
    "polygon_to_raster_bool",
    "polygons_to_raster_int",
    "projection",
    "raster",
    "raster_bool_to_vector",
    "raster_int_to_vector",
    "utils",
    "vector",
]


def info():
    import numpy as np
    import xarray as xr

    da = xr.DataArray(
        np.ones([1, 1]), dims=["x", "y"], coords={"x": [0], "y": [0]}, name="dummy"
    ).rio.write_crs(4326)
    df = da.to_dataframe()

    accessors_help = ""
    accessors_help += str(df.rv) + "\n"
    accessors_help += str(da.rv) + "\n"
    accessors_help += str(da.morph)

    print("The following accessors have been added:\n\n" + accessors_help)
