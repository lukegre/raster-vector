import numpy as np
import xarray as xr

from raster_vector.conversion import raster_bool_to_vector


def test_raster_bool_to_vector_with_holes():
    """Test that raster_bool_to_vector correctly identifies and retains polygon holes."""
    # Create a 10x10 boolean array
    data = np.zeros((10, 10), dtype=bool)
    # Create a 6x6 square
    data[2:8, 2:8] = True
    # Create a 2x2 hole in the middle
    data[4:6, 4:6] = False

    lon = np.linspace(0, 1, 10)
    lat = np.linspace(0, 1, 10)

    da = xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"), name="mask_with_hole")
    da.rio.write_crs("EPSG:4326", inplace=True)

    gdf = raster_bool_to_vector(da)

    # There should only be 1 polygon detected
    assert len(gdf) == 1

    geom = gdf.geometry.iloc[0]
    # The polygon should have 1 interior ring (the hole)
    assert len(geom.interiors) == 1
