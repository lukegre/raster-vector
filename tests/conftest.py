import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon


@pytest.fixture
def sample_da_bool():
    """Small 2D boolean DataArray with EPSG:4326 CRS."""
    data = np.zeros((10, 10), dtype=bool)
    data[2:5, 2:5] = True
    data[7:9, 7:9] = True

    lon = np.linspace(0, 1, 10)
    lat = np.linspace(0, 1, 10)

    da = xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"), name="mask")
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da


@pytest.fixture
def sample_da_int():
    """Small 2D integer DataArray with EPSG:4326 CRS."""
    data = np.zeros((10, 10), dtype=int)
    data[2:5, 2:5] = 1
    data[6:8, 6:8] = 2

    lon = np.linspace(0, 1, 10)
    lat = np.linspace(0, 1, 10)

    da = xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"), name="categories")
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da


@pytest.fixture
def sample_gdf():
    """Sample GeoDataFrame with simple polygons."""
    poly1 = Polygon([(0.2, 0.2), (0.5, 0.2), (0.5, 0.5), (0.2, 0.5)])
    poly2 = Polygon([(0.6, 0.6), (0.8, 0.6), (0.8, 0.8), (0.6, 0.8)])

    gdf = gpd.GeoDataFrame(
        {"category": ["A", "B"], "value": [1, 2]}, geometry=[poly1, poly2], crs="EPSG:4326"
    )
    return gdf
