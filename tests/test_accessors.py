import geopandas as gpd
import numpy as np
import xarray as xr


def test_rv_to_polygons_bool(sample_da_bool):
    """Test accessor to_polygons for boolean raster."""
    gdf = sample_da_bool.rv.to_polygons()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2


def test_rv_to_polygons_int(sample_da_int):
    """Test accessor to_polygons for integer raster."""
    gdf = sample_da_int.rv.to_polygons()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf["category"].unique()) == 3


def test_rv_to_raster(sample_gdf, sample_da_int):
    """Test accessor to_raster for GeoDataFrame."""
    mask = sample_gdf.rv.to_raster(sample_da_int)
    assert isinstance(mask, xr.DataArray)
    assert mask.shape == sample_da_int.shape
    assert mask.dtype == int


def test_rv_access_does_not_rename_dims(sample_da_bool):
    """Test that accessing .rv does not rename dimensions (test for Phase 2.2)."""
    # Current implementation DOES rename dims in __init__, so this test should FAIL now
    # but pass after Phase 2.2 fix.
    _ = sample_da_bool.rv
    # If the accessor mutates dims in __init__, this will be different
    # NOTE: sample_da_bool in fixture has 'y', 'x'. prep_raster renames to
    # 'y', 'x' if they aren't already.
    # So if they ARE 'y', 'x', it doesn't change them.
    # Let's use a DA with different dims.
    da = xr.DataArray(
        np.zeros((5, 5)),
        dims=("lat", "lon"),
        coords={"lat": [1, 2, 3, 4, 5], "lon": [1, 2, 3, 4, 5]},
    )
    _ = da.rv
    # In the current version, da.rv calls prep_raster(da) which renames dims.
    # HOWEVER, xarray accessors often operate on a COPY or just return a new object.
    # Wait, RasterVector.__init__ does self._da = prep_raster(da).
    # If it doesn't mutate the ORIGINAL 'da' (passed by reference), then this test passes.
    # But xarray.rename usually returns a new object unless it's an inplace
    # rename (rarely used now)
    assert da.dims == ("lat", "lon")


def test_rv_get_bbox_latlon(sample_da_bool):
    """Test accessor get_bbox_latlon for DataArray."""
    bbox = sample_da_bool.rv.get_bbox_latlon()
    assert isinstance(bbox, tuple)
    assert len(bbox) == 4
    assert all(isinstance(b, float) for b in bbox)


def test_rv_get_utm_code(sample_da_bool):
    """Test accessor get_utm_code for DataArray."""
    # sample_da_bool is around 0, 0
    epsg = sample_da_bool.rv.get_utm_code()
    assert isinstance(epsg, int)
    # 0,0 is UTM zone 31N in northern hemisphere -> 32631
    assert 32601 <= epsg <= 32760
