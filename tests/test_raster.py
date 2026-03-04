import numpy as np
import pytest
import xarray as xr

from raster_vector.raster import _auto_crs, get_bounds_latlon, prep_raster


def test_prep_raster_renames_dims():
    """Test that prep_raster renames dimensions to 'x' and 'y'."""
    da = xr.DataArray(
        np.zeros((5, 5)),
        dims=("lat", "lon"),
        coords={"lat": [1, 2, 3, 4, 5], "lon": [1, 2, 3, 4, 5]},
    )
    da_prepped = prep_raster(da, x_axis=-1, y_axis=-2)
    assert "x" in da_prepped.dims
    assert "y" in da_prepped.dims
    assert da_prepped.dims == ("y", "x")


def test_prep_raster_sorts_y():
    """Test that prep_raster sorts 'y' in descending order."""
    da = xr.DataArray(
        np.zeros((5, 5)), dims=("y", "x"), coords={"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]}
    )
    da_prepped = prep_raster(da)
    assert (da_prepped.y.values == np.array([5, 4, 3, 2, 1])).all()


def test_auto_crs_warns(caplog):
    """Test that _auto_crs issues a warning when bounds match lat/lon."""
    from loguru import logger

    def sink(message):
        import logging

        logging.log(message.record["level"].no, message.record["message"])

    handler_id = logger.add(sink)
    try:
        da = xr.DataArray(
            np.zeros((5, 5)),
            dims=("y", "x"),
            coords={"y": [10, 11, 12, 13, 14], "x": [20, 21, 22, 23, 24]},
        )
        da_prepped = _auto_crs(da)
        assert "No CRS found in DataArray." in caplog.text
        assert da_prepped.rio.crs is None
    finally:
        logger.remove(handler_id)


def test_get_bounds_latlon(sample_da_bool):
    """Test get_bounds_latlon returns correct bounding box tuple."""
    bounds = get_bounds_latlon(sample_da_bool)
    assert isinstance(bounds, tuple)
    assert len(bounds) == 4
    # rioxarray/rasterio calculate bounds based on pixel edges.
    # For linspace(0, 1, 10), pixel spacing is 1/9, so edges are at -1/18 and 1 + 1/18
    expected_val = 1 / 18
    assert bounds[0] == pytest.approx(-expected_val)
    assert bounds[1] == pytest.approx(-expected_val)
    assert bounds[2] == pytest.approx(1 + expected_val)
    assert bounds[3] == pytest.approx(1 + expected_val)
