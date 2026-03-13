import geopandas as gpd
import numpy as np
import xarray as xr

from raster_vector.conversion import (
    polygon_to_raster_bool,
    polygons_to_raster_int,
    raster_bool_to_vector,
    raster_int_to_vector,
)


def test_raster_bool_to_vector_basic(sample_da_bool):
    """Test basic boolean raster to vector conversion."""
    gdf = raster_bool_to_vector(sample_da_bool)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2  # Two blobs in sample_da_bool
    assert gdf.crs == sample_da_bool.rio.crs


def test_raster_bool_to_vector_combine(sample_da_bool):
    """Test boolean raster to vector with combine_polygons=True."""
    gdf = raster_bool_to_vector(sample_da_bool, combine_polygons=True)
    assert len(gdf) == 1
    assert isinstance(gdf.iloc[0].geometry, (gpd.base.BaseGeometry,))


def test_raster_bool_to_vector_buffer(sample_da_bool):
    """Test boolean raster to vector with buffer and simplify."""
    gdf = raster_bool_to_vector(sample_da_bool, buffer_dist=0.01, simplify_dist=0.01)
    assert len(gdf) == 2
    assert not gdf.is_empty.any()


def test_raster_int_to_vector_basic(sample_da_int):
    """Test basic integer raster to vector conversion."""
    gdf = raster_int_to_vector(sample_da_int, column_name="class")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "class" in gdf.columns
    # Unique values in sample_da_int are 0, 1, 2. Background (0) is included as a category.
    # So 3 categories total.
    assert len(gdf["class"].unique()) == 3


def test_raster_int_to_vector_large_categories():
    """Test that a large number of categories works without ValueError."""
    data = np.arange(205).reshape((5, 41))
    da = xr.DataArray(data, dims=("y", "x"), name="too_many")
    # should not raise ValueError anymore!
    gdf = raster_int_to_vector(da)
    assert len(gdf) == 205


def test_raster_bool_to_vector_ascending_lat_positions():
    """Polygons must appear at the correct geographic position when y is ascending."""
    # Put True values in rows 0-2 of an ascending-lat array (southern region: lat 0..0.222)
    data = np.zeros((10, 10), dtype=bool)
    data[0:3, 3:7] = True

    lat = np.linspace(0, 1, 10)  # ascending: row 0 = lat 0 (south), row 9 = lat 1 (north)
    lon = np.linspace(0, 1, 10)
    da = xr.DataArray(data, coords={"y": lat, "x": lon}, dims=("y", "x"))
    da.rio.write_crs("EPSG:4326", inplace=True)

    gdf = raster_bool_to_vector(da)

    assert len(gdf) == 1
    centroid_y = gdf.geometry.centroid.y.values[0]
    assert centroid_y < 0.5, (
        f"Polygon centroid at lat={centroid_y:.3f}, expected in southern half (< 0.5). "
        "Likely a y-axis flip due to ascending latitude not being sorted before rasterization."
    )


def test_polygon_to_raster_bool_basic(sample_gdf, sample_da_bool):
    """Test basic polygon to boolean raster conversion."""
    poly = sample_gdf.geometry.iloc[0]
    mask = polygon_to_raster_bool(poly, sample_da_bool)
    assert isinstance(mask, xr.DataArray)
    assert mask.dtype == bool
    assert mask.shape == sample_da_bool.shape


def test_polygons_to_raster_int_basic(sample_gdf, sample_da_int):
    """Test basic multiple polygon to integer raster conversion."""
    mask = polygons_to_raster_int(sample_gdf, sample_da_int)
    assert isinstance(mask, xr.DataArray)
    assert mask.dtype == int
    assert mask.shape == sample_da_int.shape
    assert mask.max() == 2  # Two polygons in sample_gdf


def test_polygons_to_raster_int_overlap(sample_gdf, sample_da_int):
    """Test that overlapping polygons rasterize successfully (last one overwrites)."""
    # Create overlapping polygons: poly1 and poly1 (exact overlap)
    gdf_overlap = gpd.GeoDataFrame(
        {"category": ["A", "B"]},
        geometry=[sample_gdf.geometry.iloc[0], sample_gdf.geometry.iloc[0]],
        crs=sample_gdf.crs,
    )

    mask = polygons_to_raster_int(gdf_overlap, sample_da_int)

    assert mask.max() == 2


def test_polygon_to_raster_bool_crs_mismatch(sample_da_bool):
    """Test that polygon_to_raster_bool reprojects if CRS differs."""
    # Create a polygon in Web Mercator (EPSG:3857)
    # 0,0 in 4326 is 0,0 in 3857.
    # Let's create a small box in 3857.
    from shapely.geometry import box

    poly_3857 = box(0, 0, 1000, 1000)
    gdf_3857 = gpd.GeoDataFrame(geometry=[poly_3857], crs="EPSG:3857")

    # Target is sample_da_bool (EPSG:4326, bounds approx 0,0 to 1,1)
    # If it DOES NOT reproject, it will look for a 1000x1000 box in 4326 space (huge).
    # If it DOES reproject, it will be a tiny sliver near the origin.
    mask = polygon_to_raster_bool(gdf_3857, sample_da_bool)

    assert mask.any()
    # Check that it's not the whole array (which might happen if scales are wildly
    # off and we don't reproject)
    assert not mask.all()
