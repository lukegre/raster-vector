import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import xarray as xr
from loguru import logger
from rasterio.features import rasterize
from shapely import geometry
from shapely.geometry import mapping


def raster_bool_to_vector(
    da: xr.DataArray, combine_polygons=False, buffer_dist=0, simplify_dist=0
) -> gpd.GeoDataFrame:
    """
    Converts a rasterized mask to a vectorized representation.

    Parameters
    ----------
    da : xr.DataArray (bool)
        The rasterized mask to convert to a polygon
    combine_polygons : bool, optional
        If True, all polygons are combined into a single polygon. if False, each
        connected component is a separate polygon. The default is False.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the vectorized representation of the mask.
    """

    assert da.dtype == bool, "Input array must be boolean"
    assert da.ndim == 2, "Input array must be 2D"
    transform = da.rio.transform()

    arr = da.values.astype(np.uint8)
    shapes = rasterio.features.shapes(arr, transform=transform)
    crs = da.rio.crs

    if crs is None:
        logger.warning("No CRS found in DataArray, assuming EPSG:4326 (lat/lon)")
        crs = "EPSG:4326"

    def get_coord(s):
        return geometry.shape(s[0])

    polygons = [get_coord(shape) for shape in shapes if shape[1] == 1]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    if combine_polygons:
        gdf = gdf.dissolve()

    if buffer_dist > 0:
        gdf["geometry"] = gdf.buffer(buffer_dist).buffer(-buffer_dist)
    if simplify_dist > 0:
        gdf["geometry"] = gdf.simplify(simplify_dist)

    gdf = gdf.set_crs(crs)

    return gdf


def raster_int_to_vector(
    da: xr.DataArray, names=None, column_name="category", buffer_dist=0, simplify_dist=0
) -> gpd.GeoDataFrame:
    """
    Converts a rasterized mask with several classes to a vectorized representation.

    Parameters
    ----------
    da : xr.DataArray (int)
        The rasterized mask with several categories.
    names : list, optional
        The names of the categories. The default is None, in which case the categories are numbered.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the vectorized representation of the mask.
    """

    assert da.dtype == int, "Input array must be integer"

    mask_values = np.sort(np.unique(da.values))
    n_categories = mask_values.size

    if names is None:
        names = [str(i) for i in mask_values]
    else:
        assert len(names) == n_categories, (
            f"Number of names (n={len(names)}) must match number of categories (n={n_categories})"
        )

    polygons = []
    category_labels = []
    for m, name in zip(mask_values, names, strict=True):
        # logger.debug(f"Converting category {name} [{m}] to vector")
        mask = da == m
        gdf = raster_bool_to_vector(
            da=mask, combine_polygons=True, simplify_dist=simplify_dist, buffer_dist=buffer_dist
        )
        polygons.append(gdf)
        category_labels.extend([name] * len(gdf))

    polygons = pd.concat(polygons, ignore_index=True)
    polygons[column_name] = category_labels

    return polygons


def polygon_to_raster_bool(polygon, da_target):
    """
    Convert a Shapely polygon to a binary mask that matches the grid of an xarray.DataArray.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or geopandas.GeoDataFrame
        The polygon to convert to a raster mask.
    da_target : xr.DataArray
        The target grid to match the mask to (spatial dimensions must be 'x' and 'y').

    Returns
    -------
    mask_da : xr.DataArray
        A boolean DataArray with the mask on the target grid.
    """

    if isinstance(polygon, (gpd.GeoSeries, gpd.GeoDataFrame)):
        if (
            polygon.crs is not None
            and da_target.rio.crs is not None
            and polygon.crs != da_target.rio.crs
        ):
            polygon = polygon.to_crs(da_target.rio.crs)
        polygon = polygon.union_all()

    # Get the spatial dimensions of the data array
    if "x" in da_target.dims and "y" in da_target.dims:
        x, y = "x", "y"
    else:
        raise ValueError("Data array must have 'x' and 'y' dimensions")

    # make sure lat is descending, otherwise upside down coords
    da_target = da_target.sortby("y", ascending=False)

    # Define the transformation from pixel coordinates to geographical coordinates
    transform = rasterio.transform.from_bounds(
        min(da_target[x].values),
        min(da_target[y].values),
        max(da_target[x].values),
        max(da_target[y].values),
        len(da_target[x]),
        len(da_target[y]),
    )

    # Rasterize the polygon
    mask = rasterize(
        [mapping(polygon)],
        out_shape=(len(da_target[y]), len(da_target[x])),
        transform=transform,
        fill=0,
        out=None,
        all_touched=True,
        dtype=np.uint8,
    )

    # Create a DataArray from the mask
    mask_da = xr.DataArray(mask, dims=(y, x), coords={y: da_target[y], x: da_target[x]}).astype(
        bool
    )
    mask_da = mask_da.rio.write_crs(da_target.rio.crs)

    return mask_da


def polygons_to_raster_int(
    df: gpd.GeoDataFrame, da_target: xr.DataArray, by_column=None, **kwargs
) -> xr.DataArray:
    """
    Convert a GeoDataFrame with polygons to a raster mask with integer values.

    Each row of polygons in the GeoDataFrame is converted to a separate integer
    value in the raster mask. The integer values are assigned in the order of the
    rows in the GeoDataFrame.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The GeoDataFrame with polygons to convert to a raster mask.
    da_target : xr.DataArray
        The target grid to match the mask to (spatial dimensions must be 'x' and 'y').
    by_column : str, optional
        The column in the GeoDataFrame to group the polygons by. If None, then each
        row is converted to a separate integer value. The default is None.
    kwargs : dict, optional
        Ignored, kept for backwards compatibility.

    Returns
    -------
    xr.DataArray
        A DataArray with the raster mask with integer values.
    """

    if df.crs is not None and da_target.rio.crs is not None and df.crs != da_target.rio.crs:
        df = df.to_crs(da_target.rio.crs)

    if by_column is not None:
        assert by_column in df.columns, f"Column {by_column} not found in DataFrame"
        df = df.dissolve(by=by_column).reset_index()

    # Get the spatial dimensions of the data array
    if "x" in da_target.dims and "y" in da_target.dims:
        x, y = "x", "y"
    else:
        raise ValueError("Data array must have 'x' and 'y' dimensions")

    da_target = da_target.sortby("y", ascending=False)

    transform = rasterio.transform.from_bounds(
        min(da_target[x].values),
        min(da_target[y].values),
        max(da_target[x].values),
        max(da_target[y].values),
        len(da_target[x]),
        len(da_target[y]),
    )

    shapes = ((geom, i + 1) for i, geom in enumerate(df.geometry))

    mask = rasterize(
        shapes,
        out_shape=(len(da_target[y]), len(da_target[x])),
        transform=transform,
        fill=0,
        out=None,
        all_touched=True,
        dtype=np.int32,
    )

    polygons = xr.DataArray(mask, dims=(y, x), coords={y: da_target[y], x: da_target[x]}).astype(
        int
    )
    polygons = polygons.rio.write_crs(da_target.rio.crs)

    return polygons
