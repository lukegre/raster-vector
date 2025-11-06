# Raster - Vector conversion tool

This package is built on the geopandas and xarray libraries to provide easy conversion between raster and vector data formats.

See the [demo notebook](demo.ipynb) for usage examples.

## Installation
You can install the package using uv:

```bash
uv add git+https://gitlab.com/geospatial-tools/raster-vector.git
```

or pip:

```bash
pip install git+https://gitlab.com/geospatial-tools/raster-vector.git
```

Alternatively, you can clone the repository and install it in editable mode for development:

```bash
git clone https://gitlab.com/geospatial-tools/raster-vector.git
cd raster-vector
pip install -e .
```

## Usage

Import the package and use the provided functions to convert between raster and vector formats.

```python
import raster_vector as rv

# load your raster data as an xarray DataArray
# da_mask_int = ...

# Convert raster integer mask to vector polygons
gdf_polygons = da_mask_int.rv.to_polygons()
# and back to raster, but now with index of vector objects
da_objects = gdf_polygons.rv.to_raster(da_target=da_mask_int)
```

For more detailed examples, refer to the [demo notebook](demo.ipynb).