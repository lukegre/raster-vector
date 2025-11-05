
def enable_xarray_wrapper(func):
    import functools
    import xarray as xr

    """Adds an xarray wrapper for a function without core dimensions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return xr.apply_ufunc(func, *args, kwargs=kwargs)
    return wrapper