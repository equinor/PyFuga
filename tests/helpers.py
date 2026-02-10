import xarray as xr

OLD_TO_NEW = {
    "dyxu0": "dbx_const",
    "dyxu1": "dbx_lin",
    "dyxv0": "dby_const",
    "dyxv1": "dby_lin",
    "dyxw0": "dbz_const",
    "dyxw1": "dbz_lin",
}

NEW_TO_OLD = {new: old for old, new in OLD_TO_NEW.items()}


def expose_old_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Return a view of the dataset with old variable names (dyxu0, etc.)
    for compatibility with older code/tests.
    """
    rename = {new: old for new, old in NEW_TO_OLD.items() if new in ds.data_vars}
    if rename:
        ds = ds.rename(rename)
    return ds


def expose_new_names(ds: xr.Dataset) -> xr.Dataset:
    """
    Return a view of the dataset with new variable names (dbx_const, etc.)
    """
    rename = {old: new for old, new in OLD_TO_NEW.items() if old in ds.data_vars}
    if rename:
        ds = ds.rename(rename)
    return ds
