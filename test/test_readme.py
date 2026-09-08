"""Test code snippets in README.md file"""

import numpy as np
import zarr

from zarr_sqlite import SQLiteStore

def test_readme_snippet(tmp_path):
    with SQLiteStore(tmp_path / "my_zarr_file.zarrdb") as store:
        root = zarr.create_group(store=store)
        foo = root.create_group('foo')
        bar = foo.create_group('bar')
        z1 = bar.create_array(name='baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
        z1[:] = 42

    # Verify the SQLite database file was created on disk
    db_path = tmp_path / "my_zarr_file.zarrdb"
    assert db_path.exists()

    # Reopen read-only and verify the data was persisted correctly
    with SQLiteStore(db_path, read_only=True) as store:
        root = zarr.open_group(store=store, mode='r')
        assert "foo" in root
        assert "foo/bar" in root

        z1 = root["foo/bar/baz"]
        assert z1.shape == (10000, 10000)
        assert z1.dtype == np.dtype('int32')

        # Read a small sample rather than the full 400 MB array
        assert z1[0, 0] == 42
        np.testing.assert_array_equal(z1[0:5, 0:5], 42)
