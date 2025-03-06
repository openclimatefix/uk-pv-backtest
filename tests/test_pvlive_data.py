"""Tests for the get_pvlive_data module."""

import os
import tempfile
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pvlive_api import PVLive

from get_pvlive_data import (
    GCS_BUCKET,
    GCS_NATIONAL_PATH,
    GCS_PATH,
    TEMP_DIR,
    download_zarr_from_gcs,
    get_gsp_pvlive_data,
    get_national_pvlive_data,
    upload_national_data,
    upload_zarr_to_gcs,
)


@pytest.fixture
def mock_pvlive():
    """Create a mock PVLive client with predefined response data."""
    sample_times = pd.date_range("2023-01-01", "2023-01-02", freq="30min")
    mock_national_data = pd.DataFrame({
        "datetime_gmt": sample_times,
        "generation_mw": np.random.rand(len(sample_times)) * 100,
        "capacity_mwp": np.random.rand(len(sample_times)) * 200,
        "installedcapacity_mwp": np.random.rand(len(sample_times)) * 300,
        "gsp_id": [0] * len(sample_times)
    })

    with mock.patch('pvlive_api.PVLive') as mock_pvl:
        mock_pvl_instance = mock_pvl.return_value
        mock_pvl_instance.between.return_value = mock_national_data
        yield mock_pvl


def test_upload_national_data(tmp_path):
    """Test that upload_national_data function works correctly."""
    test_csv = os.path.join(tmp_path, "test.csv")
    test_df = pd.DataFrame({"test": [1, 2, 3]})
    test_df.to_csv(test_csv, index=False)

    with mock.patch('google.cloud.storage.Client') as mock_client:
        mock_bucket = mock_client.return_value.bucket.return_value
        mock_blob = mock_bucket.blob.return_value

        upload_national_data(test_csv, "test-bucket", "test-path.csv")

        mock_client.assert_called_once()
        mock_client.return_value.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("test-path.csv")
        mock_blob.upload_from_filename.assert_called_once_with(test_csv)


@mock.patch('get_pvlive_data.upload_zarr_to_gcs')
@mock.patch('pvlive_api.PVLive')
def test_get_gsp_pvlive_data(mock_pvlive_class, mock_upload_zarr):
    """Test the GSP data retrieval and processing."""
    mock_pvlive = mock.MagicMock()
    mock_pvlive_class.return_value = mock_pvlive

    def mock_between_side_effect(*args, **kwargs):
        entity_id = kwargs.get('entity_id', 0)
        sample_times = pd.date_range("2023-01-01", "2023-01-02", freq="30min")
        return pd.DataFrame({
            "datetime_gmt": sample_times,
            "generation_mw": np.random.rand(len(sample_times)),
            "capacity_mwp": np.random.rand(len(sample_times)),
            "installedcapacity_mwp": np.random.rand(len(sample_times)),
            "gsp_id": [entity_id] * len(sample_times)
        })

    mock_pvlive.between.side_effect = mock_between_side_effect

    with tempfile.TemporaryDirectory() as temp_dir:
        with mock.patch('numpy.arange', return_value=np.array([0, 1, 2])):
            test_times = pd.date_range("2023-01-01", "2023-01-02", freq="30min")

            with mock.patch('pandas.date_range', return_value=test_times):
                with mock.patch('shutil.rmtree'):
                    with mock.patch('get_pvlive_data.TEMP_DIR', temp_dir):
                        get_gsp_pvlive_data()

        mock_upload_zarr.assert_called_once()


@mock.patch('google.cloud.storage.Client')
def test_download_zarr_from_gcs(mock_client):
    """Test downloading Zarr from GCS."""
    mock_bucket = mock_client.return_value.bucket.return_value
    mock_blob1 = mock.MagicMock()
    mock_blob1.name = "test-path/test.zarr/.zgroup"
    mock_blob2 = mock.MagicMock()
    mock_blob2.name = "test-path/test.zarr/.zattrs"

    mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

    with tempfile.TemporaryDirectory() as temp_dir:
        download_zarr_from_gcs("test-bucket", "test-path/test.zarr", temp_dir)

        mock_client.assert_called_once()
        mock_client.return_value.bucket.assert_called_once_with("test-bucket")
        mock_bucket.list_blobs.assert_called_once_with(prefix="test-path/test.zarr")
        assert mock_blob1.download_to_filename.call_count == 1
        assert mock_blob2.download_to_filename.call_count == 1
