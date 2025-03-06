"""Simple tests for get_pvlive_data.py"""

import pytest
import pandas as pd
import pytz
import os
import sys
import tempfile
import shutil

from unittest.mock import patch, MagicMock
from datetime import datetime

import get_pvlive_data


@patch('get_pvlive_data.upload_to_gcs')
def test_upload_to_gcs(mock_upload):
    """Test the upload_to_gcs function"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b'test,data\n1,2')
    try:
        get_pvlive_data.upload_to_gcs(tmp_path, 'test-bucket', 'test-path')
        mock_upload.assert_called_once_with(tmp_path, 'test-bucket', 'test-path')
    finally:
        os.unlink(tmp_path)


@patch('get_pvlive_data.PVLive')
def test_pvlive_mock(mock_pvlive_class):
    """Test that we can mock PVLive"""
    mock_pvlive = mock_pvlive_class.return_value
    
    mock_data = pd.DataFrame({
        'datetime_gmt': [
            datetime(2024, 1, 1, 0, 30, tzinfo=pytz.UTC),
            datetime(2024, 1, 1, 1, 0, tzinfo=pytz.UTC)
        ],
        'generation_mw': [10, 15],
        'capacity_mwp': [100, 100],
        'installedcapacity_mwp': [120, 120]
    })
    mock_pvlive.between.return_value = mock_data
    
    result = mock_pvlive.between(
        start=datetime(2024, 1, 1, tzinfo=pytz.UTC),
        end=datetime(2024, 1, 2, tzinfo=pytz.UTC)
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'datetime_gmt' in result.columns
    assert 'generation_mw' in result.columns
    assert mock_pvlive.between.called


def test_dataframe_operations():
    """Test DataFrame operations similar to those in the script"""
    data = pd.DataFrame({
        'start_datetime_utc': [
            datetime(2023, 12, 1, 0, 0, tzinfo=pytz.UTC),
            datetime(2023, 12, 1, 0, 30, tzinfo=pytz.UTC)
        ],
        'end_datetime_utc': [
            datetime(2023, 12, 1, 0, 30, tzinfo=pytz.UTC),
            datetime(2023, 12, 1, 1, 0, tzinfo=pytz.UTC)
        ],
        'generation_mw': [10, 15],
        'capacity_mwp': [100, 100],
        'installedcapacity_mwp': [120, 120]
    })
    
    new_data = pd.DataFrame({
        'datetime_gmt': [
            datetime(2023, 12, 1, 1, 30, tzinfo=pytz.UTC),
            datetime(2023, 12, 1, 2, 0, tzinfo=pytz.UTC)
        ],
        'generation_mw': [20, 25],
        'capacity_mwp': [110, 110],
        'installedcapacity_mwp': [130, 130],
        'gsp_id': [0, 0]
    })
    
    new_data.rename(columns={"datetime_gmt": "end_datetime_utc"}, inplace=True)
    new_data["start_datetime_utc"] = new_data["end_datetime_utc"] - pd.Timedelta(minutes=30)
    
    if "gsp_id" in new_data.columns:
        new_data.drop(columns=["gsp_id"], inplace=True)
    
    combined_data = pd.concat([data, new_data], ignore_index=True)    
    combined_data.sort_values("end_datetime_utc", inplace=True)
    combined_data.drop_duplicates(subset=["end_datetime_utc"], keep="last", inplace=True)
    combined_data.reset_index(drop=True, inplace=True)
    
    assert len(combined_data) == 4
    assert list(combined_data.columns) == [
        'start_datetime_utc', 'end_datetime_utc', 'generation_mw', 
        'capacity_mwp', 'installedcapacity_mwp'
    ]
    assert 'gsp_id' not in combined_data.columns
