"""Tests for abcd_tools.utils.config_loader."""

import tempfile
from pathlib import Path

import pytest

from abcd_tools.utils import config_loader


def test_load_yaml():
    """Test loading YAML configuration file."""
    # Create temporary YAML file
    yaml_content = """
test_key: test_value
nested:
  key1: value1
  key2: value2
list_items:
  - item1
  - item2
  - item3
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Load the YAML file
        result = config_loader.load_yaml(temp_path)

        # Assertions
        assert isinstance(result, dict)
        assert result['test_key'] == 'test_value'
        assert 'nested' in result
        assert result['nested']['key1'] == 'value1'
        assert result['nested']['key2'] == 'value2'
        assert 'list_items' in result
        assert len(result['list_items']) == 3
        assert 'item1' in result['list_items']
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_save_yaml():
    """Test saving YAML configuration file."""
    test_data = {
        'key1': 'value1',
        'key2': 'value2',
        'nested': {
            'inner_key': 'inner_value'
        },
        'list': [1, 2, 3]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name

    try:
        # Save the data
        config_loader.save_yaml(test_data, temp_path)

        # Load it back and verify
        loaded_data = config_loader.load_yaml(temp_path)

        assert loaded_data == test_data
        assert loaded_data['key1'] == 'value1'
        assert loaded_data['nested']['inner_key'] == 'inner_value'
        assert loaded_data['list'] == [1, 2, 3]
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_load_yaml_file_not_found():
    """Test loading non-existent YAML file raises error."""
    with pytest.raises(FileNotFoundError):
        config_loader.load_yaml('/nonexistent/path/file.yaml')
