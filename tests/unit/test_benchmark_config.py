"""
Unit tests for benchmark_config.py
"""
import pytest
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import patch, mock_open

from mia.benchmark_config import (
    BenchmarkConfig,
    BENCHMARK_CONFIGS,
    get_benchmark_config,
    list_available_benchmarks,
    create_benchmark_report
)


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_benchmark_config_creation(self):
        """Test creating a BenchmarkConfig instance."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark description",
            iterations=15,
            warmup_iterations=3,
            timeout_seconds=120.0,
            memory_limit_mb=512,
            parameters={"key": "value"}
        )

        assert config.name == "test_benchmark"
        assert config.description == "Test benchmark description"
        assert config.iterations == 15
        assert config.warmup_iterations == 3
        assert config.timeout_seconds == 120.0
        assert config.memory_limit_mb == 512
        assert config.parameters == {"key": "value"}

    def test_benchmark_config_default_parameters(self):
        """Test BenchmarkConfig with default parameters."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark description"
        )

        assert config.name == "test_benchmark"
        assert config.description == "Test benchmark description"
        assert config.iterations == 10  # default
        assert config.warmup_iterations == 3  # default
        assert config.timeout_seconds == 300.0  # default
        assert config.memory_limit_mb == 1024  # default
        assert config.parameters == {}  # default

    def test_benchmark_config_post_init(self):
        """Test __post_init__ method initializes parameters dict."""
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark description",
            parameters=None
        )

        assert config.parameters == {}

    def test_benchmark_config_with_parameters(self):
        """Test BenchmarkConfig with provided parameters."""
        params = {"input_sizes": [100, 500], "models": ["test1", "test2"]}
        config = BenchmarkConfig(
            name="test_benchmark",
            description="Test benchmark description",
            parameters=params
        )

        assert config.parameters == params


class TestBenchmarkConfigs:
    """Test predefined benchmark configurations."""

    def test_benchmark_configs_exist(self):
        """Test that BENCHMARK_CONFIGS contains expected benchmarks."""
        expected_benchmarks = [
            'text_processing',
            'audio_processing',
            'vision_processing',
            'multimodal_processing',
            'memory_operations',
            'concurrent_requests'
        ]

        for benchmark_name in expected_benchmarks:
            assert benchmark_name in BENCHMARK_CONFIGS
            assert isinstance(BENCHMARK_CONFIGS[benchmark_name], BenchmarkConfig)

    def test_text_processing_config(self):
        """Test text_processing benchmark configuration."""
        config = BENCHMARK_CONFIGS['text_processing']

        assert config.name == 'text_processing'
        assert config.description == 'Benchmark text processing and LLM inference'
        assert config.iterations == 20
        assert config.warmup_iterations == 5
        assert config.parameters is not None
        assert 'input_sizes' in config.parameters
        assert 'models' in config.parameters

    def test_audio_processing_config(self):
        """Test audio_processing benchmark configuration."""
        config = BENCHMARK_CONFIGS['audio_processing']

        assert config.name == 'audio_processing'
        assert config.description == 'Benchmark audio transcription and processing'
        assert config.iterations == 10
        assert config.warmup_iterations == 3
        assert config.parameters is not None
        assert 'audio_lengths' in config.parameters
        assert 'sample_rates' in config.parameters

    def test_vision_processing_config(self):
        """Test vision_processing benchmark configuration."""
        config = BENCHMARK_CONFIGS['vision_processing']

        assert config.name == 'vision_processing'
        assert config.description == 'Benchmark image analysis and processing'
        assert config.iterations == 15
        assert config.warmup_iterations == 4
        assert config.parameters is not None
        assert 'image_sizes' in config.parameters
        assert 'formats' in config.parameters

    def test_multimodal_processing_config(self):
        """Test multimodal_processing benchmark configuration."""
        config = BENCHMARK_CONFIGS['multimodal_processing']

        assert config.name == 'multimodal_processing'
        assert config.description == 'Benchmark combined multimodal processing'
        assert config.iterations == 10
        assert config.warmup_iterations == 3
        assert config.parameters is not None
        assert 'text_length' in config.parameters
        assert 'audio_length' in config.parameters
        assert 'image_size' in config.parameters

    def test_memory_operations_config(self):
        """Test memory_operations benchmark configuration."""
        config = BENCHMARK_CONFIGS['memory_operations']

        assert config.name == 'memory_operations'
        assert config.description == 'Benchmark memory and caching operations'
        assert config.iterations == 50
        assert config.warmup_iterations == 10
        assert config.parameters is not None
        assert 'cache_sizes' in config.parameters
        assert 'item_sizes' in config.parameters

    def test_concurrent_requests_config(self):
        """Test concurrent_requests benchmark configuration."""
        config = BENCHMARK_CONFIGS['concurrent_requests']

        assert config.name == 'concurrent_requests'
        assert config.description == 'Benchmark concurrent request handling'
        assert config.iterations == 5
        assert config.warmup_iterations == 2
        assert config.parameters is not None
        assert 'concurrency_levels' in config.parameters
        assert 'request_types' in config.parameters


class TestGetBenchmarkConfig:
    """Test get_benchmark_config function."""

    def test_get_valid_benchmark_config(self):
        """Test getting a valid benchmark configuration."""
        config = get_benchmark_config('text_processing')

        assert isinstance(config, BenchmarkConfig)
        assert config.name == 'text_processing'

    def test_get_invalid_benchmark_config(self):
        """Test getting an invalid benchmark configuration raises ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark: invalid_benchmark"):
            get_benchmark_config('invalid_benchmark')


class TestListAvailableBenchmarks:
    """Test list_available_benchmarks function."""

    def test_list_available_benchmarks(self):
        """Test listing all available benchmark names."""
        benchmarks = list_available_benchmarks()

        assert isinstance(benchmarks, list)
        assert len(benchmarks) == 6
        assert 'text_processing' in benchmarks
        assert 'audio_processing' in benchmarks
        assert 'vision_processing' in benchmarks
        assert 'multimodal_processing' in benchmarks
        assert 'memory_operations' in benchmarks
        assert 'concurrent_requests' in benchmarks


class TestCreateBenchmarkReport:
    """Test create_benchmark_report function."""

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_create_benchmark_report(self, mock_json_dump, mock_file, mock_makedirs):
        """Test creating a benchmark report."""
        results = {
            'text_processing': {'status': 'completed', 'average_time': 0.5},
            'memory_operations': {'status': 'completed', 'average_time': 0.1}
        }

        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.timestamp.return_value = 1234567890
            mock_datetime.now.return_value.isoformat.return_value = '2023-01-01T00:00:00'

            report_path = create_benchmark_report(results, 'test_output_dir')

            # Verify directory creation
            mock_makedirs.assert_called_once_with('test_output_dir', exist_ok=True)

            # Verify file operations
            mock_file.assert_called_once()
            expected_path = os.path.join('test_output_dir', 'benchmark_report_1234567890.json')
            assert report_path == expected_path

            # Verify JSON dump was called
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            report_data = args[0]

            assert 'timestamp' in report_data
            assert 'results' in report_data
            assert report_data['results'] == results

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_create_benchmark_report_default_dir(self, mock_json_dump, mock_file, mock_makedirs):
        """Test creating a benchmark report with default output directory."""
        results = {'test': 'data'}

        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.timestamp.return_value = 1234567890

            report_path = create_benchmark_report(results)

            # Verify default directory
            mock_makedirs.assert_called_once_with('benchmarks/results', exist_ok=True)

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_create_benchmark_report_with_complex_data(self, mock_json_dump, mock_file, mock_makedirs):
        """Test creating a benchmark report with complex data types."""
        results = {
            'benchmark1': {
                'iterations': [{'time': 0.1}, {'time': 0.2}],
                'summary': {'avg': 0.15, 'min': 0.1, 'max': 0.2}
            }
        }

        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.timestamp.return_value = 1234567890

            create_benchmark_report(results)

            # Verify JSON dump handles complex data
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            report_data = args[0]

            assert report_data['results'] == results
            # Verify default=str handles datetime serialization
            assert 'default' in kwargs
            assert callable(kwargs['default'])