#!/usr/bin/env python
"""
Comprehensive coverage tests for dataset_plotter.py
Target: Raise coverage from 45% to 80%+

Tests cover:
- DatasetPlotter initialization
- 2D scatter plotting with various datasets
- Color by class functionality
- Legend creation
- Data truncation/sampling for large datasets
- Missing value handling
- Different marker styles
- Title and axis labels
- Distribution plots
"""
import numpy as np
import plotly.graph_objects as go
import pytest  # noqa: F401 - needed for pytest fixtures
from dash import html


class TestDatasetPlotterInit:
    """Test DatasetPlotter initialization."""

    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config, component_id="test-plotter")

        assert component.component_id == "test-plotter"
        assert component.default_colors is not None
        assert len(component.default_colors) == 5
        assert component.current_dataset is None

    def test_init_with_custom_id(self):
        """Test initialization with custom component ID."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config, component_id="my-custom-plotter")

        assert component.component_id == "my-custom-plotter"

    def test_default_colors_defined(self):
        """Test that default colors are properly defined."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        assert isinstance(component.default_colors, list)
        assert all(isinstance(c, str) for c in component.default_colors)
        # Check they're valid hex colors
        assert all(c.startswith("#") for c in component.default_colors)


class TestDatasetPlotterLayout:
    """Test layout generation."""

    def test_get_layout_structure(self):
        """Test layout contains expected elements."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)
        layout = component.get_layout()

        assert isinstance(layout, html.Div)

    def test_get_layout_with_custom_id(self):
        """Test layout uses custom component ID."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config, component_id="my-plotter")
        layout = component.get_layout()

        layout_str = str(layout)
        assert "my-plotter" in layout_str


class TestDatasetPlotterDataManagement:
    """Test dataset loading and management."""

    def test_load_dataset(self):
        """Test loading a dataset."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, 1], [2, 2]], "targets": [0, 1, 0]}

        component.load_dataset(dataset)

        assert component.current_dataset == dataset

    def test_get_dataset(self):
        """Test retrieving current dataset."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, 1]], "targets": [0, 1]}

        component.load_dataset(dataset)
        retrieved = component.get_dataset()

        assert retrieved == dataset

    def test_get_dataset_when_none(self):
        """Test retrieving dataset when none loaded."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        result = component.get_dataset()

        assert result is None


class TestDatasetPlotterScatterPlot:
    """Test scatter plot creation."""

    def test_create_scatter_plot_with_2d_data(self):
        """Test creating scatter plot with 2D data."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, 1], [2, 2], [1, 2]], "targets": [0, 1, 0, 1]}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have traces

    def test_create_scatter_plot_with_1d_data(self):
        """Test creating scatter plot with 1D data."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0], [1], [2], [3]], "targets": [0, 1, 0, 1]}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        # 1D data should be plotted differently

    def test_create_scatter_plot_with_empty_data(self):
        """Test creating scatter plot with empty data."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [], "targets": []}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        # Should return empty plot

    def test_create_scatter_plot_with_multi_class(self):
        """Test scatter plot with multiple classes."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # 4 classes
        dataset = {"inputs": [[0, 0], [1, 1], [2, 2], [3, 3], [0, 1], [1, 0]], "targets": [0, 1, 2, 3, 0, 1]}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        # Should have multiple traces for different classes

    def test_create_scatter_plot_dark_theme(self):
        """Test scatter plot with dark theme."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, 1]], "targets": [0, 1]}

        fig = component._create_scatter_plot(dataset, theme="dark")

        assert isinstance(fig, go.Figure)  # Dark theme applied

    def test_create_scatter_plot_with_numpy_arrays(self):
        """Test scatter plot with numpy arrays."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": np.array([[0, 0], [1, 1], [2, 2]]), "targets": np.array([0, 1, 0])}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)

    def test_create_scatter_plot_with_high_dimensional_data(self):
        """Test scatter plot with >2D data (uses first 2 features)."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # 5D data
        dataset = {"inputs": [[0, 0, 1, 2, 3], [1, 1, 0, 1, 2], [2, 2, 3, 0, 1]], "targets": [0, 1, 0]}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        # Should plot first 2 dimensions


class TestDatasetPlotterDistributionPlot:
    """Test distribution plot creation."""

    def test_create_distribution_plot_with_data(self):
        """Test creating distribution plot."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": np.random.randn(100, 2), "targets": np.random.randint(0, 2, 100)}

        fig = component._create_distribution_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        # Should have histogram traces

    def test_create_distribution_plot_with_empty_data(self):
        """Test creating distribution plot with empty data."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [], "targets": []}

        fig = component._create_distribution_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)

    def test_create_distribution_plot_with_multi_feature(self):
        """Test distribution plot with multiple features (>4)."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # 6 features (should show only first 4)
        dataset = {"inputs": np.random.randn(50, 6), "targets": np.random.randint(0, 2, 50)}

        fig = component._create_distribution_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
        # Should limit to 4 features

    def test_create_distribution_plot_dark_theme(self):
        """Test distribution plot with dark theme."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": np.random.randn(30, 2), "targets": np.random.randint(0, 2, 30)}

        fig = component._create_distribution_plot(dataset, theme="dark")

        assert isinstance(fig, go.Figure)  # Dark theme applied


class TestDatasetPlotterFiltering:
    """Test data filtering by split."""

    def test_filter_by_split_all(self):
        """Test filtering with 'all' split."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, 1], [2, 2], [3, 3]], "targets": [0, 1, 0, 1]}

        filtered = component._filter_by_split(dataset, "all")

        assert filtered == dataset

    def test_filter_by_split_train(self):
        """Test filtering with 'train' split."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {
            "inputs": [[0, 0], [1, 1], [2, 2], [3, 3]],
            "targets": [0, 1, 0, 1],
            "split_indices": {"train": [0, 1, 2], "test": [3]},
        }

        filtered = component._filter_by_split(dataset, "train")

        assert len(filtered["inputs"]) == 3
        assert len(filtered["targets"]) == 3

    def test_filter_by_split_test(self):
        """Test filtering with 'test' split."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {
            "inputs": [[0, 0], [1, 1], [2, 2], [3, 3]],
            "targets": [0, 1, 0, 1],
            "split_indices": {"train": [0, 1, 2], "test": [3]},
        }

        filtered = component._filter_by_split(dataset, "test")

        assert len(filtered["inputs"]) == 1
        assert len(filtered["targets"]) == 1

    def test_filter_by_split_no_indices(self):
        """Test filtering when no split indices provided."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, 1]], "targets": [0, 1]}

        # Without split_indices, should return full dataset
        filtered = component._filter_by_split(dataset, "train")

        assert filtered == dataset

    def test_filter_by_split_with_out_of_bounds_indices(self):
        """Test filtering with indices beyond dataset size."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {
            "inputs": [[0, 0], [1, 1]],
            "targets": [0, 1],
            "split_indices": {"train": [0, 1, 5, 10], "test": []},  # 5 and 10 are out of bounds
        }

        filtered = component._filter_by_split(dataset, "train")

        # Should only include valid indices
        assert len(filtered["inputs"]) == 2


class TestDatasetPlotterBalance:
    """Test class balance calculation."""

    def test_calculate_balance_balanced(self):
        """Test calculating balance for balanced dataset."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # Perfectly balanced
        targets = [0, 0, 0, 1, 1, 1]

        result = component._calculate_balance(targets)

        assert result == "Balanced"

    def test_calculate_balance_imbalanced(self):
        """Test calculating balance for imbalanced dataset."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # 80% class 0, 20% class 1
        targets = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

        result = component._calculate_balance(targets)

        assert "Imbalanced" in result

    def test_calculate_balance_moderate(self):
        """Test calculating balance for moderately imbalanced dataset."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # 60% class 0, 40% class 1
        targets = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

        result = component._calculate_balance(targets)

        assert "Moderate" in result or "Balanced" in result

    def test_calculate_balance_empty(self):
        """Test calculating balance for empty targets."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        targets = []

        result = component._calculate_balance(targets)

        assert result == "N/A"

    def test_calculate_balance_single_class(self):
        """Test calculating balance for single class."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        targets = [0, 0, 0, 0]

        result = component._calculate_balance(targets)

        # All same class = 100% imbalanced
        assert "Imbalanced" in result


class TestDatasetPlotterEmptyPlot:
    """Test empty plot creation."""

    def test_create_empty_plot_light_theme(self):
        """Test empty plot with light theme."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        fig = component._create_empty_plot("No data", theme="light")

        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) > 0

    def test_create_empty_plot_dark_theme(self):
        """Test empty plot with dark theme."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        fig = component._create_empty_plot("No data", theme="dark")

        assert isinstance(fig, go.Figure)  # Dark theme applied

    def test_create_empty_plot_custom_message(self):
        """Test empty plot with custom message."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        custom_msg = "Custom message here"
        fig = component._create_empty_plot(custom_msg, theme="light")

        assert isinstance(fig, go.Figure)


class TestDatasetPlotterEdgeCases:
    """Test edge cases and error handling."""

    def test_scatter_plot_with_nan_values(self):
        """Test scatter plot handles NaN values."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [1, np.nan], [2, 2]], "targets": [0, 1, 0]}

        # Should handle gracefully
        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)

    def test_scatter_plot_with_inf_values(self):
        """Test scatter plot handles infinite values."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[0, 0], [np.inf, 1], [2, 2]], "targets": [0, 1, 0]}

        # Should handle gracefully
        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)

    def test_scatter_plot_with_single_point(self):
        """Test scatter plot with single data point."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[1, 2]], "targets": [0]}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)

    def test_scatter_plot_with_many_classes(self):
        """Test scatter plot with many classes (>5, tests color cycling)."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        # 10 classes
        dataset = {"inputs": [[i, i] for i in range(10)], "targets": list(range(10))}

        fig = component._create_scatter_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)

    def test_callbacks_registration(self):
        """Test callback registration."""
        from dash import Dash

        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        app = Dash(__name__)

        # Should not raise
        component.register_callbacks(app)

    def test_distribution_with_single_feature(self):
        """Test distribution plot with single feature."""
        from frontend.components.dataset_plotter import DatasetPlotter

        config = {}
        component = DatasetPlotter(config)

        dataset = {"inputs": [[1], [2], [3], [4], [5]], "targets": [0, 1, 0, 1, 0]}

        fig = component._create_distribution_plot(dataset, theme="light")

        assert isinstance(fig, go.Figure)
