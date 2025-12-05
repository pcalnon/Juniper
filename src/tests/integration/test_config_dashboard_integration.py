#!/usr/bin/env python
#####################################################################
# Project:       Juniper
# Prototype:     Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
# File Name:     test_config_dashboard_integration.py
# Author:        Paul Calnon
# Version:       0.1.0
# Date:          2025-11-17
# Last Modified: 2025-11-17
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
# Description:   Integration tests for configuration and dashboard
#####################################################################

import pytest

from config_manager import get_config
from constants import TrainingConstants


@pytest.mark.integration
class TestConfigDashboardIntegration:
    """Test configuration integration with dashboard and constants."""

    def test_config_defaults_match_constants(self):
        """Test that configuration defaults match constants."""
        config_mgr = get_config(force_reload=True)
        defaults = config_mgr.get_training_defaults()

        # Defaults should match constants
        assert defaults["epochs"] == TrainingConstants.DEFAULT_TRAINING_EPOCHS
        assert defaults["learning_rate"] == TrainingConstants.DEFAULT_LEARNING_RATE
        assert defaults["hidden_units"] == TrainingConstants.DEFAULT_MAX_HIDDEN_UNITS

    def test_config_ranges_match_constants(self):
        """Test that configuration ranges match constants."""
        config_mgr = get_config(force_reload=True)

        # Epochs
        epochs_config = config_mgr.get_training_param_config("epochs")
        assert epochs_config["min"] == TrainingConstants.MIN_TRAINING_EPOCHS
        assert epochs_config["max"] == TrainingConstants.MAX_TRAINING_EPOCHS

        # Learning rate
        lr_config = config_mgr.get_training_param_config("learning_rate")
        assert lr_config["min"] == TrainingConstants.MIN_LEARNING_RATE
        assert lr_config["max"] == TrainingConstants.MAX_LEARNING_RATE

        # Hidden units
        hu_config = config_mgr.get_training_param_config("hidden_units")
        assert hu_config["min"] == TrainingConstants.MIN_HIDDEN_UNITS
        assert hu_config["max"] == TrainingConstants.MAX_HIDDEN_UNITS

    def test_config_consistency_check_passes(self):
        """Test that configuration consistency check passes."""
        config_mgr = get_config(force_reload=True)
        result = config_mgr.verify_config_constants_consistency()

        assert result is True, "Configuration should be consistent with constants"

    def test_param_validation_integration(self):
        """Test that parameter validation works with config and constants."""
        config_mgr = get_config(force_reload=True)

        # Valid values should pass
        assert config_mgr.validate_training_param_value("epochs", TrainingConstants.DEFAULT_TRAINING_EPOCHS)
        assert config_mgr.validate_training_param_value("learning_rate", TrainingConstants.DEFAULT_LEARNING_RATE)
        assert config_mgr.validate_training_param_value("hidden_units", TrainingConstants.DEFAULT_MAX_HIDDEN_UNITS)

        # Invalid values should fail
        with pytest.raises(ValueError):
            config_mgr.validate_training_param_value("epochs", TrainingConstants.MIN_TRAINING_EPOCHS - 1)

        with pytest.raises(ValueError):
            config_mgr.validate_training_param_value("epochs", TrainingConstants.MAX_TRAINING_EPOCHS + 1)
