#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
# File Name:     base_component.py
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
#
# Date:          2025-10-11
# Last Modified: 2025-12-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This module provides abstract base classes for frontend components.
#
#####################################################################################################################################################################################################
# Notes:
#
#     Base Component Classes
#     Provides abstract base classes for frontend components following a common interface.
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################
import logging
from abc import ABC, abstractmethod

# from typing import Dict, Any, Optional
from typing import Any, Dict


class BaseComponent(ABC):
    """
    Abstract base class for all dashboard components.
    Provides common interface and functionality for all visualization components.
    """

    def __init__(self, config: Dict[str, Any], component_id: str):
        """
        Initialize base component.
        Args:
            config: Component configuration dictionary
            component_id: Unique identifier for this component
        """
        self.config = config
        self.component_id = component_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_initialized = False

    @abstractmethod
    def get_layout(self) -> Any:
        """
        Get Dash layout for this component.

        Returns:
            Dash component layout
        """
        pass

    @abstractmethod
    def register_callbacks(self, app):
        """
        Register Dash callbacks for this component.

        Args:
            app: Dash application instance
        """
        pass

    def initialize(self):
        """Initialize component (called once)."""
        if not self.is_initialized:
            self.logger.info(f"Initializing component: {self.component_id}")
            self.is_initialized = True

    def cleanup(self):
        """Clean up component resources."""
        self.logger.info(f"Cleaning up component: {self.component_id}")

    def get_component_id(self) -> str:
        """Get component identifier."""
        return self.component_id

    def update_config(self, config: Dict[str, Any]):
        """
        Update component configuration.

        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        self.logger.debug(f"Configuration updated for {self.component_id}")
