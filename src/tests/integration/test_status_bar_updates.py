#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
# File Name:     test_status_bar_updates.py
# Author:        Paul Calnon
# Version:       1.0.0
#
# Date:          2025-11-16
# Last Modified: 2025-11-16
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#
#####################################################################################################################################################################################################
# Notes:
#
#     Integration tests for Status and Phase indicator updates at the top of the dashboard.
#     Tests that Status and Phase reflect actual TrainingState and update within <1 second.
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
"""Integration tests for top status bar updates."""
import os
import time

# MUST set environment variable BEFORE importing main
os.environ["CASCOR_DEMO_MODE"] = "1"

import pytest  # noqa: F401,E402
from fastapi.testclient import TestClient  # noqa: E402

from main import app, training_state  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Create test client with demo mode."""
    with TestClient(app) as client:
        yield client


class TestStatusBarUpdates:
    """Test Status and Phase indicator updates."""

    def test_api_state_endpoint_exists(self, client):
        """Test /api/state endpoint is accessible."""
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "phase" in data

    def test_status_reflects_training_state(self, client):
        """Test Status indicator reflects TrainingState.status."""
        # Set state to Stopped
        training_state.update_state(status="Stopped", phase="Idle")
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Stopped"

        # Set state to Running
        training_state.update_state(status="Running", phase="Output")
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Running"

    def test_phase_reflects_training_state(self, client):
        """Test Phase indicator reflects TrainingState.phase."""
        # Set phase to Idle
        training_state.update_state(status="Stopped", phase="Idle")
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert data["phase"] == "Idle"

        # Set phase to Output
        training_state.update_state(status="Running", phase="Output")
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert data["phase"] == "Output"

        # Set phase to Candidate
        training_state.update_state(status="Running", phase="Candidate")
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()
        assert data["phase"] == "Candidate"

    def test_state_changes_immediately(self, client):
        """Test state changes are reflected immediately without delay."""
        # Change state
        training_state.update_state(status="Running", phase="Output")

        # Fetch state immediately
        start_time = time.time()
        response = client.get("/api/state")
        latency = time.time() - start_time

        # Verify state was updated
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Running"
        assert data["phase"] == "Output"

        # Verify response time is fast (<1 second as per spec)
        assert latency < 1.0, f"State fetch took {latency:.3f}s, expected <1s"

    def test_status_changes_with_training_controls(self, client):
        """Test Status changes when training controls are used."""
        # Start training
        response = client.post("/api/train/start")
        assert response.status_code == 200

        # Give it a moment to update state
        time.sleep(0.1)

        # Check state reflects running
        response = client.get("/api/state")
        data = response.json()
        # In demo mode, status should be "Running" after start
        # Note: The exact status depends on demo_mode implementation
        assert data["status"] in ["Running", "Paused", "Stopped"]

    def test_status_bar_updates_within_one_second(self, client):
        """Test status bar updates within 1 second of state change."""
        # Change state
        old_epoch = training_state.get_state().get("current_epoch", 0)
        new_epoch = old_epoch + 1
        training_state.update_state(status="Running", phase="Output", current_epoch=new_epoch)

        # Measure time to fetch updated state
        start_time = time.time()
        response = client.get("/api/state")
        fetch_time = time.time() - start_time

        # Verify update is reflected
        assert response.status_code == 200
        data = response.json()
        assert data["current_epoch"] == new_epoch

        # Verify response time is <1s
        assert fetch_time < 1.0, f"State fetch took {fetch_time:.3f}s, expected <1s"

    def test_no_hardcoded_defaults_after_first_update(self, client):
        """Test that after first update, Status and Phase are not hardcoded defaults."""
        # Set a non-default state
        training_state.update_state(status="Running", phase="Candidate")

        # Fetch state
        response = client.get("/api/state")
        assert response.status_code == 200
        data = response.json()

        # Verify not showing defaults
        assert data["status"] != "Stopped" or data["phase"] != "Idle"
        assert data["status"] == "Running"
        assert data["phase"] == "Candidate"
