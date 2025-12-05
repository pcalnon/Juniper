#!/usr/bin/env python
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
# File Name:     training_state_machine.py
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date:          2025-11-16
# Last Modified: 2025-12-03
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    Formal finite state machine for training control state management.
#    Ensures deterministic state transitions and prevents invalid operations.
#
#####################################################################################################################################################################################################
# Notes:
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
# import json
import logging
from enum import Enum, auto
from typing import Optional


class TrainingPhase(Enum):
    """Training phase enumeration."""

    IDLE = auto()
    OUTPUT = auto()
    CANDIDATE = auto()
    INFERENCE = auto()


class TrainingStatus(Enum):
    """Training status enumeration."""

    STOPPED = auto()
    STARTED = auto()
    PAUSED = auto()


class Command(Enum):
    """Control command enumeration."""

    START = auto()
    STOP = auto()
    PAUSE = auto()
    RESUME = auto()
    RESET = auto()


class TrainingStateMachine:
    """
    Formal finite state machine for training control.

    States:
    - Stopped: Training is not active
    - Started: Training is active (with sub-states: Output, Candidate, Inference)
    - Paused: Training is paused (remembers previous sub-state)

    Transitions:
    - Stopped → Started (on start command)
    - Started → Paused (on pause command, remember sub-state)
    - Paused → Started (on resume or start command, restore sub-state)
    - Started → Stopped (on stop command)
    - Any → Stopped (on reset command)
    """

    def __init__(self):
        """Initialize state machine in Stopped state."""
        self.logger = logging.getLogger(__name__)
        self.__status = TrainingStatus.STOPPED
        self.__phase = TrainingPhase.IDLE
        self.__paused_phase: Optional[TrainingPhase] = None

        # Candidate phase-specific state
        self.__candidate_sub_state: Optional[dict] = None

    def get_status(self) -> TrainingStatus:
        """Get current training status."""
        return self.__status

    def get_phase(self) -> TrainingPhase:
        """Get current training phase."""
        return self.__phase

    def get_paused_phase(self) -> Optional[TrainingPhase]:
        """Get phase that was active when paused."""
        return self.__paused_phase

    def is_stopped(self) -> bool:
        """Check if in Stopped state."""
        return self.__status == TrainingStatus.STOPPED

    def is_started(self) -> bool:
        """Check if in Started state."""
        return self.__status == TrainingStatus.STARTED

    def is_paused(self) -> bool:
        """Check if in Paused state."""
        return self.__status == TrainingStatus.PAUSED

    def handle_command(self, command: Command) -> bool:
        """
        Handle control command and perform state transition.

        Args:
            command: Control command to execute

        Returns:
            True if transition successful, False if invalid
        """
        if command == Command.START:
            return self._handle_start()
        elif command == Command.STOP:
            return self._handle_stop()
        elif command == Command.PAUSE:
            return self._handle_pause()
        elif command == Command.RESUME:
            return self._handle_resume()
        elif command == Command.RESET:
            return self._handle_reset()
        else:
            self.logger.error(f"Unknown command: {command}")
            return False

    def _handle_start(self) -> bool:
        """Handle START command."""
        if self.__status == TrainingStatus.STOPPED:
            return self._stop_to_start_transition()
        elif self.__status == TrainingStatus.PAUSED:
            return self._check_for_paused_state("State transition: Paused → Started (")
        elif self.__status == TrainingStatus.STARTED:
            # Already started, ignore
            self.logger.warning("Invalid transition: START while already Started")
            return False

        return False

    def _stop_to_start_transition(self):
        # Stopped → Started: begin fresh training
        self.__status = TrainingStatus.STARTED
        self.__phase = TrainingPhase.OUTPUT
        self.__paused_phase = None
        self.__candidate_sub_state = None
        self.logger.info("State transition: Stopped → Started (Output)")
        return True

    def _handle_stop(self) -> bool:
        """Handle STOP command."""
        if self.__status in (TrainingStatus.STARTED, TrainingStatus.PAUSED):
            return self._start_pause_to_stop_transition()
        elif self.__status == TrainingStatus.STOPPED:
            # Already stopped, ignore
            self.logger.warning("Invalid transition: STOP while already Stopped")
            return False
        return False

    def _start_pause_to_stop_transition(self):
        return self._update_status_and_output(" → Stopped")

    def _handle_pause(self) -> bool:
        """Handle PAUSE command."""
        if self.__status == TrainingStatus.STARTED:
            # Started → Paused: save current phase
            self.__status = TrainingStatus.PAUSED
            self.__paused_phase = self.__phase
            self.logger.info(f"State transition: Started → Paused (saved phase: {self.__phase.name})")
            return True

        elif self.__status == TrainingStatus.PAUSED:
            # Already paused, ignore
            self.logger.warning("Invalid transition: PAUSE while already Paused")
            return False

        elif self.__status == TrainingStatus.STOPPED:
            # Cannot pause when stopped
            self.logger.warning("Invalid transition: PAUSE while Stopped")
            return False
        return False

    def _handle_resume(self) -> bool:
        """Handle RESUME command."""
        if self.__status == TrainingStatus.PAUSED:
            return self._check_for_paused_state("State transition: Paused → Started (restored phase: ")
        elif self.__status == TrainingStatus.STARTED:
            # Already started, ignore
            self.logger.warning("Invalid transition: RESUME while already Started")
            return False

        elif self.__status == TrainingStatus.STOPPED:
            # Cannot resume when stopped
            self.logger.warning("Invalid transition: RESUME while Stopped")
            return False

        return False

    def _check_for_paused_state(self, arg0):
        self.__status = TrainingStatus.STARTED
        if self.__paused_phase:
            self.__phase = self.__paused_phase
            self.logger.info(f"{arg0}{self.__phase.name})")
        else:
            self.__phase = TrainingPhase.OUTPUT
            self.logger.info("State transition: Paused → Started (Output, no saved phase)")
        self.__paused_phase = None
        return True

    def _handle_reset(self) -> bool:
        """Handle RESET command."""
        return self._update_status_and_output(" → Stopped (RESET)")

    def _update_status_and_output(self, arg0):
        prev_status = self.__status.name
        self.__status = TrainingStatus.STOPPED
        self.__phase = TrainingPhase.IDLE
        self.__paused_phase = None
        self.__candidate_sub_state = None
        self.logger.info(f"State transition: {prev_status}{arg0}")
        return True

    def set_phase(self, phase: TrainingPhase) -> None:
        """
        Set current training phase (only when Started).

        Args:
            phase: New training phase
        """
        if self.__status != TrainingStatus.STARTED:
            self.logger.warning(f"Cannot set phase to {phase.name} while status is {self.__status.name}")
            return

        prev_phase = self.__phase
        self.__phase = phase
        self.logger.debug(f"Phase change: {prev_phase.name} → {phase.name}")

    def save_candidate_state(self, state: dict) -> None:
        """
        Save candidate phase sub-state for resume.

        Args:
            state: Candidate phase state dictionary
        """
        self.__candidate_sub_state = state.copy()
        self.logger.debug(f"Saved candidate sub-state: {state}")

    def get_candidate_state(self) -> Optional[dict]:
        """Get saved candidate phase sub-state."""
        return self.__candidate_sub_state

    def get_state_summary(self) -> dict:
        """
        Get current state as dictionary.

        Returns:
            Dictionary with status, phase, and paused_phase
        """
        return {
            "status": self.__status.name,
            "phase": self.__phase.name,
            "paused_phase": self.__paused_phase.name if self.__paused_phase else None,
            "has_candidate_state": self.__candidate_sub_state is not None,
        }
