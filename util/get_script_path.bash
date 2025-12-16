#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Script Name:   get_script_path.bash
# Script Path:   <Project>/<Sub-Project>/juniper_canopy/util/get_script_path.bash
# Conf File:     get_script_path.conf
# Conf Path:     <Project>/<Sub-Project>/<Application>/conf/  # TODO: Add parent project dir
#
# Author:        Paul Calnon
# Version:       1.0.0
#
# Date:          2025-08-01
# Last Modified: 2025-12-15
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     This script finds the actual path of source files in the Juniper python project code base
#
#####################################################################################################################################################################################################
# Notes:
#
#####################################################################################################################################################################################################
# References:
#
#     SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
#     CONF_PATH="$(dirname "$(dirname "${SCRIPT_PATH}")")/conf"
#     CONF_FILENAME="$(basename -s ".bash" "${SCRIPT_PATH}").conf"
#     CONF_FILE="${SCRIPT_PATH}/${SCRIPT_FILENAME}"
#
#     CONF_PATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/conf"
#     CONF_FILENAME="$(basename -s ".bash" "$(realpath "${BASH_SOURCE[0]}")").conf"
#     CONF_FILE="${CONF_PATH}/${CONF_FILENAME}"
#
#     CONF_FILE="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/conf/$(basename -s ".bash" "$(realpath "${BASH_SOURCE[0]}")").conf"
#
#     source "${CONF_FILE}";  SUCCESS="$?"
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define function for the script
#####################################################################################################################################################################################################
get_script_path() {
    local source="${BASH_SOURCE[0]}"
    while [ -L "$source" ]; do
        local dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$(cd -P "$(dirname "$source")" && pwd)/$(basename "$source")"
}


#####################################################################################################################################################################################################
# Get the path and return it
#####################################################################################################################################################################################################
SCRIPT_PATH="$(get_script_path)"
echo "${SCRIPT_PATH}"
