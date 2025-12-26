#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     run_all_tests.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2025-10-11
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#
#####################################################################################################################################################################################################
# Notes:
#
########################################################################################################)#############################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Initialize script by sourcing the init_conf.bash config file
#####################################################################################################################################################################################################
set -o functrace

# shellcheck disable=SC2155
export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")" && INIT_CONF="conf/init.conf"
# shellcheck disable=SC2015,SC1091 source=conf/init.conf
[[ -f "${INIT_CONF}" ]] && source "${INIT_CONF}" || { echo "Init Config File Not Found. Unable to Continue."; exit 1; }


#####################################################################################################################################################################################################
#
#####################################################################################################################################################################################################
# trunk-ignore(shellcheck/SC2015)
# trunk-ignore(shellcheck/SC2312)
# shellcheck disable=SC2015
[[ "$(uname)" == "${OS_NAME_LINUX}" ]]  && export HOME_DIR="/home/${USERNAME}" || { [[ "$(uname)" == "${OS_NAME_MACOS}"  ]] && export HOME_DIR="/Users/${USERNAME}" || { echo "Error: Invalid OS Type! Exiting..."  && set -e && exit 1; }; }

cd "${PROJ_DIR}"

if [[ "${COVERAGE_REPORT}" == "${FALSE}" ]]; then
    echo "pytest -v src/tests"
    pytest -v src/tests
elif [[ "${COVERAGE_REPORT}" == "${TRUE}" ]]; then
    echo -ne " \
    pytest -v ./src/tests \n \
        --cov=src \n \
        --cov-report=xml:src/tests/reports/coverage.xml \n \
        --cov-report=term-missing \n \
        --cov-report=html:src/tests/reports/coverage \n \
        --junit-xml=src/tests/reports/junit/results.xml \n \
        --continue-on-collection-errors \n \
        \n"

    pytest -v ./src/tests \
        --cov=src \
        --cov-report=xml:src/tests/reports/coverage.xml \
        --cov-report=term-missing \
        --cov-report=html:src/tests/reports/coverage \
        --junit-xml=src/tests/reports/junit/results.xml \
        --continue-on-collection-errors \

else
    echo "Coverage Report flag has an Invalid Value"
    exit 1
fi

echo "Running the Juniper Canopy project's Full Test Suite $( [[ "$?" == "${TRUE}" ]] && echo "Succeeded!" || echo "Failed." )"
exit 0
