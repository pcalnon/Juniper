#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     source_tree.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2025-10-11
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     This script display the contents of the source, config, and log directories for a project in Tree format
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
# Source script config file
#####################################################################################################################################################################################################
set -o functrace
# shellcheck disable=SC2155
export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source "conf/init.conf"; SUCCESS="$?"

# shellcheck disable=SC1091
[[ "${SUCCESS}" != "0" ]] && { source "conf/config_fail.conf"; log_error "${SUCCESS}" "${PARENT_PATH_PARAM}" "conf/init.conf" "${LINENO}" "${LOG_FILE}"; }
log_debug "Successfully Configured Current Script: $(basename "${PARENT_PATH_PARAM}"), by Sourcing the Init Config File: ${INIT_CONF}, Returned: \"${SUCCESS}\""


##################################################################################
# Define Script Constants
##################################################################################
if [[ "${*}" == "" ]]; then
  echo "Warning: No Input Parameters Provided"
  #DIR_LIST="./conf ./data ./src ./util"
  DIR_LIST="${CONFIG_DIR_NAME} ${DATA_DIR_NAME} ${DOCUMENT_DIR_NAME} ${SOURCE_DIR_NAME} ${UTILITY_DIR_NAME}"
else
  DIR_LIST="${*}"
fi
echo "Dir list: ${DIR_LIST}"


##################################################################################
# Validate subdirectory list
##################################################################################
cd "${BASE_DIR}" || exit 1
WORKING_LIST=""
for DIR in ${DIR_LIST}; do
    if [[ -d ${DIR} ]]; then
        WORKING_LIST="${WORKING_LIST}${DIR} "
    fi
done


##################################################################################
# Print directory listing as Tree structure
##################################################################################
tree "${WORKING_LIST}"
