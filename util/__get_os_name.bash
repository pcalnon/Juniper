#!/bin/bash
#####################################################################################################################################################################################################
# Application: Juniper
# Script Name: __get_os_name.bash
# Script Path: <Project>/util/__get_os_name.bash
#
# Description: This script returns the name of the OS on the current host
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Notes:
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Examples:
#
#####################################################################################################################################################################################################


#####################################################################################################
# Define GLobal Debug Constants
#####################################################################################################
# TRUE="true"
export TRUE="0"
# FALSE="false"
export FALSE="1"

# export DEBUG="${TRUE}"
export DEBUG="${FALSE}"


#####################################################################################################
# Define Global Functions
####################################################################################################
# Define local Functions
get_script_path() {
    local source="${BASH_SOURCE[0]}"
    while [ -L "$source" ]; do
        # shellcheck disable=SC2155
        local dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ $source != /* ]] && source="$dir/$source"
    done
    echo "$(cd -P "$(dirname "$source")" && pwd)/$(basename "$source")"
}


####################################################################################################
# Define Global Environment DirectoryConfiguration Constants
# ROOT_PROJ_NAME="JuniperCanopy"
# PROJ_NAME="juniper_canopy"
# DEV_DIR="Development"
# LANGUAGE_NAME="python"
# ROOT_PROJ_DIR="${HOME}/${DEV_DIR}/${LANGUAGE_NAME}/${ROOT_PROJ_NAME}/${PROJ_NAME}"
# ROOT_CONF_DIR="${ROOT_PROJ_DIR}/${ROOT_CONF_NAME}"
# ROOT_CONF_FILE="${ROOT_CONF_DIR}/${ROOT_CONF_FILE_NAME}"
# source ${ROOT_CONF_FILE}

# shellcheck disable=SC2155
export SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
# shellcheck disable=SC2155
export SCRIPT_PATH="$(dirname "$(get_script_path)")"
# shellcheck disable=SC2155
export SCRIPT_PROJ_PATH="$(dirname "${SCRIPT_PATH}")"
# shellcheck disable=SC2155
export ROOT_PROJ_DIR_NAME="$(basename "${SCRIPT_PROJ_PATH}")"
# shellcheck disable=SC2155
export SCRIPT_LANG_PATH="$(dirname "${SCRIPT_PROJ_PATH}")"
# shellcheck disable=SC2155
export ROOT_LANG_DIR_NAME="$(basename "${SCRIPT_LANG_PATH}")"
# shellcheck disable=SC2155
export SCRIPT_DEVELOPMENT_PATH="$(dirname "${SCRIPT_LANG_PATH}")"
# shellcheck disable=SC2155
export ROOT_DEV_DIR_NAME="$(basename "${SCRIPT_DEVELOPMENT_PATH}")"
# shellcheck disable=SC2155
export SCRIPT_ROOT_PATH="$(dirname "${SCRIPT_DEVELOPMENT_PATH}")"

export ROOT_PROJ_NAME="${ROOT_PROJ_DIR_NAME}"
export ROOT_CONF_NAME="conf"
export ROOT_CONF_FILE_NAME="common.${ROOT_CONF_NAME}"

export ROOT_PROJ_DIR="${SCRIPT_PROJ_PATH}"
export ROOT_CONF_DIR="${ROOT_PROJ_DIR}/${ROOT_CONF_NAME}"
export ROOT_CONF_FILE="${ROOT_CONF_DIR}/${ROOT_CONF_FILE_NAME}"

# NOTE: Do NOT source common.conf here - this script is called as a subprocess
# and sourcing common.conf without PARENT_PATH_PARAM causes issues.
# This is a simple utility script that only needs to return the OS name.

##################################################################################
# Determine Host OS
##################################################################################
# shellcheck disable=SC2155
if [[ -f /etc/os-release ]]; then
    export CURRENT_OS="$(grep -e "^NAME=" /etc/os-release | awk -F "\"" '{print $2;}')"
elif [[ "$(uname)" == "Darwin" ]]; then
    export CURRENT_OS="MacOS"
else
    export CURRENT_OS="$(uname -s)"
fi
echo "${CURRENT_OS}"
