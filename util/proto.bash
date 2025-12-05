#!/usr/bin/env bash
########################################################################################################################################################################################
# Application: Juniper
# Script Name: try.bash
# Script Path: <Project>/util/try.bash
#
# Description: This script performs the following actions for the current Project:
#
#                 1.  Applies the cargo linter to the source files
#                 2.  Builds the current project with the debug target
#                 3.  Sets the expected Environment Variables for the Application
#                 4.  Adds the expected command line arguments
#                 5.  Executes the project's binary
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Notes:
#     
#
#
#######################################################################################################################################################################################

#####################################################################################################
# Specify the Python script to run:
####################################################################################################
#PYTHON_SCRIPT_NAME="binary_classifier.py"
#PYTHON_SCRIPT_NAME="mnist_classifier.py"
#PYTHON_SCRIPT_NAME="mnist_dynamic_classifier.py"
#PYTHON_SCRIPT_NAME="dynamic_classifier.py"
#PYTHON_SCRIPT_NAME="full_dynamic_classifier.py"
#PYTHON_SCRIPT_NAME="static_nn.py"
#PYTHON_SCRIPT_NAME="dynamic_layer_nn-00.py"
#PYTHON_SCRIPT_NAME="dynamic_layer_nn-01.py"
#PYTHON_SCRIPT_NAME="dynamic_layer_nn-02.py"
#PYTHON_SCRIPT_NAME="dynamic_nodes_nn-00.py"


PARAMS="$@"

# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/auto_grad_test.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-0.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-1.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-2.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-3.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-3a.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-4.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-4a.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/claude_sonnet_3.7_0.py"
SCRIPT="claude_sonnet_3.7_0.py"


#export PYTHON_SCRIPT_NAME="$(filename "${SCRIPT}")"
export PYTHON_SCRIPT_NAME="$(basename "${SCRIPT}")"
#export PYTHON_SCRIPT_DIR="$(dirname "${SCRIPT}")"
export PYTHON_SCRIPT_DIR="prototypes"

export PYTHON_SCRIPT_PATH="${PYTHON_SCRIPT_DIR}/${PYTHON_SCRIPT_NAME}"


#####################################################################################################
# Define Global Configuration File Constants
####################################################################################################
# export ROOT_PROJ_NAME="dynamic_nn"
# export ROOT_PROJ_NAME="juniper"
export ROOT_PROJ_NAME="Juniper"
export ROOT_CONF_NAME="conf"
export ROOT_CONF_FILE_NAME="script_util.cfg"
export ROOT_PROJ_DIR="${HOME}/Development/python/${ROOT_PROJ_NAME}"
export ROOT_CONF_DIR="${ROOT_PROJ_DIR}/${ROOT_CONF_NAME}"
export ROOT_CONF_FILE="${ROOT_CONF_DIR}/${ROOT_CONF_FILE_NAME}"
source "${ROOT_CONF_FILE}"


####################################################################################################
# Configure Script Environment
####################################################################################################
# Determine Project Dir
export BASE_DIR=$(${GET_PROJECT_SCRIPT} "${BASH_SOURCE}")
# Determine Host OS
export CURRENT_OS=$(${GET_OS_SCRIPT})
# Define Script Functions
source "${DATE_FUNCTIONS_SCRIPT}"


####################################################################################################
# Define Script Constants
####################################################################################################
export DATA_DIR="${BASE_DIR}/${DATA_DIR_NAME}"
export SOURCE_DIR="${BASE_DIR}/${SOURCE_DIR_NAME}"
export CONFIG_DIR="${BASE_DIR}/${CONFIG_DIR_NAME}"
export LOGGING_DIR="${BASE_DIR}/${LOGGING_DIR_NAME}"
export UTILITY_DIR="${BASE_DIR}/${UTILITY_DIR_NAME}"

export PYTHON="$(which python3)"

export PYTHON_SCRIPT="${SOURCE_DIR}/${PYTHON_SCRIPT_PATH}"


####################################################################################################
# Update the Python Path for the script
####################################################################################################
PATH_DEL=":"
#PATH_FOUND="$(grep "${SOURCE_DIR}" "${PYTHON_PATH}")"
PATH_FOUND="$(echo "${PYTHONPATH}" | grep "${SOURCE_DIR}")"
if [[ "${PATH_FOUND}" == "" ]]; then
    [[ ( "${PYTHON_PATH}" == "" ) || ( "${PYTHONPATH: -1}" == "${PATH_DEL}" ) ]] && PATH_DEL=""
fi
export PYTHONPATH="${PYTHON_PATH}${PATH_DEL}${SOURCE_DIR}"


####################################################################################################
# Display Environment Values
####################################################################################################
echo "Base Dir: ${BASE_DIR}"
echo "Current OS: ${CURRENT_OS}"
echo "Python: ${PYTHON} (ver: $(${PYTHON} --version))"
echo "Python Path: ${PYTHON_PATH}"
echo "Python Script: ${PYTHON_SCRIPT}"
echo " "


####################################################################################################
# Execute Python script
####################################################################################################
#echo "time ${PYTHON} ${PYTHON_SCRIPT} >./output.log"
echo "time ${PYTHON} ${PYTHON_SCRIPT} ${PARAMS}"

#time ${PYTHON} ${PYTHON_SCRIPT} >./output.log
time ${PYTHON} ${PYTHON_SCRIPT} ${PARAMS}
