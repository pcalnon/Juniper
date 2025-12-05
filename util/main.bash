#!/usr/bin/env bash

PROJECT_NAME="Juniper"

WORKING_DIR="${HOME}/Development/python/${PROJECT_NAME}"

SOURCE_DIR="${WORKING_DIR}/src"
PYTHON_FILE_NAME="main.py"

PYTHON_FILE="${SOURCE_DIR}/${PYTHON_FILE_NAME}"

python3 --version
python3 ${PYTHON_FILE}
