#!/usr/bin/env bash


#####################################################################################################################################################################################################
# Set up environment variables
#####################################################################################################################################################################################################
CONTENT="\
# Juniper Canopy Environment Variables\n\
CASCOR_ENV=development\n\
CASCOR_DEBUG=\"${TRUE}\"\n\
CASCOR_LOG_LEVEL=\"${DEBUG_LEVEL}\"\n\
CASCOR_CONSOLE_LOG_LEVEL=INFO\n\
CASCOR_FILE_LOG_LEVEL=DEBUG\n\
CASCOR_CONFIG_PATH=conf/app_config.yaml\n\n\
# Performance settings\n\
OMP_NUM_THREADS=4\n\
MKL_NUM_THREADS=4\n\
NUMBA_NUM_THREADS=4\n\n\
# Application settings\n\
CASCOR_HOST=127.0.0.1\n\
CASCOR_PORT=8050\n\
EOF"

echo -ne "${CONTENT}" >bla.env


JUNK="
Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10
"

echo "${JUNK}" >blabla.txt
