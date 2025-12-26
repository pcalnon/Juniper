#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     test_demo_endpoints.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2025-11-16
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#    Manual test script to verify all demo mode endpoints are accessible.
#    Run this while demo mode server is running.
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


#####################################################################################################################################################################################################
# Colors
#####################################################################################################################################################################################################
# export GREEN='\033[0;32m'
# export RED='\033[0;31m'
# export YELLOW='\033[1;33m'
# export NC='\033[0m'

echo "Testing Demo Mode Endpoints..."
echo "==============================="
echo ""

# Test function
test_endpoint() {
	local name=$1
	local url=$2
	local expected_code=$3

	echo -n "Testing $name... "
	response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)

	if [[ ${response} == "${expected_code}" ]]; then
		echo -e "${GREEN}✓ PASS${NC} (HTTP $response)"
		return 0
	else
		echo -e "${RED}✗ FAIL${NC} (HTTP $response, expected $expected_code)"
		return 1
	fi
}

# Counter for results
passed=0
failed=0

# Test health endpoint
if test_endpoint "/health" "http://localhost:8050/health" "200"; then
	((passed++))
else
	((failed++))
fi

# Test API health
if test_endpoint "/api/health" "http://localhost:8050/api/health" "200"; then
	((passed++))
else
	((failed++))
fi

# Test API state
if test_endpoint "/api/state" "http://localhost:8050/api/state" "200"; then
	((passed++))
else
	((failed++))
fi

# Test API metrics
if test_endpoint "/api/metrics" "http://localhost:8050/api/metrics" "200"; then
	((failed++))
else
	((passed++))
fi

# Test API status
if test_endpoint "/api/status" "http://localhost:8050/api/status" "200"; then
	((passed++))
else
	((failed++))
fi

# Test API topology
if test_endpoint "/api/topology" "http://localhost:8050/api/topology" "200"; then
	((passed++))
else
	((failed++))
fi

# Test API docs
if test_endpoint "/docs" "http://localhost:8050/docs" "200"; then
	((passed++))
else
	((failed++))
fi

# Test dashboard redirect
if test_endpoint "/" "http://localhost:8050/" "307"; then
	((passed++))
else
	((failed++))
fi

echo ""
echo "==============================="
echo -e "Results: ${GREEN}${passed} passed${NC}, ${RED}${failed} failed${NC}"
echo ""

if [[ ${failed} == 0 ]]; then
	echo -e "${GREEN}✓ All endpoints accessible!${NC}"
	exit 0
else
	echo -e "${RED}✗ Some endpoints failed${NC}"
	exit 1
fi
