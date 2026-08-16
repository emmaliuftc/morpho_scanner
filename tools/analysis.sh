#!/bin/bash
# Benchmarking and Quality Analysis Script

PYTHON_ENV=".venv_nerf/bin/python"

if [ "$1" == "test1" ]; then
    echo "Running Test 1: Volume Comparison"
    PLY1="$2"
    PLY2="$3"
    
    if [ ! -f "$PLY1" ] || [ ! -f "$PLY2" ]; then
        echo "Error: Must provide two valid .ply files for comparison."
        echo "Usage: ./tools/analysis.sh test1 <ply1> <ply2>"
        exit 1
    fi
    
    $PYTHON_ENV tools/calc_volume.py "$PLY1" "$PLY2"
elif [ "$1" == "test2" ]; then
    echo "Running Test 2: Z-Level Volume Loss Analysis"
    PLY1="$2"
    PLY2="$3"
    
    if [ ! -f "$PLY1" ] || [ ! -f "$PLY2" ]; then
        echo "Error: Must provide two valid .ply files for analysis."
        echo "Usage: ./tools/analysis.sh test2 <ply1> <ply2>"
        exit 1
    fi
    
    $PYTHON_ENV tools/calc_z_area.py "$PLY1"
    echo ""
    $PYTHON_ENV tools/calc_z_area.py "$PLY2"
else
    echo "Usage:"
    echo "  ./tools/analysis.sh test1 <ply1> <ply2>"
    echo "  ./tools/analysis.sh test2 <ply1> <ply2>"
fi
