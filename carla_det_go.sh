#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_DET GO"
echo "-----"

MODE="CARLA_DET"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_DET GO DONE"
echo "----------"

