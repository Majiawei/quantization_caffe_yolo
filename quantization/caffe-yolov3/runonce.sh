#!/bin/bash
cd build
rm -rf *
cmake ..
make -j52
# cd x86_64/bin
# ./detectnet
