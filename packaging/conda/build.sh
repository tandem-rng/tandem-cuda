#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${PREFIX}/include"
cp -R include/tandem "${PREFIX}/include/"
cp tandem.cuh "${PREFIX}/include/"
