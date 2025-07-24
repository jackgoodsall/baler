#!/usr/bin/env bash
set -euo pipefail

poetry run baler --project ATLAS hepmc --mode train
poetry run baler --project ATLAS hepmc --mode compress
poetry run baler --project ATLAS hepmc --mode decompress
poetry run baler --project ATLAS hepmc --mode plot