#!/bin/bash

set -e

ENV_NAME="py-env"

python3 -m venv "${ENV_NAME}"

source "${ENV_NAME}/bin/activate"

echo "Upgrading pip"
python3 -m pip install --upgrade pip

echo "Upgrading setuptools"
python3 -m pip install --upgrade setuptools

echo "Install packages from requirements.txt"
python3 -m pip install -r requirements.txt

deactivate
