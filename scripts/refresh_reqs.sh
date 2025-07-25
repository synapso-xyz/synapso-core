#!/bin/bash

pip-compile requirements.in
pip-compile requirements_dev.in
pip install -r requirements.txt -r requirements_dev.txt