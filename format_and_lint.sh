#!/bin/bash

poetry run black .
poetry run ruff . --fix
