#!/bin/bash

set -e
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <baseline-name=directory-of-the-baseline>"
    exit 1
fi
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../$1

echo "Formatting started"
poetry run python -m isort .
poetry run python -m black -q .
poetry run python -m docformatter -i -r .
poetry run python -m ruff check --fix .
echo "Formatting done"
