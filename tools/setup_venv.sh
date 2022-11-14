#!/bin/bash
# ensure the cwd is consistent
SCRIPT_PATH=${0%/*}
echo "${SCRIPT_PATH}"
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then 
    cd $SCRIPT_PATH
fi

cd ..

python -m venv venv

# activate the virtual environment
source ./venv/bin/activate

# install requirements from file
pip install -r requirements.txt
