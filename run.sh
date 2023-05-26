#!/bin/bash

REPO_URL_script=https://github.com/haberger/render_tracebot.git
REPO_NAME_script="render_tracebot"
REPO_URL_proc=https://github.com/haberger/Bproc_tracebot.git
REPO_NAME_Bproc_tracebot="render_tracebot"
CLONE_DIR="/workspace"

cd $CLONE_DIR

# Check if the folder already exists
if [ -d "$CLONE_DIR/$REPO_NAME_script" ]; then
  echo "Folder already exists. Skipping clone."
else
  # Clone the repository
  git clone "$REPO_URL_script" "$CLONE_DIR/$REPO_NAME_script"
fi

if [ -d "$CLONE_DIR/$REPO_NAME_Bproc_tracebot" ]; then
  echo "Folder already exists. Skipping clone."
else
  # Clone the repository
  git clone "$REPO_URL_proc" "$CLONE_DIR/$REPO_NAME_Bproc_tracebot"
  cd "$CLONE_DIR/$REPO_NAME_Bproc_tracebot"
  pip install -e "$CLONE_DIR/$REPO_NAME_Bproc_tracebot"
fi

for i in {1..50} #{1..k} k*1000=how many images are rendered
do
  blenderproc run playground.py render_cfg.yaml
  echo "$i"
done