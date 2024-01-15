#!/bin/bash

for i in {1..60} #{1..k} k*1000=how many images are rendered
do
  blenderproc run playground.py render_cfg.yaml
  echo "$i"
done
