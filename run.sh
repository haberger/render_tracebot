#!/bin/sh
for i in {1..1}
do
  blenderproc run render_base.py render_cfg.yaml
done
python proc_to_bop_pyrender