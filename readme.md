1.  build and activate conda env
```
conda env create -f env.yml
conda activate render_tracebot
```

2. set paths in render_cfg.yaml
3. run render.py
```
blenderproc run render.py render_cfg.yaml
```