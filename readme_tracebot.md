1. install miniconda

[miniconda](https://docs.conda.io/en/latest/miniconda.html#)

1.  build and activate conda env
```
conda env create -f env.yml
conda activate render_tracebot
```

For physics rendering

3. setup render_cfg.yaml

4. if models_info.json does not exist call
```
python create_bop_models_info_json.py render_cfg.yaml 
```

5. start rendering

 - Only rgb and depth
```
blenderproc run render_base.py render_cfg.yaml
```

- full bop_pipeline
```
./run.sh
```
For bop template rendering:

3. setup temp_cfg.yaml

4. run script
```
blenderproc run bop_templates.py temp_cfg.yaml
```






1: "MediumRinseFluidK_Bottle.stl"
2: "SmallSoyBroth_Bottle.stl"
3: "tracebot_needle_without_cap.stl"
4: "tracebot_cap.stl"
5: "RedCap.stl"
6: "FluidContainer.stl"
7: "LargeRinseFluidA_Bottle.stl"
8: "yellow cap"
10: "white_clamp.stl"
11: "red_clamp.stl"


docker run --rm -it --runtime=nvidia --gpus all --mount type=bind,source=/media/david/Sasha_external/David/tracebot,target=/workspace  implicit_depth:latest