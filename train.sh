#!/bin/sh
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg experiments/Tu_indoor/ddrnet23_slim.yaml
