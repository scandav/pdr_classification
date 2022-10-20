#!/bin/bash

salloc --gres=gpu:rtx3090:1 --account=ws_00000 --cpus-per-task=4 --mem-per-cpu=4G --nodes=1 --ntasks=1 --ntasks-per-node=1 --time=2:00:00 --partition=gpu-invest --job-name=debugg
