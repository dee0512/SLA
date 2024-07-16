#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-1080ti
#SBATCH --gres=gpu:1
#SBATCH --time=01-07:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

python test_sla.py
exit
