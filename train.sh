#!/usr/bin/env bash
#
#SBATCH --partition=gypsum-1080ti
#SBATCH --gres=gpu:1
#SBATCH --time=01-07:00:00
#SBATCH --mem=16000
#SBATCH --output=outputs/output_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
env_name=${2:-InvertedPendulum-v2}
p=${3:-1.0}
echo $seed $env_name $p

python main.py --seed $seed --p $p --env_name $env_name
exit