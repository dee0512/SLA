#!/usr/bin/env bash

env_name=InvertedPendulum-v2
count=0
for seed in {0..9}; do
    if ! [ -f "models/AsymTD3_${env_name}_${seed}_final_actor" ];
    then
      sbatch asymmetricTD3.sh $seed $env_name
      count=$((count + 1))
    fi
done
echo Launched $count jobs