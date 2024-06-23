#!/usr/bin/env bash

env_name=Pendulum-v2
count=0
for seed in {0..9}; do
    for p in 0.5; do
        if ! [ -f "models/sla_${env_name}_${seed}_${p}_final_actor" ];
        then
          sbatch train.sh $seed $env_name $p
          count=$((count + 1))
        fi
    done
done
echo Launched $count jobs