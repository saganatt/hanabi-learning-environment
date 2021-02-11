#!/bin/bash

for seed in 283723 12345 39845 23458 98437 ; do
  echo "Seed: ${seed}"
  python examples/rl_env_example.py --agent_class QAgent --num_episodes 100000 \
    --num_test_episodes 1000 --seed $seed > "qagent_seed_${seed}_n_100k_debug.txt" 2>&1 &
  python examples/rl_env_example.py --agent_class SimpleAgent --num_episodes 1000 \
    --num_test_episodes 1000 --seed $seed > "simple_agent_seed_${seed}_n_1000_debug.txt" 2>&1 &
done
wait
echo "Done"
