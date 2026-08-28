#!/bin/bash
# Launch full CA-PINN training on the server with nohup
# Usage: bash launch_training.sh

cd /home/student01/pngyuo/ca_pinn
source /home/student01/pngyuo/env.sh

export MPLBACKEND=Agg
export OMP_NUM_THREADS=10

mkdir -p experiments/outputs

# Kill any existing training process
pkill -f train_full_server.py 2>/dev/null
sleep 2

# Launch training in background with nohup
nohup python -u experiments/train_full_server.py > train_full.log 2>&1 &

echo "Training launched with PID: $!"
echo "Log file: /home/student01/pngyuo/ca_pinn/train_full.log"
echo "Check progress: tail -f /home/student01/pngyuo/ca_pinn/train_full.log"
