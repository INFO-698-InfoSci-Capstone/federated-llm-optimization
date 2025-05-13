#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=l3d-fflorist
#SBATCH --account=jdass
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=45:00:00
#SBATCH --gres=gpu:4
#SBATCH -o %J.out
#SBATCH --open-mode=append


# --------------------
### Code Section
# --------------------

set -oe pipefail

module load python
module load cuda11
module load pytorch/nvidia

source fedllm/bin/activate
export HF_HOME=/groups/jdass/huggingface_cache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 1.0
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.99
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.98
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.97
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.96
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.95
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.94
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.93
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.92
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.91
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.90
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.89
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.88
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.87
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.86
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.85
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.84
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.83
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.82
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.81
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --florist True --num_clients 8 --threshold 0.80
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --stacking True --num_clients 8
python3 main.py --global_model 'meta-llama/Llama-3.2-1B' --data_path  "./data" --output_dir './Llama-3.2-1B-dolly-homo-3-1-8/' --num_communication_rounds 3 --local_num_epochs 1 --num_clients 8

