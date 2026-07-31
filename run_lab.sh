#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --partition=alpha 
#SBATCH --account=p_scads_finetune 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=9:00:00          
#SBATCH --output=log/pipeline-%j.out 

module purge
cd /home/yuzh952h/workspaces/horse/yuzh952h-cb/
source env.sh
source myenv/bin/activate
cd CitationBias
python playground.py