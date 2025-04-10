#!/bin/bash
#SBATCH --ntasks 1    		                              # Number of tasks to run
#SBATCH --cpus-per-task=64                                # CPU cores/threads
#SBATCH --gres=gpu:0       	                              # Number of GPUs (per node)
#SBATCH --mem 64000        	                              # Reserve 64 GB RAM for the job
#SBATCH --time 1-00:00    	                              # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	                          # Partition to submit to
#SBATCH --job-name cap-f1                                # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm-capf1.out             # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm-capf1.err  	  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist bird                                   # run only on bird

# activate shell
source /home/kapilg/.local/share/virtualenvs/local-blurred-captioning-exploration-bH5G8VRE/bin/activate

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/scripts/evaluation

# run cap f1 in parallel
python execute_cap_f1_parallel.py \
    --input-file ../../data/study-2-output/final-evaluated-captions/low-quality_evaluation_5432-images_2025-04-10_15:29.json \
    --num-workers 32 \
    --output-path ./results

# deactivate environment
exit