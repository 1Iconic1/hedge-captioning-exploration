#!/bin/bash
#SBATCH --ntasks 1    		                              # Number of tasks to run
#SBATCH --cpus-per-task=8                                 # CPU cores/threads
#SBATCH --gres=gpu:0       	                              # Number of GPUs (per node)
#SBATCH --mem 32000        	                              # Reserve 32 GB RAM for the job
#SBATCH --time 4-00:00    	                              # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	                          # Partition to submit to
#SBATCH --job-name generate_samples_cpu                   # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm-generate-samples-cpu.out    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm-generate-samples-cpu.err  	  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist dizzy                                   # run only on dizzy

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/

# activate shell
source .venv/bin/activate

# run code
cd chi-2026/
python generate_samples.py \
    --input-file ./coded-data/cleaned/final-image-sample_945-images_09-23-25.csv \
    --models gpt \
    --num-samples 10 \
    --greedy-response True

# deactivate environment
exit
