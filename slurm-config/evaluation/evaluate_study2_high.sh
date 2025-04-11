#!/bin/bash
#SBATCH --ntasks 1    		                              # Number of tasks to run
#SBATCH --cpus-per-task=64                                # CPU cores/threads
#SBATCH --gres=gpu:1       	                              # Number of GPUs (per node)
#SBATCH --mem 96000        	                              # Reserve 64 GB RAM for the job
#SBATCH --time 1-00:00    	                              # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	                          # Partition to submit to
#SBATCH --job-name study2-evaluate                        # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm-study2-eval-high.out     # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm-study2-eval-high.err  	  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist bird                                   # run only on bird

# activate shell
source /home/kapilg/.local/share/virtualenvs/local-blurred-captioning-exploration-bH5G8VRE/bin/activate

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/scripts/evaluation

# high-quality images
# this dataset has no evaluation so far
python evaluate_captions.py \
    --input ../../data/study-2-output/final-labeled-data/high-quality_evaluation_5428-images_2025-04-10_15:18.json \
    --image-folder ../../data/caption-dataset/train \
    --output-dir ../../data/study-2-output/final-evaluated-captions

# deactivate environment
exit
