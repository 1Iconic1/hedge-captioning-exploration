#!/bin/bash
#SBATCH --ntasks 1    		                              # Number of tasks to run
#SBATCH --cpus-per-task=64                                # CPU cores/threads
#SBATCH --gres=gpu:1       	                              # Number of GPUs (per node)
#SBATCH --mem 64000        	                              # Reserve 64 GB RAM for the job
#SBATCH --time 2-00:00    	                              # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	                          # Partition to submit to
#SBATCH --job-name study2-evaluate                        # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm-study2-eval.out     # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm-study2-eval.err  	  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist bird                                   # run only on bird

# activate shell
source /home/kapilg/.local/share/virtualenvs/local-blurred-captioning-exploration-bH5G8VRE/bin/activate

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/scripts/evaluation

# run code
## full combined data
# python evaluate_captions.py \
#     --input ../../data/study-2-output/labeled-data/combined-caption-output/combined-caption-output_7304-images_2025-03-29_21:40:00.json \
#     --image-folder ../../data/caption-dataset/train \
#     --output-dir ../../data/study-2-output/labeled-data/evaluation-results

## single model
python evaluate_captions.py \
    --input ../../data/study-2-output/labeled-data/high-quality-images/molmo-caption-output/Molmo-7B-O-0924_caption-output_5428-images_start-0_end-5428_2025-04-08_17:42.json \
    --image-folder ../../data/caption-dataset/train \
    --output-dir ../../data/study-2-output/labeled-data/evaluation-results

# deactivate environment
exit
