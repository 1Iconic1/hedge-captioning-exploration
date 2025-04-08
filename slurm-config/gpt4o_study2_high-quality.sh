#!/bin/bash
#SBATCH --ntasks 1    		                              # Number of tasks to run
#SBATCH --cpus-per-task=16                                # CPU cores/threads
#SBATCH --gres=gpu:0       	                              # Number of GPUs (per node)
#SBATCH --mem 64000        	                              # Reserve 64 GB RAM for the job
#SBATCH --time 4-00:00    	                              # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	                          # Partition to submit to
#SBATCH --job-name gpt4o_study2_high-quality_cpu-only     # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm-gpt4o-study2-high-quality-full.out    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm-gpt4o-study2-high-quality-full.err  	  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist bird                                   # run only on bird

# activate shell
source /home/kapilg/.local/share/virtualenvs/local-blurred-captioning-exploration-bH5G8VRE/bin/activate

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/scripts/

# run code
python gpt4o_captioner.py \
    --input-file "../data/study-2-input/high-quality-image_iq-1_text-no-text.json" \
    --output-dir "../data/study-2-output/labeled-data/high-quality-images/gpt4o-caption-output" \
    --scratch-path "../data/scratch/study-2/gpt4o-caption-output"

# deactivate environment
exit
