#!/bin/bash
#SBATCH --ntasks 1    		                # Number of tasks to run
#SBATCH --cpus-per-task=16                  # CPU cores/threads
#SBATCH --gres=gpu:1       	                # Number of GPUs (per node)
#SBATCH --mem 64000        	                # Reserve 64 GB RAM for the job
#SBATCH --time 2-00:00    	                # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	            # Partition to submit to
#SBATCH --job-name vizwiz_captioner         # The name of the job that is running
#SBATCH --output /scratch/kapil/slurm.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm.err  	# File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist dizzy                     # run only on dizzy

# activate shell
source /home/kapilg/.local/share/virtualenvs/local-blurred-captioning-exploration-bH5G8VRE/bin/activate

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/scripts/

# run code
python vizwiz_captioner.py

# deactivate environment
exit
