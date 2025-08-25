#!/bin/bash
#SBATCH --ntasks 1    		                # Number of tasks to run
#SBATCH --cpus-per-task=32                 	# CPU cores/threads
#SBATCH --gres=gpu:2       	                # Number of GPUs (per node)
#SBATCH --mem 63488        	                # Reserve 64 GB RAM for the job
#SBATCH --time 7-00:00    	                # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	            # Partition to submit to
#SBATCH --job-name chi26-all-models         # The name of the job that is running
#SBATCH --output /scratch/kapil/chi26-all-models.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/chi26-all-models.err  	# File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist bird                     	# run only on bird

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/

# activate shell
source .venv/bin/activate

# run script
cd chi-2026
python generate_captions.py

# deactivate environment
exit
