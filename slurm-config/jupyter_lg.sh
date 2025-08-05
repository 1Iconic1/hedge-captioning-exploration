#!/bin/bash
#SBATCH --ntasks 1    		                # Number of tasks to run
#SBATCH --cpus-per-task=32                 	# CPU cores/threads
#SBATCH --gres=gpu:2       	                # Number of GPUs (per node)
#SBATCH --mem 262144        	                # Reserve 64 GB RAM for the job
#SBATCH --time 1-00:00    	                # Max Runtime in D-HH:MM
#SBATCH --partition liv.p    	            	# Partition to submit to
#SBATCH --output /scratch/kapil/slurm.out   	# File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /scratch/kapil/slurm.err  	# File to which STDERR will be written, %j inserts jobid
#SBATCH --nodelist dizzy                     	# run only on dizzy

# go to the correct directory
cd /home/kapilg/projects/local-blurred-captioning-exploration/

# activate shell 
source .venv/bin/activate

# run notebook
jupyter lab --no-browser --port=9585

# deactivate environment
exit
