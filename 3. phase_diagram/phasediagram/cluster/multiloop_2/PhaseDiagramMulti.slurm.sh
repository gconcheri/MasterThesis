#!/bin/bash
#SBATCH --job-name=PhaseDiagramMulti
#SBATCH --chdir=./
#SBATCH --output=./PhaseDiagramMulti.%A_%3a.out  # %J=jobid.step, %N=node.
#
# To support getting emails, adjust the following two lines and remove the `# `,i.e. make them start with `#SBATCH `
#SBATCH --mail-type=ALL  # or `fail,end`, but it's not recommended
#SBATCH --mail-user=giovanni.concheri@tum.de  # adjust...
# NOTE: use ONLY YOUR UNIVERSITY EMAIL, DON'T USE/FORWARD EMAIL to other email providers like gmail.com!
# You can get a lot of emails from the cluster, and other email providers then sometimes mark the whole university as sending spam.
# This might results in your professor not being able to write emails to his friends anymore...
#SBATCH --time=5-00:00:00
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --qos=longrun
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --array=1-825

set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
# module load python/3.7
# conda activate tenpy


. /home/t30/all/go56vod/envs/MasterProject/bin/activate #adjust to my environment


# use SLURM_CPUS_PER_TASK, if not set default to SLURM_CPUS_ON_NODE
USE_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE}}
if [ -z "$USE_NUM_THREADS" ]
then
	USE_NUM_THREADS="$(nproc --all)"
	echo "WARNING: SLURM_CPUS_ON_NODE not set! Using all cores on machine, NTHREADS=$USE_NUM_THREADS"
fi
# When requesting --cpus-per-task 32 on nodes with CPU hyperthreading,
# slurm will allocate the job 32 threads = 16 physical cores x 2 (hyper)threads per core.
# For many numerical applications, e.g BLAS/LAPACK functions like matrix diagonalization, 
# it is better to ignore hyperthreading and rather set NUM_THREADS to the number of physical cores.
# Hence we divide by 2 here:
USE_NUM_THREADS=$(($USE_NUM_THREADS / 2 ))
export OMP_NUM_THREADS=$USE_NUM_THREADS  # number of CPUs per node, total for all the tasks below.
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=$USE_NUM_THREADS  # number of CPUs per node, total for all the tasks below.
export NUMBA_NUM_THREADS=$USE_NUM_THREADS

echo "Running task $SLURM_ARRAY_TASK_ID specified in PhaseDiagramMulti.config.pkl on $HOSTNAME at $(date) with $USE_NUM_THREADS threads"
python /home/t30/all/go56vod/Desktop/MasterThesis/cluster_jobs.py run PhaseDiagramMulti.config.pkl $SLURM_ARRAY_TASK_ID
# if you want to redirect output to file, you can append the following to the line above:
#     &> "PhaseDiagramMulti.task_$SLURM_ARRAY_TASK_ID.out"
echo "finished at $(date)"
