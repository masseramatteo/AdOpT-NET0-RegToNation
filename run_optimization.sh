#!/bin/bash
#SBATCH --job-name=network_opt_snellius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=fat_genoa
#SBATCH --time=50:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.massera@uu.nl
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

echo "==========================="
echo "Running on node:          $HOSTNAME"
echo "Array task index:         $SLURM_ARRAY_TASK_ID"
echo "Array task count:         $SLURM_ARRAY_TASK_COUNT"
echo "SLURM_JOB_ID:             $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK:      $SLURM_CPUS_PER_TASK"
echo "Allowed CPUs (cgroup):    $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
echo "Started:                  $(date)"
echo "==========================="

### --- CREATE LOGS FOLDER ---
mkdir -p $HOME/AdOpT-NET0-RegToNation/four_node_configuration/logs

### --- LOAD MODULES ---
module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
echo "Python module loaded."

### --- ACTIVATE VENV ---
source $HOME/AdOpT-NET0-RegToNation/venv/bin/activate
echo "Activated virtual environment."

### --- GUROBI ---
export GUROBI_HOME=$HOME/opt/gurobi-installs/gurobi1301/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
echo "Gurobi runtime paths configured."

### --- MOVE TO CODE FOLDER ---
cd $HOME/AdOpT-NET0-RegToNation/four_node_configuration
echo "Working directory: $(pwd)"

### --- RUN ---
python FOUR_NODE_run_creation_and_gurobi_optimization_parallel.py

echo "==========================="
echo "Finished: $(date)"
echo "==========================="
