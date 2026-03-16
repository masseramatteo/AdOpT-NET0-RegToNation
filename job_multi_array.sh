#!/bin/bash
#SBATCH --job-name=network_opt_snellius
#SBATCH --nodes=1                     # 1 nodo per task dell'array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192           # full Genoa fat node
#SBATCH --partition=genoa
#SBATCH --time=12:00:00
#SBATCH --array=0                   # 🔥 2 nodi in parallelo (modifica qui per più nodi)
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=m.massera@uu.nl
#SBATCH --output=job_%A_%a.out        # output separato per array
#SBATCH --error=job_%A_%a.err         # error separato per array

echo "==========================="
echo "Running on node: $HOSTNAME"
echo "Array job index: $SLURM_ARRAY_TASK_ID"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Allowed CPUs (cgroup): $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}')"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_CPU_BIND: $SLURM_CPU_BIND"
echo "Started: $(date)"
echo "==========================="

### --- LOAD MODULES ---
module purge
module load Python/3.13.1-GCCcore-14.2.0
echo "Python module loaded."

### --- ACTIVATE VENV ---
source $HOME/AdOpT-NET0-RegToNation/venv/bin/activate
echo "Activated virtual environment."

### --- ENSURE GUROBI LIBRARY PATH ---
export GUROBI_HOME=$HOME/opt/gurobi-installs/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
echo "Configured Gurobi runtime paths."

### --- MOVE TO CODE FOLDER ---
cd $HOME/AdOpT-NET0-RegToNation/generic_node_optimization
echo "Working directory: $(pwd)"

### --- RUN PYTHON SCRIPT ---
python multi_node_run_creation_and_gurobi_optimization_parallel.py

echo "==========================="
echo "Finished at: $(date)"
echo "==========================="
