#!/bin/bash
#SBATCH --job-name=network_opt_snellius
#SBATCH --nodes=1                     # 1 nodo per task dell'array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192           # full Genoa fat node
#SBATCH --partition=genoa
#SBATCH --time=40:00:00
#SBATCH --array=0-1                   # 🔥 2 nodi in parallelo (modifica qui per più nodi)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.massera@uu.nl
#SBATCH --output=job_%A_%a.out        # output separato per array
#SBATCH --error=job_%A_%a.err         # error separato per array

echo "==========================="
echo "Running on node: $HOSTNAME"
echo "Array job index: $SLURM_ARRAY_TASK_ID"
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
