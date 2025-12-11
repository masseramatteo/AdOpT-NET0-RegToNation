#!/bin/bash
#SBATCH --job-name=network_opt_snellius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.massera@uu.nl
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

echo "Running on node: $HOSTNAME"
echo "Job started at: $(date)"

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
python run_creation_and_gurobi_optimization_parallel.py

echo "Job finished at: $(date)"
