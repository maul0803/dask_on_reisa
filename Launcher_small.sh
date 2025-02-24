#!/bin/bash

# spack load pdiplugin-pycall@1.6.0 pdiplugin-mpi@1.6.0;

PDI_PREFIX=${HOME}/opt/pdi_py39
export PATH=${PDI_PREFIX}/bin:${PATH}

#PARTITION=cpu_short    # For Ruche cluster
PARTITION=short         # For FT3 cluster

MAIN_DIR=$PWD

GR='\033[0;32m'
BL='\033[0;34m'
NC='\033[0m' # No Color

# CHECKING SOFTWARE
echo -n -e "${BL}PDI"   
echo -e "${GR} `which pdirun`${NC}"
echo -n -e "${BL}MPI"   
echo -e "${GR} `which mpirun`${NC}"
echo -n -e "${BL}PYTHON"
echo -e "${GR} `which python`${NC}"
echo -n -e "${BL}RAY"   
echo -e "${GR} `which ray`${NC}"
echo -e "Running in $MAIN_DIR\n"

# COMPILING
(CC=gcc CXX=g++ pdirun cmake .) > /dev/null 2>&1
pdirun make -B simulation

# MPI VALUES
PARALLELISM1=4  # Number of MPI nodes along the x-axis
PARALLELISM2=8  # Number of MPI nodes along the y-axis
MPI_PER_NODE=32 # Number of MPI processes per node (unchanged)

# DATASIZE
DATASIZE1=$((256*$PARALLELISM1)) # Number of elements along the x-axis
DATASIZE2=$((512*$PARALLELISM2)) # Number of elements along the y-axis

# STEPS 
GENERATION=25 # Number of simulation iterations

# ANALYTICS HARDWARE
WORKER_NODES=1 # Reduced to 1 worker node
CPUS_PER_WORKER=40 # Number of CPUs per worker (unchanged)

# AUXILIARY VALUES
SIMUNODES=$(($PARALLELISM2 * $PARALLELISM1 / $MPI_PER_NODE)) # Number of simulation nodes
NNODES=$(($WORKER_NODES + $SIMUNODES + 1)) # Workers + head + simulation
NPROC=$(($PARALLELISM2 * $PARALLELISM1 + $NNODES + 1)) # Total number of deployed tasks
MPI_TASKS=$(($PARALLELISM2 * $PARALLELISM1)) # Number of MPI tasks
GLOBAL_SIZE=$(($DATASIZE1 * $DATASIZE2 * 8 / 1000000)) # Global size in MB
LOCAL_SIZE=$(($GLOBAL_SIZE / $MPI_TASKS)) # Local size in MB

# MANAGING FILES
date=$(date +%Y-%m-%d_%R)
OUTPUT=outputs/$date\_P$MPI_TASKS\_SN$SIMUNODES\_LS$LOCAL_SIZE\_GS$GLOBAL_SIZE\_I$GENERATION\_AN$WORKER_NODES
`which python` prescript.py $DATASIZE1 $DATASIZE2 $PARALLELISM1 $PARALLELISM2 $GENERATION $WORKER_NODES $MPI_PER_NODE $CPUS_PER_WORKER $WORKER_THREADING # Create config.yml
mkdir -p $OUTPUT
mkdir logs 2>/dev/null
touch logs/jobs.log
cp *.yml *.py simulation Script.sh $OUTPUT

# RUNNING
cd $OUTPUT
echo -e "Executing sbatch --parsable -N $NNODES --mincpus=40 --partition ${PARTITION} --ntasks=$NPROC Script.sh $SIMUNODES $MPI_PER_NODE $CPUS_PER_WORKER) in $PWD    "
echo -e "Executing $(sbatch --parsable -N $NNODES --mincpus=40 --partition ${PARTITION} --ntasks=$NPROC Script.sh $SIMUNODES $MPI_PER_NODE $CPUS_PER_WORKER) in $PWD    " >> $MAIN_DIR/logs/jobs.log
cd $MAIN_DIR
