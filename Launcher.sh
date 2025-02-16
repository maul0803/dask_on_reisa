#!/bin/bash

# spack load pdiplugin-pycall@1.6.0 pdiplugin-mpi@1.6.0;

PDI_PREFIX=${HOME}/opt/pdi_py39
export PATH=${PDI_PREFIX}/bin:${PATH}

#PARTITION=cpu_short    # Ruche
PARTITION=short         # FT3

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
PARALLELISM1=8 # MPI nodes axis x
PARALLELISM2=16 # MPI nodes axis y
MPI_PER_NODE=32 # MPI processes per simulation node

# DATASIZE
DATASIZE1=$((256*$PARALLELISM1)) # Number of elements axis x
DATASIZE2=$((512*$PARALLELISM2)) # Number of elements axis y

# STEPS 
GENERATION=25 # Number of iterations on the simulation

# ANALYTICS HARDWARE
WORKER_NODES=2 # DEISA uses (MPI_PROCESSES/4) worker nodes  with 48 threads each one
CPUS_PER_WORKER=40 # 24 # Parallelism on each worker

# AUXILIAR VALUES
SIMUNODES=$(($PARALLELISM2 * $PARALLELISM1 / $MPI_PER_NODE)) # NUMBER OF SIMULATION NODES
NNODES=$(($WORKER_NODES + $SIMUNODES + 1)) # WORKERS + HEAD + SIMULATION (CLIENT WILL BE WITHIN THE HEAD NODE)
NPROC=$(($PARALLELISM2 * $PARALLELISM1 + $NNODES + 1)) # NUMBER OF DEPLOYED TASKS (MPI + ALL RAY INSTANCES + CLIENT)
MPI_TASKS=$(($PARALLELISM2 * $PARALLELISM1)) # NUMBER OF DEPLOYED TASKS (MPI + ALL RAY INSTANCES + CLIENT)
GLOBAL_SIZE=$(($DATASIZE1 * $DATASIZE2 * 8 / 1000000)) # NUMBER OF DEPLOYED TASKS (MPI + ALL RAY INSTANCES + CLIENT)
LOCAL_SIZE=$(($GLOBAL_SIZE / $MPI_TASKS)) # NUMBER OF DEPLOYED TASKS (MPI + ALL RAY INSTANCES + CLIENT)

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
