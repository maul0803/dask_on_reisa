#!/bin/bash

# Check if cluster_config.yml exists
if [ ! -f cluster_config.yml ]; then
  echo "Error: cluster_config.yml file not found!" >&2
  exit 1
fi

# Extract values from cluster_config.yml
PDI_PREFIX=$(grep "pdi_prefix" cluster_config.yml | awk -F': ' '{print $2}' | tr -d ' ')
PARTITION=$(grep "partition" cluster_config.yml | awk -F': ' '{print $2}' | tr -d ' ')
CORES_PER_NODE=$(grep "cores_per_node" cluster_config.yml | awk -F': ' '{print $2}' | tr -d ' ')
RAM_RAW=$(grep "ram_per_node" cluster_config.yml | awk -F': ' '{print $2}' | tr -d ' ')

# Validate extracted values
if [ -z "$PDI_PREFIX" ] || [ -z "$PARTITION" ] || [ -z "$CORES_PER_NODE" ] || [ -z "$RAM_RAW" ]; then
  echo "Error: Failed to retrieve values from cluster_config.yml!" >&2
  exit 1
fi

# Export PDI path
export PATH=${PDI_PREFIX}/bin:${PATH}

# Convert RAM_PER_NODE to MB based on its unit
if [[ "$RAM_RAW" == *G ]]; then
  RAM_PER_NODE=$(echo "$RAM_RAW" | tr -d 'G')  # Remove 'G'
  RAM_PER_NODE=$((RAM_PER_NODE * 1000))  # Convert to MB
elif [[ "$RAM_RAW" == *M ]]; then
  RAM_PER_NODE=$(echo "$RAM_RAW" | tr -d 'M')  # Already in MB
else
  echo "Error: Unknown memory unit in cluster_config.yml (expected G or M)" >&2
  exit 1
fi

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
WORKER_NODES=2 # DEISA uses (MPI_PROCESSES/4) worker nodes with 48 threads each one

# Compute memory per CPU (90% of total memory)
MEM_PER_CPU=$(echo "scale=0; ($RAM_PER_NODE * 0.9 / $CORES_PER_NODE)" | bc)

# Validate computed memory per CPU
if [ "$MEM_PER_CPU" -le 0 ]; then
  echo "Error: Invalid memory per CPU calculation!" >&2
  exit 1
fi

# AUXILIARY VALUES
SIMUNODES=$((PARALLELISM2 * PARALLELISM1 / MPI_PER_NODE))
NNODES=$((WORKER_NODES + SIMUNODES + 1))
NPROC=$((PARALLELISM2 * PARALLELISM1 + NNODES + 1))
MPI_TASKS=$((PARALLELISM2 * PARALLELISM1))
GLOBAL_SIZE=$((DATASIZE1 * DATASIZE2 * 8 / 1000000))
LOCAL_SIZE=$((GLOBAL_SIZE / MPI_TASKS))

# MANAGING FILES
date=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT=outputs/$date\_P$MPI_TASKS\_SN$SIMUNODES\_LS$LOCAL_SIZE\_GS$GLOBAL_SIZE\_I$GENERATION\_AN$WORKER_NODES
`which python` prescript.py $DATASIZE1 $DATASIZE2 $PARALLELISM1 $PARALLELISM2 $GENERATION $WORKER_NODES $MPI_PER_NODE $CORES_PER_NODE 
mkdir -p $OUTPUT
mkdir logs 2>/dev/null
touch logs/jobs.log
cp *.yml *.py simulation Script.sh $OUTPUT

# RUNNING
cd $OUTPUT
echo -e "Executing sbatch --parsable --nodes=$NNODES --mincpus=${CORES_PER_NODE} --mem-per-cpu=${MEM_PER_CPU}M --partition ${PARTITION} --ntasks=$NPROC Script.sh $SIMUNODES $MPI_PER_NODE $CORES_PER_NODE) in $PWD    "
echo -e "Executing $(sbatch --parsable --nodes=$NNODES --mincpus=${CORES_PER_NODE} --mem-per-cpu=${MEM_PER_CPU}M --partition ${PARTITION} --ntasks=$NPROC Script.sh $SIMUNODES $MPI_PER_NODE $CORES_PER_NODE) in $PWD    " >> $MAIN_DIR/logs/jobs.log
cd $MAIN_DIR
