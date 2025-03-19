#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH -o reisa.log
#SBATCH --error reisa.log
#SBATCH --mem-per-cpu=4G
#SBATCH --wait-all-nodes=1
##SBATCH --exclusive
###################################################################################################
start=$(date +%s%N)
echo -e "Slurm job started at $(date +%d/%m/%Y_%X)\n"

# ENVIRONMENT VARIABLES
unset RAY_ADDRESS;
export RAY_record_ref_creation_sites=0
export RAY_SCHEDULER_EVENTS=0
export OMP_NUM_THREADS=1 # To prevent errors
export RAY_PROFILING=0
# export RAY_task_events_report_interval_ms=1000   
export RAY_memory_monitor_refresh_ms=500
export RAY_memory_usage_threshold=0.99
export RAY_verbose_spill_logs=0
export REISA_DIR=$PWD

# LOCAL VARIABLES
REDIS_PASSWORD=$(uuidgen)
MPI_TASKS=$(($SLURM_NTASKS - $SLURM_NNODES - 1))
MPI_PER_NODE=$2
CPUS_PER_WORKER=$3
NUM_SIM_NODES=$1
WORKER_NUM=$(($SLURM_JOB_NUM_NODES - 1 - $NUM_SIM_NODES))
IN_SITU_RESOURCES=0

# GET ALLOCATED NODES
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODES_ARRAY=($NODES)
SIMARRAY=(${NODES_ARRAY[@]:$WORKER_NUM+1:$NUM_SIM_NODES});
for nodo in ${SIMARRAY[@]}; do
  SIM_NODE_LIST+="$nodo,"
done
SIM_NODE_LIST=${SIM_NODE_LIST%,}
echo -e "Initing Ray (1 head node + $WORKER_NUM worker nodes + $NUM_SIM_NODES simulation nodes) on nodes: \n\t${NODES_ARRAY[@]}"

# GET HEAD NODE INFO
head_node=${NODES_ARRAY[0]}
    head_node_ip=$(srun -N 1 -n 1 --relative=0 echo $(/sbin/ip -f inet addr show ib0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p') &)
ulimit -n 16384 >> reisa.log
port=6379
echo -e "Head node: $head_node_ip:$port\n"
export RAY_ADDRESS=$head_node_ip:$port

# START RAY IN THE HEAD NODE
srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=$CPUS_PER_WORKER \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --redis-password "$REDIS_PASSWORD" --include-dashboard False\
    --num-cpus $CPUS_PER_WORKER --block --resources='{"compute": 0, "head":1}' --system-config='{"local_fs_capacity_threshold":0.999}'  1>/dev/null 2>&1 &
echo $RAY_ADDRESS > ../../address.var

cnt=0
k=0
max=10
 # WAIT FOR HEAD NODE
while [ $cnt -lt 1 ] && [ $k -lt $max ]; do
    sleep 5
    cnt=$(ray status --address=$RAY_ADDRESS 2>/dev/null | grep -c node_)
    k=$((k+1))
done

# START RAY IN COMPUTING NODES
for ((i = 1; i <= WORKER_NUM; i++)); do
    node_i=${NODES_ARRAY[$i]}
    srun --nodes=1 --ntasks=1 --relative=$i --cpus-per-task=$CPUS_PER_WORKER --mem=128G \
        ray start --address $RAY_ADDRESS --redis-password "$REDIS_PASSWORD" \
        --num-cpus $(($CPUS_PER_WORKER)) --block --resources="{\"compute\": ${CPUS_PER_WORKER}, \"transit\": 1}" --object-store-memory=$((96*10**9)) 1>/dev/null 2>&1 &
done


# START RAY IN SIMULATION NODES
for ((; i < $SLURM_JOB_NUM_NODES; i++)); do
    node_i=${NODES_ARRAY[$i]}
    srun  --nodes=1 --ntasks=1 --relative=$i --cpus-per-task=$((1+$IN_SITU_RESOURCES)) --mem=128G \
      ray start --address $RAY_ADDRESS --redis-password "$REDIS_PASSWORD" \
      --num-cpus=$((1+$IN_SITU_RESOURCES)) --block --resources="{\"actor\": 1, \"compute\": ${IN_SITU_RESOURCES}}" --object-store-memory $((96*10**9))  1>/dev/null 2>&1 &
done

cnt=0
k=0
max=10
# WAIT FOR ALL THE RAY NODES BEFORE START THE SIMULATION
while [ $cnt -lt $SLURM_JOB_NUM_NODES ] && [ $k -lt $max ]; do
    sleep 10
    cnt=$(ray status --address=$RAY_ADDRESS 2>/dev/null | grep -c node_)
    k=$((k+1))
done
end=$(date +%s%N)

# LAUNCH THE CLIENT WITHIN THE HEAD NODE
srun --oversubscribe --overcommit --nodes=1 --ntasks=1 --relative=0 -c 1\
    `which python` reduction.py &
    #`which python` derivative.py &
    # `which python` client.py --ray-timeline &
client=$!

# ray status --address=$RAY_ADDRESS

# LAUNCH THE SIMULATION
pdirun srun --oversubscribe --overcommit -N $NUM_SIM_NODES --ntasks-per-node=$MPI_PER_NODE\
    -n $MPI_TASKS --nodelist=$SIM_NODE_LIST --cpus-per-task=1\
        ./simulation $SLURM_JOB_ID &
sim=$!

# PRINT CLUSTER DEPLOYING TIME
elapsed=$((end-start))
elapsed=$(bc <<< "scale=5; $elapsed/1000000000")
sleep 1
printf "\n%-21s%s\n" "RAY_DEPLOY_TIME:" "$elapsed"

# WAIT FOR THE RESULTS
wait $client
wait $sim

echo -e "\nSlurm job finished at $(date +%d/%m/%Y_%X)"
