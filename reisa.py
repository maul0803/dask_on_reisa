from datetime import datetime
from math import ceil
import itertools
import yaml
import time
import ray
import sys
import gc
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# "Background" code for the user

# This class will be the key to able the user to deserialize the data transparently
class RayList(list):
    def __call__(self, index): # Square brackets operation to obtain the data behind the references.
        item = super().__getitem__(index)
        if isinstance(index, slice):
            free(item)
        else:
            free([item])

    def __getitem__(self, index): # Square brackets operation to obtain the data behind the references.
        item = super().__getitem__(index)
        if isinstance(index, slice):
            return ray.get(RayList(item))
        else:
            return ray.get(item)


import os
import ray
import yaml


class Reisa:
    def __init__(self, file, address):
        self.iterations = 0
        self.mpi_per_node = 0
        self.mpi = 0
        self.datasize = 0
        self.workers = 0
        self.actors = list()

        # SPREAD DATA AS MUCH AS POSSIBLE THE DATA IN A DISTRIBUTED ENVIRONMENT: https://docs.ray.io/en/latest/ray-more-libs/dask-on-ray.html
        # Set environment variables for Ray
        os.environ["RAY_scheduler_spread_threshold"] = "0.0"
        # If the head node should not execute any computations
        os.environ["RAY_NUM_CPUS"] = "0"  # Specify that the head node should not use any CPUs

        # Initialize Ray
        if os.environ.get("REISA_DIR"):
            ray.init("ray://" + address + ":10001", runtime_env={"working_dir": os.environ.get("REISA_DIR")})
        else:
            ray.init("ray://" + address + ":10001", runtime_env={"working_dir": os.environ.get("PWD")})

        # Get the configuration of the simulation
        with open(file, "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.iterations = data["MaxtimeSteps"]
                self.mpi_per_node = data["mpi_per_node"]
                self.mpi = data["parallelism"]["height"] * data["parallelism"]["width"]
                self.workers = data["workers"]
                self.datasize = data["global_size"]["height"] * data["global_size"]["width"]
            except yaml.YAMLError as e:
                eprint(e)

        return

    def get_result(self, process_func, iter_func, global_func=None, selected_iters=None, kept_iters=None, timeline=False):
        max_tasks = ray.available_resources()['compute']
        results = list()
        actors = self.get_actors()
        
        if selected_iters is None:
            selected_iters = [i for i in range(self.iterations)]
        if kept_iters is None:
            kept_iters = self.iterations

        # process_task = ray.remote(max_retries=-1, resources={"compute":1}, scheduling_strategy="DEFAULT")(process_func)
        # iter_task = ray.remote(max_retries=-1, resources={"compute":1, "transit":0.5}, scheduling_strategy="DEFAULT")(iter_func)

        @ray.remote(max_retries=-1, resources={"compute":1}, scheduling_strategy="DEFAULT")
        def process_task(rank: int, i: int, queue):
            return process_func(rank, i, queue)
            
        iter_ratio=1/(ceil(max_tasks/self.mpi)*2)

        @ray.remote(max_retries=-1, resources={"compute":1, "transit":iter_ratio}, scheduling_strategy="DEFAULT")
        def iter_task(i: int, actors):
            current_results = [actor.trigger.remote(process_task, i) for j, actor in enumerate(actors)]
            current_results = ray.get(current_results)
            
            if i >= kept_iters-1:
                [actor.free_mem.remote(current_results[j], i-kept_iters+1) for j, actor in enumerate(actors)]
            
            return iter_func(i, RayList(itertools.chain.from_iterable(current_results)))

        start = time.time() # Measure time
        results = [iter_task.remote(i, actors) for i in selected_iters]
        ray.wait(results, num_returns=len(results)) # Wait for the results
        eprint("{:<21}".format("EST_ANALYTICS_TIME:") + "{:.5f}".format(time.time()-start) + " (avg:"+"{:.5f}".format((time.time()-start)/self.iterations)+")")
        if global_func:
            return global_func(RayList(results))
        else:
            tmp = ray.get(results)
            output = {} # Output dictionary

            for i, _ in enumerate(selected_iters):
                if tmp[i] is not None:
                    output[selected_iters[i]] = tmp[i]
            
            if timeline:
                ray.timeline(filename="timeline-client.json")

            return output

    # Get the actors created by the simulation
    def get_actors(self):
        timeout = 60
        start_time = time.time()
        error = True
        self.actors = list()
        while error:
            try:
                for rank in range(0, self.mpi, self.mpi_per_node):
                    self.actors.append(ray.get_actor("ranktor"+str(rank), namespace="mpi"))
                error = False
            except Exception as e:
                self.actors=list()
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time >= timeout:
                    raise Exception("Cannot get the Ray actors. Client is exiting")
            time.sleep(1)

        return self.actors

    # Erase iterations from simulation memory
    def shutdown(self):
        if self.actors:
            for actor in self.actors:
                ray.kill(actor)
            ray.shutdown()