from datetime import datetime
from math import ceil
import itertools
import yaml
import time
import ray
import sys
import gc
import os
import dask.array as da
from ray.util.dask import ray_dask_get, enable_dask_on_ray, disable_dask_on_ray
#get_result -> iter_task -> trigger -> process_task -> process_func
#	   		-> iter_func
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# "Background" code for the user
class Reisa:
    def __init__(self, file, address):
        self.iterations = 0
        self.mpi_per_node = 0
        self.mpi = 0
        self.datasize = 0
        self.workers = 0
        self.actors = list()
        
        # Init Ray
        if os.environ.get("REISA_DIR"):
            ray.init("ray://"+address+":10001", runtime_env={"working_dir": os.environ.get("REISA_DIR")})
        else:
            ray.init("ray://"+address+":10001", runtime_env={"working_dir": os.environ.get("PWD")})
       
        # Get the configuration of the simulatin
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
            enable_dask_on_ray()
            current_results = [actor.trigger.remote(process_task, i) for actor in actors]
            current_results = ray.get(current_results)
            #print("current_results_init:", type(current_results)) #<class 'list'>
            #print("current_results_init[0]:", type(current_results[0])) #<class 'list'>
            #print("current_results_init[0][0]:", type(current_results[0][0])) #<class 'ray._raylet.ObjectRef'>
            current_results_list = list(itertools.chain.from_iterable(current_results))
            #print("current_results_list:", type(current_results_list)) #<class 'list'>
            #print("current_results_list[0]:", type(current_results_list[0])) #<class 'ray._raylet.ObjectRef'>
            current_results_list = ray.get(current_results_list)
            #print("current_results_list2:", type(current_results_list)) #<class 'list'>
            #print("current_results_list2[0]:", type(current_results_list[0])) #<class 'dask.array.core.Array'>
            current_results_array = da.stack(current_results_list, axis=0)
            #print("current_results_array:", type(current_results_array)) #<class 'dask.array.core.Array'>
            current_results = iter_func(i, current_results_array)
            #print("current_results:", type(current_results)) #<class 'dask.array.core.Array'>
            return current_results
        start = time.time() # Measure time
        results = [iter_task.remote(i, actors) for i in selected_iters]
        results = ray.get(results)
        #print("results:", type(results)) #<class 'list'>
        #print("results[0]:", type(results[0])) #<class 'dask.array.core.Array'>
        tmp = da.stack(results, axis=0)
        #ray.wait(results, num_returns=len(results)) # Wait for the results
        eprint("{:<21}".format("EST_ANALYTICS_TIME:") + "{:.5f}".format(time.time()-start) + " (avg:"+"{:.5f}".format((time.time()-start)/self.iterations)+")")
        if global_func:
            return global_func(tmp) #RayList(results) TODO
        else:
            output = {} # Output dictionary
            tmp = tmp.compute(scheduler=ray_dask_get)
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