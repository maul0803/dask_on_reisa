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


# get_result -> iter_task -> trigger -> process_task -> process_func
#            -> iter_func
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

    def get_result(self, process_func, iter_func, global_func=None, selected_iters=None, kept_iters=None,
                   timeline=False):
        """
        Retrieve the results of the simulation.

        :param process_func: Function that processes each task.
        :param iter_func: Function that processes the iterations.
        :param global_func: Optional function to process the entire result.
        :param selected_iters: List of iterations to compute.
        :param kept_iters: Number of iterations to keep.
        :param timeline: Boolean flag to enable Ray's timeline output.
        :return: Dictionary containing the results of the selected iterations.
        """
        max_tasks = ray.available_resources()['compute']
        results = list()
        actors = self.get_actors()

        if selected_iters is None:
            selected_iters = [i for i in range(self.iterations)]
        if kept_iters is None:
            kept_iters = self.iterations

        @ray.remote(max_retries=-1, resources={"compute": 1}, scheduling_strategy="DEFAULT")
        def process_task(rank: int, i: int, queue):
            """
            A task for processing each iteration for a given rank.

            :param rank: The rank of the process.
            :param i: The iteration index.
            :param queue: Queue used for processing.
            :return: Result of processing.
            """
            return process_func(rank, i, queue)

        iter_ratio = 1 / (ceil(max_tasks / self.mpi) * 2)

        @ray.remote(max_retries=-1, resources={"compute": 1, "transit": iter_ratio}, scheduling_strategy="DEFAULT")
        def iter_task(i: int, actors):
            """
            A task for processing iterations by triggering process tasks, executed by process nodes from the list of process actors (actors).
            iter_func is executed by an analytic node.

            :param i: The iteration index.
            :param actors: List of process actors that handle the tasks.
            :return: Processed results from the iterations.
            """
            enable_dask_on_ray()
            # current_results is a list of ObjectRef instances
            current_results = [actor.trigger.remote(process_task, i) for actor in actors]  # type: #List[ray._raylet.ObjectRef]
            current_results = ray.get(current_results)  # type: #List[List[ray._raylet.ObjectRef]]
            # current_results_list is a flattened list of ObjectRef instances
            current_results_list = list(
                itertools.chain.from_iterable(current_results))  # type: #List[ray._raylet.ObjectRef]
            # current_results_list after ray.get, which will contain the actual data (like Dask arrays)
            current_results_list = ray.get(current_results_list)  # type: #List[dask.array.core.Array]
            # current_results_array is a Dask array formed by stacking the results
            current_results_array = da.stack(current_results_list, axis=0)  # type: #dask.array.core.Array
            # current_results after applying the iteration function
            current_results = iter_func(i, current_results_array)  # type: #dask.array.core.Array
            return current_results

        start = time.time()  # Measure time
        results = [iter_task.remote(i, actors) for i in selected_iters]
        results = ray.get(results)
        tmp = da.stack(results, axis=0)

        eprint(
            "{:<21}".format("EST_ANALYTICS_TIME:") + "{:.5f}".format(time.time() - start) + " (avg:" + "{:.5f}".format(
                (time.time() - start) / self.iterations) + ")")

        if global_func:
            return global_func(tmp)  # RayList(results) TODO
        else:
            output = {}  # Output dictionary
            tmp = tmp.compute(scheduler=ray_dask_get)
            for i, _ in enumerate(selected_iters):
                if tmp[i] is not None:
                    output[selected_iters[i]] = tmp[i]

            if timeline:
                ray.timeline(filename="timeline-client.json")

            return output

    def get_actors(self):
        """
        Retrieve the actors created by the simulation.

        :return: List of ProcessActors.
        """
        timeout = 60
        start_time = time.time()
        error = True
        self.actors = list()
        while error:
            try:
                for rank in range(0, self.mpi, self.mpi_per_node):
                    self.actors.append(ray.get_actor("ranktor" + str(rank), namespace="mpi"))
                error = False
            except Exception as e:
                self.actors = list()
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time >= timeout:
                    raise Exception("Cannot get the Ray actors. Client is exiting")
            time.sleep(1)

        return self.actors

    def shutdown(self):
        """
        Shutdown the simulation and clean up memory.

        :return: None
        """
        if self.actors:
            for actor in self.actors:
                ray.kill(actor)
            ray.shutdown()
