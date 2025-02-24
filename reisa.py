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
    """
    Print messages to the standard error stream.

    :param args: Positional arguments to print.
    :param kwargs: Keyword arguments for the print function.
    """
    print(*args, file=sys.stderr, **kwargs)


# "Background" code for the user
class Reisa:
    """
    A class that initializes and manages a Ray and Dask based simulation environment.
    """

    def __init__(self, file, address):
        """
        Initialize the Reisa simulation by reading configuration parameters
        from a YAML file and setting up Ray.

        :param file: Path to the YAML configuration file.
        :param address: Address of the Ray cluster.
        """
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
        actors = self.get_actors()

        if selected_iters is None:
            selected_iters = [i for i in range(self.iterations)]
        if kept_iters is None:
            kept_iters = self.iterations

        @ray.remote(max_retries=-1, resources={"compute": 1}, scheduling_strategy="DEFAULT")
        def process_task(rank: int, i: int, queue):
            """
            Remote function to process a simulation step.

            :param rank: Rank of the process.
            :param i: Current iteration.
            :param queue: Data queue containing simulation values.
            :return: Processed result.
            """
            return process_func(rank, i, queue)

        iter_ratio = 1 / (ceil(max_tasks / self.mpi) * 2)

        @ray.remote(max_retries=-1, resources={"compute": 1, "transit": iter_ratio}, scheduling_strategy="DEFAULT")
        def iter_task(i: int, actors):
            """
            Remote function to process an iteration.

            :param i: Current iteration index.
            :param actors: list of Ray actors managing the simulation.
            :return: Processed iteration result.
            """
            enable_dask_on_ray()
            current_results = [actor.trigger.remote(process_task, i) for actor in actors]  # type: #List[ray._raylet.ObjectRef]
            current_results = ray.get(current_results)  # type: #List[List[ray._raylet.ObjectRef]]
            current_results_list = list(
                itertools.chain.from_iterable(current_results))  # type: #List[ray._raylet.ObjectRef]
            current_results_list = ray.get(current_results_list)  # type: #List[dask.array.core.Array]
            current_results_array = da.stack(current_results_list, axis=0)  # type: #dask.array.core.Array
            current_results = iter_func(i, current_results_array)  # type: #dask.array.core.Array
            return current_results

        start = time.time()  # Measure time
        results = [iter_task.remote(i, actors) for i in selected_iters]
        ray.wait(results, num_returns=len(results))  # Wait for the results
        eprint(
            "{:<21}".format("EST_ANALYTICS_TIME:") + "{:.5f}".format(time.time() - start) + " (avg:" + "{:.5f}".format(
                (time.time() - start) / self.iterations) + ")")
        tmp = ray.get(results)
        if global_func:
            return global_func(tmp)  # RayList(results) TODO
        else:
            output = {}  # Output dictionary

            for i, _ in enumerate(selected_iters):
                if tmp[i] is not None:
                    output[selected_iters[i]] = tmp[i]

            if timeline:
                ray.timeline(filename="timeline-client.json")

            return output

    def get_actors(self):
        """
        Retrieve the Ray actors managing the simulation.

        :return: A list of Ray actors.
        :raises Exception: If one of the Ray actors is not available after a timeout period.
        """
        timeout = 60
        start_time = time.time()
        self.actors = list()
        while True:
            try:
                for rank in range(0, self.mpi, self.mpi_per_node):
                    self.actors.append(ray.get_actor("ranktor"+str(rank), namespace="mpi"))
                return self.actors
            except Exception:
                self.actors = list()
                if time.time() - start_time >= timeout:
                    raise Exception("Cannot get the Ray actors. Client is exiting.")
            time.sleep(1)

    def shutdown(self):
        """
        Shutdown the simulation and clean up memory.

        :return: None
        """
        if self.actors:
            for actor in self.actors:
                ray.kill(actor)
            ray.shutdown()
