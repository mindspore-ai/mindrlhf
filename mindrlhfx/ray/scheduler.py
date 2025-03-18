import os
import socket
import ray

import mindspore as ms
from mindspore import mint

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class Scheduler:
    def __init__(self):
        # set up ip and port
        self._setup_ip_and_port()

        # init scheduler
        self.success = False
        if not ms.communication._comm_helper._is_initialized():
            mint.distributed.init_process_group(
                backend="hccl"
            )
            self.success = True

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(('', 0))
            return sock.getsockname()[1]

    def _get_availale_ms_sched_host_port(self):
        return ray._private.services.get_node_ip_address(), str(self._get_free_port())

    def _setup_ip_and_port(self):
        ms_sched_host, ms_sched_port = self._get_availale_ms_sched_host_port()
        rank_zero_info = {
            "MS_SCHED_HOST": ms_sched_host,
            "MS_SCHED_PORT": ms_sched_port,
        }
        # store env info
        store_center_name = os.environ.get("STORE_CENTER_NAME", None)
        if not store_center_name:
            print("store center name is not set, automatically set to WorkerDict_tmp_store_center", flush=True)
            store_center_name = "WorkerDict_tmp_store_center"
        self.store_center = create_ip_port_store_center(store_center_name, rank_zero_info)
        os.environ.update(rank_zero_info)
    
    def get_status(self):
        return self.success


def create_scheduler(name, world_size, placement_group, store_center_name):
    env_vars = {
        "MS_ROLE": "MS_SCHED",
        "MS_WORKER_NUM": str(world_size),
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
        'WORLD_SIZE': str(world_size),
        'STORE_CENTER_NAME': store_center_name
    }
    options = {
        "scheduling_strategy":
                PlacementGroupSchedulingStrategy(placement_group=placement_group,
                                                 placement_group_bundle_index=0),
        'runtime_env': {'env_vars': env_vars},
        'name': name
    }
    return Scheduler.options(**options).remote()


@ray.remote
class IpPortStoreCenter:

    def __init__(self, rank_zero_info):
        self.rank_zero_info = rank_zero_info

    def get_rank_zero_info(self):
        return self.rank_zero_info


def create_ip_port_store_center(name, info):
    return IpPortStoreCenter.options(name=name).remote(info)
