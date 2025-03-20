# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from abc import ABC

import ray
from ray.util import list_named_actors
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import mindspore as ms

from mindrlhfx.ray.scheduler import create_scheduler
from mindrlhfx.ray.resource import RayResourcePool, ResourcePoolManager


class WorkerDict:
    '''
    One WorkerDict object is remotely launched only once so that it corresponds to one process.
    Pay attention that one WorkerDict object may init multiple roles of user-defined worker.

    Args:
        role_to_worker_cls(dict): Dict of Role enum and user-defined worker class .
            Each role of worker only has ONE instance in WorkerDict.
    '''
    def __init__(self, role_to_worker_cls):
        self._set_device_id()
        self.init_status = False
        # The worker instance is instantiated here.
        # self.worker_dict is a dict of Role enum and user-defined worker instance.
        # TODO ZPaC: support to pass args and kwargs for worker class.
        self.worker_dict = {role: worker_cls() for role, worker_cls in role_to_worker_cls.items()}
        self.init_status = True


    def get_init_status(self):
        return self.init_status


    def get_role_instance(self, role):
        if role not in self.worker_dict:
            raise ValueError(f"Worker {role} is not instantiated in this worker dict.")
        return self.worker_dict[role]


    def call_worker_method(self, role, method_name, *args, **kwargs):
        '''
        Call method of worker with specified role. 
        '''
        if role not in self.worker_dict:
            raise ValueError(f"Worker {role} is not instantiated in this worker dict.")
        return getattr(self.worker_dict[role], method_name)(*args, **kwargs)


    def _set_device_id(self):
        # Ray has already adapt for Ascend by 'NPU' key.
        # This is assigned by option's 'resource' key when spawn this actor.
        device_id = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
        print("Worker Dict device_id: ", device_id, flush=True)
        ms.set_device(device_target="Ascend", device_id=device_id)


class WorkerGroup(ABC):
    '''
    This class is instantiated in driver process. It's the proxy of remote workers.
    It contains workers with the same role, which are distributed across multiple processes.
    Args:
        role(str): Role of this worker group. All remote workers with role are accessed by this class.
        worker_dict_actor_list(list[]): 
    '''
    def __init__(self, role, worker_dict_actor_list):
        self.role = role
        self.worker_dict_actor_list = worker_dict_actor_list


    def __getattr__(self, method_name):
        def wrapper(*args, **kwargs):
            # TODO ZPaC: Add dispatch and collect strategy and decorator processing here. 
            # Currently gather all outputs of remote workers and return to driver process.
            results = [ray.get(worker_dict_actor.call_worker_method.remote(self.role, method_name, *args, **kwargs))
                       for worker_dict_actor in self.worker_dict_actor_list]
            return results
        return wrapper


class RayWorkerFactory(ABC):
    '''
    This class manage resources and remote workers.
    It spawns remote WorkerDict and scheduler resources.
    '''
    def __init__(self, pool_id_to_role_list={}, pool_id_to_nproc_list={}):
        if len(pool_id_to_role_list) != len(pool_id_to_role_list):
            raise ValueError("Input 0 and input 1's pool sizes are different.")
        # Role to WorkerGroup object map.
        self.worker_groups = {}
        # Create resource pool and spawn remote actors.
        # Step 1: Initialize resource pool.
        # Step 2: Spawn remote WorkerDict actors.
        pass


    def _spawn_scheduler(self, world_size, pg, name_prefix):
        scheduler_name = f"{name_prefix}_scheduler"  # e.g. WorkerDict_global_pool_1_2_scheduler
        store_center_name = f"{name_prefix}_store_center"
        scheduler_actor = create_scheduler(name=scheduler_name, world_size=world_size, placement_group=pg,
                                           store_center_name=store_center_name)
        assert scheduler_actor is not None, f"failed to create scheduler_actor: {scheduler_name}"
        scheduler_actor.get_status.remote()


        store_center_actor = None
        for _ in range(30):
            if store_center_name not in list_named_actors():
                time.sleep(1)
            else:
                store_center_actor = ray.get_actor(store_center_name)
                break
        assert store_center_actor is not None, f"failed to get store_center_actor: {store_center_name} in {list_named_actors(all_namespaces=True)}"
        rank_zero_info = ray.get(store_center_actor.get_rank_zero_info.remote())
        ms_sched_host, ms_sched_port = rank_zero_info['MS_SCHED_HOST'], rank_zero_info['MS_SCHED_PORT']
        print(f"MS_SCHED_HOST: {ms_sched_host}, MS_SCHED_PORT: {ms_sched_port} for pool {name_prefix}", flush=True)

        return ms_sched_host, ms_sched_port
    

    def _spawn_worker(self, role_to_worker_cls, ms_sched_host, ms_sched_port, world_size, local_rank, pg, name_prefix):
        remote_cls = ray.remote(WorkerDict)
        # We pass in environment variable at option so that user-defined Worker can use environment variable to set.
        env_vars = {
            'WORLD_SIZE': str(world_size),
            "MS_ROLE": "MS_WORKER",
            "MS_WORKER_NUM": str(world_size),
            "MS_SCHED_HOST": ms_sched_host,
            "MS_SCHED_PORT": ms_sched_port,
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1"
        }

        name = f"{name_prefix}:{local_rank}"  # e.g. WorkerDict_global_pool_1_2:5
        options = self._prepare_options(placement_group=pg,
                                        placement_group_bundle_idx=local_rank + 1,
                                        env_vars=env_vars,
                                        worker_name=name)
        # Spawn remote WorkerDict actor(non-blocking).
        worker_dict_actor = remote_cls.options(**options).remote(role_to_worker_cls)
        return worker_dict_actor

    def _prepare_options(self, placement_group, placement_group_bundle_idx, env_vars, worker_name):
        options = {
            "scheduling_strategy":
                PlacementGroupSchedulingStrategy(placement_group=placement_group,
                                                 placement_group_bundle_index=placement_group_bundle_idx),
            'runtime_env': env_vars,
            'name': worker_name,
            "resources": {
                "NPU": 1
            },
            'lifetime': 'detached'
        }

        print("prepare options returns: ", options, flush=True)
        return options

    def spawn_colocated_workers(self, pool_id, nproc_list, role_to_worker_cls):
        '''
        Spawn remote WorkerDict actor encapsulating multiple role of user-defined workers.
        They are colocated on the same resource pool, which is initialized by inputs 'pool_id' and 'nproc_list'.
        '''
        # For each pool:
        # 1. There are sum(nproc_list) processes and same number of WorkerDict objects.
        # 2. The placement group number is the same as node number(length of nproc_list).
        # 3. Only one scheduler is remotely spawned.
        self.worker_dict_actor_list = []
        resource_pool = RayResourcePool(pool_id, nproc_list)
        pgs = resource_pool.create_placement_groups()
        ms_sched_host = None
        ms_sched_port = None
        for node_rank, local_world_size in enumerate(nproc_list):
            pg = pgs[node_rank]
            # Spawn scheduler of this resource pool on the first node.
            if node_rank == 0:
                ms_sched_host, ms_sched_port = self._spawn_scheduler(local_world_size, pg, f"Scheduler_{pool_id}")
            name_prefix = f"WorkerDict_{pool_id}_{node_rank}"
            for local_rank in range(local_world_size):
                worker_dict_actor = self._spawn_worker(role_to_worker_cls, # WorkerDict input.
                                                       ms_sched_host, ms_sched_port, local_world_size, local_rank, # Dynamic cluster envs.
                                                       pg, name_prefix # Ray args.
                                                       )
                self.worker_dict_actor_list.append(worker_dict_actor)
            
        # Wait all WorkerDict actors to be initialized, that's to say, all 
        # user-defined worker objects are instantiated(check WorkerDict.__init__).
        ray.get([worker_dict_actor.get_init_status.remote()
                for worker_dict_actor in self.worker_dict_actor_list])
            
        # For each role, we create a WorkerGroup as a PROXY worker locally,
        # which can be invoked in driver process and control remote user-defined workers.
        self.proxy_workers = {}
        for role, _ in role_to_worker_cls.items():
            self.proxy_workers[role] = WorkerGroup(role, self.worker_dict_actor_list)
        return self.proxy_workers