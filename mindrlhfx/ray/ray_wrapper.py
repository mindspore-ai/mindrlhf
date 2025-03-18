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
from mindrlhfx.ray.resource import RayResourcePool


# TODO: 是否需要继承其他类
class WorkerDict(ABC):
    def __init__(self, worker_dict):
        self._data = worker_dict
        self._set_device_id()
        self.init_status = False
        for k, c in self._data.items():
            # init
            self._data[k] = c()
        self.init_status = True

    def get(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"WorkerDict({repr(self._data)})"

    def _set_device_id(self):
        device_id = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
        print("Worker Dict device_id: ", device_id, flush=True)
        ms.set_device(device_target="Ascend", device_id=device_id)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get_init_status(self):
        return self.init_status


# TODO:  We expect this class could inherit from Worker class so that
# attrs of user-defined Worker can be accessed.
class RayWorker(ABC):
    '''
    This class wraps around user-defined worker when using MPMD paradigm.
    It will register to RayWorkerFactory.
    '''
    def __init__(self, key: str, worker_dict_actor, *args, **kwargs):
        self.key = key
        self.actor_list = worker_dict_actor

    def init_resource(self):
        pass

    def __getattr__(self, method_name):
        methods = []
        for actor in self.actor_list:
            cls_keys = ray.get(actor.keys.remote())
            if self.key not in cls_keys:
                raise AttributeError(f"Worker '{self.key}' not found in worker_dict")
            target_worker = ray.get(actor.get.remote(self.key))
            method = getattr(target_worker, method_name)
            if not hasattr(target_worker, method_name):
                raise AttributeError(f"Method '{method_name}' not found in Worker '{self.key}'")
            methods.append(method)
        def wrapper(*args, **kwargs):
            res = []
            for method in methods:
                res.append(method(*args, **kwargs))
            return res
        return wrapper


class RayWorkerFactory(ABC):
    '''
    This class manage all ray workers. It spawns remote ray actors and schedule resources.
    '''
    def __init__(self):
        pass

    def _spawn_worker(self, local_rank: int):
        remote_cls = ray.remote(WorkerDict)
        # we pass in environment variable at option so that Worker can use environment variable to set
        env_vars = {
            'WORLD_SIZE': str(self._world_size),
            "MS_ROLE": "MS_WORKER",
            "MS_WORKER_NUM": str(self._world_size),
            "MS_SCHED_HOST": self._ms_sched_host,
            "MS_SCHED_PORT": self._ms_sched_port,
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1"
        }

        name = f"{self.name_prefix}:{local_rank}"  # e.g. WorkerDict_global_pool_1_2:5
        # prepare options
        options = self._prepare_options(placement_group=self.pg,
                                        placement_group_bundle_idx=local_rank+1,
                                        env_vars=env_vars,
                                        worker_name=name)
        # create a ray actor
        worker_dict_actor = remote_cls.options(**options).remote(self.worker_dict)
        return worker_dict_actor

    def _create_scheduler(self):
        self._scheduler_name = f"{self.name_prefix}_scheduler"  # e.g. WorkerDict_global_pool_1_2_scheduler
        self._store_center_name = f"{self.name_prefix}_store_center"
        scheduler_actor = create_scheduler(name=self._scheduler_name, world_size=self._world_size, placement_group=self.pg,
                                           store_center_name=self._store_center_name)
        assert scheduler_actor is not None, f"failed to create scheduler_actor: {self._scheduler_name}"
        scheduler_actor.get_status.remote()


        store_center_actor = None
        for _ in range(30):
            if self._store_center_name not in list_named_actors():
                time.sleep(1)
            else:
                store_center_actor = ray.get_actor(self._store_center_name)
                break
        assert store_center_actor is not None, f"failed to get store_center_actor: {self._store_center_name} in {list_named_actors(all_namespaces=True)}"
        rank_zero_info = ray.get(store_center_actor.get_rank_zero_info.remote())
        self._ms_sched_host, self._ms_sched_port = rank_zero_info['MS_SCHED_HOST'], rank_zero_info['MS_SCHED_PORT']
        print(f"ms_sched_host: {self._ms_sched_host}, ms_sched_port: {self._ms_sched_port}", flush=True)
    
    def _prepare_options(self, placement_group, placement_group_bundle_idx, env_vars, worker_name):
        options = {
            "scheduling_strategy":
                PlacementGroupSchedulingStrategy(placement_group=placement_group,
                                                 placement_group_bundle_index=placement_group_bundle_idx),
            'runtime_env': {
                'env_vars': env_vars
            },
            'name': worker_name,
            "resources": {
                "NPU": 1
            },
            'lifetime': 'detached'
        }

        print("prepare options returns: ", options, flush=True)
        return options
    
    def _init_actors(self, ray_actors):
        actor_refs = []
        for actor in ray_actors:
            actor_refs.append(actor.get_init_status.remote())
        status = ray.get(actor_refs)
        print("init actors status: ", status, flush=True)

    def process_workers(self, resource_pool: RayResourcePool, worker_dict: WorkerDict):
        '''
        输入为Dict的集合, 其中每个Dict代表一个共部署的Worker集合
        输出为Dict的集合, 其中key和输入的Dict一致, value为对应的ray worker
        '''
        ray_actors = []
        pgs = resource_pool.create_placement_groups(strategy="STRICT_PACK")
        self._world_size = resource_pool.world_size
        self.worker_dict = worker_dict
        self.pool_id = resource_pool.pool_id

        rank = -1
        for pg_idx, local_world_size in enumerate(resource_pool.nproc_list):
            self.pg = pgs[pg_idx]
            self.name_prefix = f"WorkerDict_{self.pool_id}_{pg_idx}"
            for local_rank in range(local_world_size):
                rank += 1
                if rank == 0:
                    self._create_scheduler()
                ray_actors.append(self._spawn_worker(local_rank=local_rank))
        print(f"list_named_actors: ", list_named_actors(), flush=True)
        self._init_actors(ray_actors)
        ray_worker_dict = {}
        for key in worker_dict.keys():
            ray_worker_dict[key] = RayWorker(key, ray_actors)
        return ray_worker_dict
