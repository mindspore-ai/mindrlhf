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
import os
import yaml
from enum import Enum
from collections import OrderedDict

import ray
from ray.util.placement_group import placement_group


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    SftInfer = 7
    SftTrain = 8

    @staticmethod
    def from_string(s):
        for r in Role:
            if r.name == s or r.value == s:
                return r
        raise ValueError(f"Invalid role name or value: {s}")


class RoleList():
    """Model role list. Some models could be depolyed to same resource pool"""
    def __init__(self, role_list: list[Role]):
        self.role_list = role_list


class RayResourcePool():
    """Resource pool of ray."""
    def __init__(self, pool_id: str, nproc_list: list[int], detached=False):
        self.pool_id = pool_id
        self.nproc_list = nproc_list
        self.detached = detached
        self.pgs = None
        self.world_size = sum(nproc_list)
    
    def create_placement_groups(self, strategy="STRICT_PACK"):
        if self.pgs is not None:
            print("placement_groups have already been created, don't create again....", flush=True)
            return self.pgs
        pg_name_prefix = f"{self.pool_id}_group_{'_'.join([str(count) for count in self.nproc_list])}:"
        pg_scheme = [[]] * len(self.nproc_list)
        for i, process_count in enumerate(self.nproc_list):
            pg_scheme[i].append({"CPU": 1}) # for scheduler
            pg_scheme[i].extend([{"CPU": 1, "NPU": 1} for _ in range(process_count)])

        lifetime = 'detached' if self.detached else None

        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        ray.get([pg.ready() for pg in pgs])
        self.pgs = pgs
        return self.pgs

    def get_placement_groups(self):
        if self.pgs is None:
            print("placement_groups are not created, start creating placement groups....", flush=True)
            self.create_placement_groups()
        
        return self.pgs

    
class ResourcePoolManager:
    """
    Manage model to hardware resources.
    We should support to configure and parse mapping of models and resources in yaml,
    so that users don't need to initialize resource pools manually.
    The yaml is configured as:
    resource_pool:
        global_pool_1:
            nproc_list: [8, 8, 8, 8]
            role_list: [Actor, RefPolicy]
        global_pool_2:
            nproc_list: [4]
            role_list: [Critic]

    Converted to:
    pool_id_to_nproc_list:
    {
        'global_pool_1': [8, 8, 8, 8]
        'global_pool_2': [4]
    }
    role_to_pool_id:
    {
        Role.Actor: 'glabal_pool_1',
        Role.RefPolicy: 'glabal_pool_1',
        Role.Critic: 'glabal_pool_2',
    }
    """
    def __init__(self, cfg_path):
        # dict[str, list[int]]
        self.pool_id_to_nproc_list = {}
        # dict[Role, str]
        self.role_to_pool_id = {}
        # dict[str, RayResourcePool]
        self.pool_id_to_resoure_pool = {}

        self._read_cfg(cfg_path)
        print(f"self.pool_id_to_nproc_list: {self.pool_id_to_nproc_list}, self.role_to_pool_id: {self.role_to_pool_id}", flush=True)

    def _read_cfg(self, cfg_path):
        """ read pool info from yaml """
        filepath = os.path.realpath(cfg_path)
        with open(filepath, encoding="utf-8") as fp:
            cfg_dict = yaml.safe_load(fp)
            cfg_dict = OrderedDict(sorted(cfg_dict.items()))

        for pool_id, pool_info in cfg_dict['resource_pool'].items():
            self.pool_id_to_nproc_list[pool_id] = pool_info['nproc_list']
            for role in pool_info['role_list']:
                self.role_to_pool_id[Role.from_string(role)] = pool_id
 
    def create_resource_pool(self):
        """Create resource pool for specified c"""
        for pool_id, nproc_list in self.pool_id_to_nproc_list.items():
            resource_pool = RayResourcePool(pool_id=pool_id, nproc_list=nproc_list)
            self.pool_id_to_resoure_pool[pool_id] = resource_pool

    def get_resource_pool(self, role: Role):
        """Get the resource pool of the worker_cls"""
        return self.pool_id_to_resoure_pool[self.role_to_pool_id[role]]
