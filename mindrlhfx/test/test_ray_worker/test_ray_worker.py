import ray

from mindspore import mint
import mindspore.context as context

from mindrlhfx.ray_wrapper import RayWorkerFactory
from mindrlhfx.resource import ResourcePoolManager, Role


class WorkerA():
    def __init__(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
        mint.distributed.init_process_group(
            backend="hccl"
        )
    
    def foo(self):
        return "foo"


class WorkerB():
    def __init__(self):
        pass
    
    def bar(self):
        return "bar"


class WorkerC():
    def __init__(self):
        context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
        mint.distributed.init_process_group(
            backend="hccl"
        )
    
    def foo(self):
        return "c"


# init ray
if not ray.is_initialized():
    ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN',
                                       'ASCEND_GLOBAL_LOG_LEVEL': '1', 'HCCL_ENTRY_LOG_ENABLE': '1',
                                       'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES': '1'}})

worker_dict_1 = {
    "Actor": WorkerA,
    "RefPolicy": WorkerB
}

worker_dict_2 = {
    "Critic": WorkerC
}

# create resource pool
rpm = ResourcePoolManager("./test_ray_worker.yaml")
rpm.create_resource_pool()
rp_1 = rpm.get_resource_pool(role=Role.Actor)
rp_2 = rpm.get_resource_pool(role=Role.Critic)

factory = RayWorkerFactory()
ray_workers_1 = factory.process_workers(resource_pool=rp_1, worker_dict=worker_dict_1)
assert ray_workers_1["Actor"].foo() == ['foo', 'foo']
assert ray_workers_1["RefPolicy"].bar() == ['bar', 'bar']

ray_workers_2 = factory.process_workers(resource_pool=rp_2, worker_dict=worker_dict_2)
assert ray_workers_2["Critic"].foo() == ['c', 'c']
