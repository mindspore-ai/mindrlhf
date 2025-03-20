from mindrlhfx.ray.ray_wrapper import RayWorkerFactory, WorkerGroup

def test_ray_worker_factory():
    class WorkerA():
        def __init__(self):
            pass
        def foo(self):
            return "foo"

    class WorkerB():
        def __init__(self):
            pass
        def bar(self):
            return "bar"
    
    worker_dict = {
        "A": WorkerA,
        "B": WorkerB
    }
    factory = RayWorkerFactory()
    ray_worker_groups = factory.spawn_colocated_workers("pool", [4], worker_dict)
    print("ray workers: ", ray_worker_groups, flush=True)
    assert isinstance(ray_worker_groups["A"], WorkerGroup)
    assert isinstance(ray_worker_groups["B"], WorkerGroup)
    assert ray_worker_groups["A"].foo() == ["foo", "foo", "foo", "foo"]
    assert ray_worker_groups["B"].bar() == ["bar", "bar", "bar", "bar"]

test_ray_worker_factory()