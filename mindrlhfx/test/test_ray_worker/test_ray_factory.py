from mindrlhfx.ray_wrapper import RayWorkerFactory, RayWorker

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
        "A": WorkerA(),
        "B": WorkerB()
    }
    factory = RayWorkerFactory()
    ray_workers = factory._spawn_worker(worker_dict)
    print("ray workers: ", ray_workers, flush=True)
    assert isinstance(ray_workers["A"], RayWorker)
    assert isinstance(ray_workers["B"], RayWorker)
    assert ray_workers["A"].foo() == "foo"
    assert ray_workers["B"].bar() == "bar"

test_ray_worker_factory()