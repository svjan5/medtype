def main():
    from medtype_serving.server import MedTypeServer
    from medtype_serving.server.helper import get_run_args
    with MedTypeServer(get_run_args()) as server:
        server.join()


def benchmark():
    from medtype_serving.server.benchmark import run_benchmark
    from medtype_serving.server.helper import get_run_args, get_benchmark_parser
    args = get_run_args(get_benchmark_parser)
    run_benchmark(args)


def terminate():
    from medtype_serving.server import MedTypeServer
    from medtype_serving.server.helper import get_run_args, get_shutdown_parser
    args = get_run_args(get_shutdown_parser)
    MedTypeServer.shutdown(args)
