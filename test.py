from core import observe_daemon, print_graph, tag
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

@tag
def fibonacci(n: int) -> int:
    """
    A simple Fibonacci function to demonstrate tracing.
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@tag
def sum(a: int, *nums: int, **kwargs: int) -> int:
    """
    A simple sum function to demonstrate tracing.
    """
    return a + sum(*nums, **kwargs) if nums else a


class FibonacciManager():
    """
    A class to manage Fibonacci calculations.
    """
    def __init__(self, id: str):
        self.id = id
    @tag
    def calculate(self, n: int) -> int:
        """
        Calculate Fibonacci number with tracing.
        """
        return self._fib(n)

    @tag
    def _fib(self, n: int) -> int:
        """
        Internal Fibonacci calculation method.
        """
        if n <= 1:
            return n
        return self._fib(n - 1) + self._fib(n - 2)
    

@tag
def main():
    fibonacci_manager = FibonacciManager("fib_manager")
    fibonacci_manager.calculate(5)


with observe_daemon(deep_tracing=False):
    main()

print_graph()
