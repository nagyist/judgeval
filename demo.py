from typing import List
from judgeval.tracer import Tracer

# Initialize the tracer with your project name
judgment = Tracer(project_name="demo")

class Foo:
    @judgment.observe(span_type="class")
    @staticmethod
    def bar():
        return "Hello from Foo.bar!"
    
    @judgment.observe(span_type="instance")
    def baz(self):
        return "Hello from Foo.baz!"
    



# Use the @judgment.observe decorator to trace the tool call
@judgment.observe(span_type="tool")
def my_tool():
    return "Hello world!"

# Use the @judgment.observe decorator to trace the function
@judgment.observe(span_type="function")
def sample_function():
    tool_called = my_tool()
    message = "Called my_tool() and got: " + tool_called

    foo_instance = Foo()
    class_method_result = Foo.bar()
    instance_method_result = foo_instance.baz()

    message += f"\nClass method Foo.bar() returned: {class_method_result}"
    message += f"\nInstance method foo_instance.baz() returned: {instance_method_result}"
    return message

if __name__ == "__main__":
    res = sample_function()
    print(res)
