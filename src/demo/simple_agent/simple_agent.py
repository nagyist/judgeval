from typing import List, Dict, Any
import random
from pydantic import BaseModel
from judgeval.common.tracer import Tracer, wrap
from judgeval import JudgmentClient
import os

judgment = Tracer(project_name="random_tools_agent")

class RandomToolAgent(BaseModel):
    name: str = "Random Tool Agent"
    
    @judgment.observe(span_type="tool")
    def tool_1(self) -> str:
        """First example tool"""
        return f"{self.name} used tool_1: Generated random number {random.randint(1, 100)}"
    
    @judgment.observe(span_type="tool")
    def tool_2(self) -> str:
        """Second example tool"""
        return f"{self.name} used tool_2: Generated random string {random.choice(['A', 'B', 'C'])}"
    
    @judgment.observe(span_type="tool")
    def tool_3(self) -> str:
        """Third example tool"""
        return f"{self.name} used tool_3: Generated random boolean {random.choice([True, False])}"
    
    @judgment.observe(span_type="function")
    def run_agent(self, prompt: str) -> Dict[str, Any]:
        """
        Randomly selects and uses tools based on the prompt.
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            Dict: Contains the response and tools used
        """
        # Define available tools
        tools = {
            "tool_1": self.tool_1,
            "tool_2": self.tool_2,
            "tool_3": self.tool_3
        }
        
        # Use tools in order: tool_1, tool_2, tool_3
        selected_tools = ["tool_1", "tool_2", "tool_3"]
        
        # Use the selected tools
        responses = []
        for tool_name in selected_tools:
            tool_response = tools[tool_name]()
            responses.append(tool_response)
        
        # Combine responses
        final_response = "\n".join(responses)
        
        return final_response

# Example usage
if __name__ == "__main__":
    # Create agent
    agent = RandomToolAgent()
    
    # Test the agent
    judgment = JudgmentClient()
    from judgeval.data import Example
    from judgeval.scorers import ToolOrderScorer

    example = Example(
        input={"prompt": "Do something random"},
        expected_tools=[
            {"tool_name": "tool_1", "parameters": {'self': {'name': 'Random Tool Agent'}}},
            {"tool_name": "tool_2", "parameters": {'self': {'name': 'Random Tool Agent'}}},
            {"tool_name": "tool_3", "parameters": {'self': {'name': 'Random Tool Agent'}}}
        ]
    )

    wrong_order = Example(
        input={"prompt": "Do something random"},
        expected_tools=[
            {"tool_name": "tool_3", "parameters": {'self': {'name': 'Not random Agent'}}   },
            {"tool_name": "tool_2", "parameters": {'self': {'name': 'Not random Agent'}}},
            {"tool_name": "tool_1", "parameters": {'self': {'name': 'Not random Agent'}}}
        ]
    )
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "tests.yaml")

    judgment.assert_test(
        # examples=[example, wrong_order],  # Use examples for Example objects
        test_file=yaml_path,
        scorers=[ToolOrderScorer()],
        function=agent.run_agent,
        override=True
    )

    