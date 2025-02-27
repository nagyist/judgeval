

from judgeval.constants import ROOT_API
import requests
import asyncio
from openai import OpenAI
from judgeval.common.tracer import Tracer, wrap
import os 
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), project_name="test")

@judgment.observe(span_type="llm_call")
async def llm_call(prompt: str):
    client = wrap(OpenAI())
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    retrieval_context = ["The capital of France is Berlin"]
    trace = judgment.get_current_trace().async_evaluate(
            scorers=[FaithfulnessScorer(threshold=0.9), AnswerRelevancyScorer(threshold=0.9)],
            actual_output=response.choices[0].message.content,
            input=prompt,
            retrieval_context=retrieval_context, # Fails because of retrieval context
            model="gpt-4o"
        )
    print(trace)
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = asyncio.run(llm_call(prompt))
        # trace_id, TraceSave = trace.save()
        # print(TraceSave)
        # Purposely fail the score
    
    # response = requests.post(
    #     url=f"{ROOT_API}/datasets/store_trace_info/",
    #     json={
    #         "judgment_api_key": "ff6c3e63-604d-421a-beff-bb6f7b94f89b",
    #         "dataset_alias": "temp",
    #         "trace_data": TraceSave
    #     }
    # )
    # response = requests.get(
    #     url=f"{ROOT_API}/datasets/get_trace_info/",
    #     json={
    #         "judgment_api_key": "ff6c3e63-604d-421a-beff-bb6f7b94f89b",
    #         "dataset_alias": "temp",
    #         "trace_id": trace_id
    #     }
    # )
    # print(response.json())
    # response = requests.delete(
    #     url=f"{ROOT_API}/datasets/delete_trace/",
    #     json={
    #         "judgment_api_key": "ff6c3e63-604d-421a-beff-bb6f7b94f89b",
    #         "dataset_alias": "temp",
    #         "trace_id": trace_id
    #     }
    # )


