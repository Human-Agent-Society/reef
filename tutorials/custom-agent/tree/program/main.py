# /// script
# requires-python = ">=3.12"
# dependencies = ["langgraph==1.2.19", "langchain-openai==1.6.0", "httpx==0.28.1"]
# ///
"""A LangGraph calculator run by Reef's Python adapter.

The argument is the task instruction. The model binding is beside this file.
The final answer goes to answer.txt in the task's working directory.
"""

import json
import os
import sys
from pathlib import Path

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class SessionTransport(httpx.BaseTransport):
    """HTTP transport that saves this run's non-streaming model exchanges.

    Values: session_file is a Path to an append-only JSONL file whose parent
    exists. transport is the owned HTTP transport. One instance serves one
    synchronous graph run and closes its transport when the client closes.
    Purpose: ChatOpenAI sends requests through handle_request; Reef reads
    successful request bodies with their response body under ``response``.
    Headers, including credentials, are not written. Failed HTTP calls write
    only their status and remain failures in the model client.
    """

    def __init__(self, session_file: Path) -> None:
        if not isinstance(session_file, Path) or not session_file.parent.is_dir():
            raise ValueError("session_file must have an existing parent directory")
        self.session_file = session_file
        self.transport = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Forward a request unchanged and record its completed chat response."""
        response = self.transport.handle_request(request)
        response.read()
        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            if response.is_success:
                event = {**json.loads(request.content), "response": response.json()}
            else:
                event = {"type": "model/error", "status": response.status_code}
            with self.session_file.open("a", encoding="utf-8") as session:
                session.write(json.dumps(event, ensure_ascii=False) + "\n")
        return response

    def close(self) -> None:
        """Close the transport owned by this model client."""
        self.transport.close()


@tool
def multiply(left: int, right: int) -> int:
    """Return the product of two integers."""
    return left * right


def run(instruction: str) -> str:
    """Run a bounded graph and return its final text; raise on a failed call or empty answer."""
    if not instruction.strip():
        raise ValueError("the task instruction must not be empty")
    binding = json.loads(Path(__file__).with_name("model.json").read_text(encoding="utf-8"))
    if not isinstance(binding, dict) or binding.get("api") != "openai":
        raise ValueError("model.json must contain an openai model binding")
    for field in ("base_url", "api_key", "model"):
        if not isinstance(binding.get(field), str) or not binding[field].strip():
            raise ValueError(f"model.json needs a non-empty {field}")
    session_dir = Path(os.environ["REEF_PROGRAM_SESSION_DIR"])
    session_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(transport=SessionTransport(session_dir / "calls.jsonl")) as client:
        model = ChatOpenAI(
            model=binding["model"],
            base_url=binding["base_url"],
            api_key=binding["api_key"],
            http_client=client,
            use_responses_api=False,
            streaming=False,
            max_tokens=512,
            timeout=60,
            max_retries=0,
        ).bind_tools([multiply])

        def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            return {"messages": [model.invoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.add_node("tools", ToolNode([multiply]))
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", tools_condition)
        graph.add_edge("tools", "model")
        result = graph.compile().invoke(
            {
                "messages": [
                    SystemMessage(
                        content="Use the multiply tool for multiplication. Reply with only the final integer."
                    ),
                    HumanMessage(content=instruction),
                ]
            },
            config={"recursion_limit": 12},
        )
    answer = result["messages"][-1].content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("the graph returned no final text")
    return answer.strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: main.py <instruction>")
    answer = run(sys.argv[1])
    Path("answer.txt").write_text(answer + "\n", encoding="utf-8")
    print(answer)
