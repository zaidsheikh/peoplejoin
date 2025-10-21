# ruff: noqa: E402
import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse  # noqa: I001
import asyncio
import json
import logging
import re
from pathlib import Path

# Hack: temporarily override the autogen_ext.models._utils.parse_r1_content function
import autogen_ext.models._utils.parse_r1_content
from utils.utils import my_parse_r1_content, LLMUsageTracker, capture_stream_and_write_to_file, OrderedMessageFilterAgent
autogen_ext.models._utils.parse_r1_content.parse_r1_content = my_parse_r1_content

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents._message_filter_agent import (
    MessageFilterConfig,
    PerSourceFilter,
)
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow, Swarm
from autogen_agentchat.ui import Console
from autogen_core import EVENT_LOGGER_NAME
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

# import phoenix.otel
# tracer_provider = phoenix.otel.register(
#   project_name="peoplejoin-autogen-test",
#   endpoint="https://app.phoenix.arize.com/s/peoplejoin-autogen/v1/traces",
#   auto_instrument=True # Auto-instrument your app based on installed OI dependencies
# )


orchestrator_name = "orchestrator"


def initialize_agents(
    agents: dict,
    primary_agent_model_client: OpenAIChatCompletionClient,
    other_agents_model_client: OpenAIChatCompletionClient,
    enable_handoffs: bool
) -> None:
    # Create the orchestrator agent first
    all_agent_names = list(agents.keys())

    # All existing agents are now treated equally - no primary user distinction
    agent_descriptions = "\n".join(
        [f"{name}: {agents[name]['description']}" for name in all_agent_names]
    )

    # Instructions for the orchestrator agent
    orchestrator_instruction = (
        "You are the orchestrator agent responsible for coordinating with other agents to answer questions.\n"
        "When you receive a question, analyze what information might be needed and reach out to the appropriate agents.\n"
        "To contact an agent, start your message with 'Hello @AgentName' to get their attention.\n"
        "Make sure to include all relevant context in your message to them.\n"
        f"{'Always send your message first, then handoff to the appropriate agent. Always handoff to a single agent at a time. ' if enable_handoffs else ''}"
        "You can follow up with agents (do not forget to start your message with 'Hello @AgentName') if you need clarifications or additional information.\n"
        "Do not address multiple agents in the same message.\n"
        "If you need information from multiple agents, reach out to them one at a time.\n"
        "After you have gathered enough information from the agents, provide a final answer.\n"
        "Your final answer should start with 'Final Answer: <your answer>'.\n"
        "Here are the available agents and their descriptions:\n"
        f"{agent_descriptions}\n"
    )

    # Instructions for regular agents
    agent_instruction = (
        "an AI agent with access to specific documents and information.\n"
        "When contacted by the orchestrator or other agents, provide specific relevant information from your documents if available.\n"
        "If you do not have enough information to fully answer a question, let the requester know what information you need or what relevant information you do have.\n"
        f"{'Always send your message first, then handoff back to the orchestrator.' if enable_handoffs else ''}\n"
    )

    for agent_name, agent_info in agents.items():
        system_message = f"You are {agent_name}, {agent_instruction}"
        if agent_info.get("documents"):
            docs_summary = "\n".join([f"- {doc}" for doc in agent_info["documents"]])
            system_message += f"\n\nYour documents:\n{docs_summary}"

        handoffs: list[Handoff | str] = []
        if enable_handoffs:
            handoffs = [
                Handoff(
                    target=orchestrator_name,
                    name="handoff_to_orchestrator",
                    description="Handoff back to the orchestrator",
                )
            ]

        # Create the base assistant agent
        base_agent = AssistantAgent(
            name=f"inner_{agent_name}",
            description=agent_info.get("description"),
            system_message=system_message,
            model_client=other_agents_model_client,
            handoffs=handoffs if enable_handoffs else None,
        )

        # Wrap the agent in OrderedMessageFilterAgent to filter messages
        # Only allow messages from "user", orchestrator, and the agent itself
        message_filter = MessageFilterConfig(
            per_source=[
                PerSourceFilter(source="user"),  # All messages from user
                PerSourceFilter(source=orchestrator_name, position="last", count=1),  # all previous messages from orchestrator to this agent will still be included since the wrapped AssistantAgent keeps them in its internal context
                PerSourceFilter(source=f"inner_{agent_name}"),  # All messages from itself
                PerSourceFilter(source=agent_name),  # All messages from itself
            ]
        )

        agents[agent_name]["agent"] = OrderedMessageFilterAgent(
            name=agent_name,
            wrapped_agent=base_agent,
            filter=message_filter,
        )

    # Create the orchestrator agent
    orchestrator_handoffs: list[Handoff | str] = []
    if enable_handoffs:
        orchestrator_handoffs = [
            Handoff(
                target=agent_name,
                name=f"handoff_to_{agent_name}",
                description=f"Handoff to {agent_name}. {agents[agent_name]['description']}",
            )
            for agent_name in all_agent_names
        ]

    agents[orchestrator_name] = {
        "name": orchestrator_name,
        "description": "Coordinates with other agents to gather information and provide comprehensive answers",
        "documents": [],
        "is_orchestrator": True,
        "agent": AssistantAgent(
            name=orchestrator_name,
            description="Coordinates with other agents to gather information and provide comprehensive answers",
            system_message=orchestrator_instruction,
            model_client=primary_agent_model_client,
            handoffs=orchestrator_handoffs if enable_handoffs else None,
        )
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary_llm_model",
        type=str,
    )
    parser.add_argument(
        "--default_llm_model",
        type=str,
        default=os.environ.get("LLM_MODEL", "neulab/gpt-4.1-nano-2025-04-14"),
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.environ.get("LLM_API_KEY", "no_api_key_provided"),
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=os.environ.get("LLM_BASE_URL", "https://cmu.litellm.ai"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Directory to save outputs (default: outputs)",
    )
    parser.add_argument(
        "--groupchat_type",
        choices=["swarm", "graph"],
        default="graph",
        help="Type of group chat to use (default: graph)",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question to ask the agents",
    )
    parser.add_argument(
        "--datum_id",
        type=str,
        required=True,
        help="The datum_id of the question to ask the agents",
    )
    parser.add_argument(
        "--tenant_id",
        type=str,
        required=True,
        help="The tenant_id of the agent data: data_dir/<tenant_id>.json",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "peoplejoin-qa",
        help="Path to the tenant data directory (default: peoplejoin/data/peoplejoin-qa)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.primary_llm_model:
        args.primary_llm_model = args.default_llm_model

    print(f"Using LLM model: {args.primary_llm_model}, {args.default_llm_model}")
    print(f"LLM base URL: {args.llm_base_url}")
    print(f"Datum ID: {args.datum_id}")
    print(f"Tenant ID: {args.tenant_id}")

    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    llm_calls_log_path = args.output_dir / f"{args.datum_id}_llm_calls.jsonl"
    llm_usage = LLMUsageTracker(args.datum_id, log_file_path=llm_calls_log_path)
    logger.handlers = [llm_usage]

    # NOTE: currently previous thoughts are not included in the context sent to the LLM for completions
    # To change this behaviour, add the following code to the top of this script
    #
    # from autogen_ext.models.openai import _message_transform
    # _message_transform.single_assistant_transformer_funcs.append(
    #     lambda msg, _: {
    #         "content": f"<think>{msg.thought or ''}</think> {msg.content or ''}"
    #     }
    # )

    primary_agent_model_client = OpenAIChatCompletionClient(
        model=args.primary_llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        temperature=0.6, # Qwen/Qwen3-235B-A22B-Thinking-2507-FP8
        top_p=0.95, # Qwen/Qwen3-235B-A22B-Thinking-2507-FP8
        model_info={
            # "family": ModelFamily.GPT_41,
            "family": ModelFamily.R1, # TODO hack: using R1 for qwen3 to take advantage of parse_r1_content()
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )

    other_agents_model_client = OpenAIChatCompletionClient(
        model=args.default_llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        temperature=0.7, # Qwen/Qwen3-235B-A22B-Instruct-2507
        top_p=0.8, # Qwen/Qwen3-235B-A22B-Instruct-2507
        model_info={
            # "family": ModelFamily.GPT_41,
            "family": ModelFamily.UNKNOWN,
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
            "multiple_system_messages": True,
        },
    )
    # load users, user descriptions, user documents from example_data/movie_1.json
    agents = {}
    with open(args.data_dir / f"{args.tenant_id}.json") as f:
        data = json.load(f)
        users = [d["user_id"] for d in data.get("users", [])]
        # Note: ignoring primary_user designation - all users are treated equally now
        user_descriptions = data.get("user_id_to_descriptions_templated", {})
        user_documents = data.get("user_id_to_documents", {})
        for user in users:
            agents[user] = {
                "name": user,
                "documents": user_documents.get(user, []),
                "description": re.sub(
                    pattern=r"\bUser\b",
                    repl="Agent",
                    string=user_descriptions.get(user, ""),
                    count=1,
                    flags=re.IGNORECASE,
                ),
            }

    initialize_agents(
        agents, primary_agent_model_client, other_agents_model_client, enable_handoffs=args.groupchat_type == "swarm"
    )


    # load question
    # with open(args.data_dir / "experiment_0.json") as f:
    #     data = json.load(f)
    #     # TODO: add the original question to the config file
    #     question = (
    #         data["participant_id_to_descriptions"][primary_user]
    #         .split("She is interested in knowing")[-1]
    #         .split("which may be available in her documents")[0]
    #         .strip()
    #     )

    termination_condition = MaxMessageTermination(30) | TextMentionTermination(
        text="Final Answer:",
        sources=[orchestrator_name],
    )

    if args.groupchat_type == "swarm":
        # setup a swarm group chat with the orchestrator as the initial agent
        # Extract the agent objects and ensure orchestrator is first
        agent_list = []
        agent_list.append(agents[orchestrator_name]["agent"])

        # Add all other user agents
        for agent_name, agent_info in agents.items():
            if agent_name != orchestrator_name:
                agent_list.append(agent_info["agent"])
        team = Swarm(
            participants=agent_list, termination_condition=termination_condition
        )

    elif args.groupchat_type == "graph":
        # setup a graph-based group chat with the orchestrator as the initial agent
        graph_builder = DiGraphBuilder()
        for agent_info in agents.values():
            graph_builder.add_node(agent_info["agent"])
        for agent_name, agent_info in agents.items():
            if agent_name == orchestrator_name:
                continue

            def make_trigger_condition(agent_name: str):
                def trigger_condition(message: BaseChatMessage) -> bool:
                    pattern = re.compile(
                        rf"Hello @{re.escape(agent_name)}($|[^a-zA-Z0-9_-])", re.IGNORECASE
                    )
                    return bool(pattern.search(message.to_model_text()))

                return trigger_condition

            graph_builder.add_edge(
                source=agents[orchestrator_name]["agent"],
                target=agent_info["agent"],
                condition=make_trigger_condition(agent_name),
            )
            graph_builder.add_edge(
                source=agent_info["agent"],
                target=agents[orchestrator_name]["agent"],
                activation_group=agent_name,
            )
        terminal_agent = AssistantAgent(
            name="terminal",
            model_client=other_agents_model_client,
            system_message="You are a terminal node. Simply acknowledge receipt of the final answer.",
        )

        def final_trigger_condition(message: BaseChatMessage) -> bool:
            return "final answer:" in message.to_model_text().lower()

        graph_builder.add_node(terminal_agent)
        graph_builder.add_edge(
            source=agents[orchestrator_name]["agent"],
            target=terminal_agent,
            condition=final_trigger_condition,
        )
        graph_builder.set_entry_point(agents[orchestrator_name]["agent"])
        graph = graph_builder.build()
        all_participants = [agents[agent]["agent"] for agent in agents] + [
            terminal_agent
        ]
        team = GraphFlow(
            graph=graph,
            participants=all_participants,
            termination_condition=termination_condition,
        )

    else:
        raise ValueError(f"Unsupported groupchat_type: {args.groupchat_type}")

    stream = team.run_stream(task=args.question)
    autogen_messages_log_path = args.output_dir / f"{args.datum_id}_autogen_messages.jsonl"
    wrapped_stream = await capture_stream_and_write_to_file(stream, autogen_messages_log_path)
    await Console(wrapped_stream)

    # Print token usage summary
    print("\nToken usage summary:")
    print(f"Prompt tokens: {llm_usage.prompt_tokens}")
    print(f"Completion tokens: {llm_usage.completion_tokens}")
    print(f"Total tokens: {llm_usage.tokens}")
    print(f"LLM calls logged to: {llm_calls_log_path}")
    print(f"Autogen messages logged to: {autogen_messages_log_path}")


if __name__ == "__main__":
    asyncio.run(main())
