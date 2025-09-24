import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

# Hack: temporarily override the autogen_ext.models._utils.parse_r1_content function
import autogen_ext.models._utils.parse_r1_content
from utils.utils import my_parse_r1_content, LLMUsageTracker, OrderedMessageFilterAgent
autogen_ext.models._utils.parse_r1_content.parse_r1_content = my_parse_r1_content

from autogen_agentchat.agents import AssistantAgent
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




def initialize_agents(
    agents: dict,
    primary_agent_model_client: OpenAIChatCompletionClient,
    other_agents_model_client: OpenAIChatCompletionClient,
    enable_handoffs: bool
) -> None:
    all_agent_names = list(agents.keys())
    primary_agent_name = next(
        name for name, info in agents.items() if info.get("is_primary_user")
    )
    other_agent_names = [name for name in all_agent_names if name != primary_agent_name]

    other_agent_descriptions = "\n".join(
        [f"{name}: {agents[name]['description']}" for name in other_agent_names]
    )

    other_agent_instruction = (
        "If you do not have enough information to answer the question, let the requester know what information you need to answer the question or what information you have that might be relevant.\n"
        f"{'Always send your message first, then handoff back to the requester.' if enable_handoffs else ''}\n"
    )
    primary_agent_instruction = (
        "Otherwise if you do not have enough information to fully answer the question,\n"
        "reach out to other agents for any additional information based on their descriptions.\n"
        "To do so, start your message with 'Hello @AgentName' to get their attention.\n"
        "Make sure to include all relevant context in your message to them.\n"
        f"{'Always send your message first, then handoff to appropriate agent. Always handoff to a single agent at a time. ' if enable_handoffs else ''}"
        "You can respond back to the agent (do not forget to start your message with 'Hello @AgentName') if the answer is not complete or you need clarifications.\n"
        "Do not address multiple agents in the same message.\n"
        "If you need to get information from multiple agents, reach out to them one at a time.\n"
        "After you have gathered enough information, provide a final answer.\n"
        "Your final answer should start with 'Final Answer: <your answer>'.\n"
        "Here are the other agents' descriptions:\n"
        f"{other_agent_descriptions}\n"
    )

    for agent_name, agent_info in agents.items():
        system_message = (
            f"You are {agent_name}, an AI agent tasked with answering questions.\n"
            + "Provide specific relevant information if it is available in your documents.\n"
            + (
                primary_agent_instruction
                if agent_info.get("is_primary_user")
                else other_agent_instruction
            )
        )
        if agent_info.get("documents"):
            docs_summary = "\n".join([f"- {doc}" for doc in agent_info["documents"]])
            system_message += f"\n\nYour documents:\n{docs_summary}"

        handoffs: list[Handoff | str] = []
        if agent_info.get("is_primary_user"):
            handoffs = [
                Handoff(
                    target=other_agent,
                    name=f"handoff_to_{other_agent}",
                    description=f"Handoff to {other_agent}. {agents[other_agent]['description']}",
                )
                for other_agent in other_agent_names
            ]
        else:
            handoffs = [
                Handoff(
                    target=primary_agent_name,
                    name="handoff_to_requestor",
                    description="Handoff back to the requestor",
                )
            ]

        agents[agent_name]["agent"] = AssistantAgent(
            name=agent_name,
            description=agent_info.get("description"),
            system_message=system_message,
            model_client=primary_agent_model_client if agent_info.get("is_primary_user") else other_agents_model_client,
            handoffs=handoffs if enable_handoffs else None,
        )


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
        "--data_dir", type=Path, default=Path(__file__).parent / "example_data"
    )
    parser.add_argument(
        "--log_llm_calls",
        type=Path,
        default=Path("llm_calls.jsonl"),
        help="Path to log LLM calls as JSON lines (default: llm_calls.jsonl)",
    )
    parser.add_argument(
        "--groupchat_type",
        choices=["swarm", "graph"],
        default="graph",
        help="Type of group chat to use (default: graph)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if not args.primary_llm_model:
        args.parimary_llm_model = args.default_llm_model

    print(f"Using LLM model: {args.primary_llm_model}, {args.default_llm_model}")
    print(f"LLM base URL: {args.llm_base_url}")

    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    llm_usage = LLMUsageTracker(log_file_path=args.log_llm_calls)
    logger.handlers = [llm_usage]


    primary_agent_model_client = OpenAIChatCompletionClient(
        model=args.primary_llm_model,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
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
    with open(args.data_dir / "movie_1.json") as f:
        data = json.load(f)
        users = [d["user_id"] for d in data.get("users", [])]
        primary_user = data["primary_user"]["user_id"]
        user_descriptions = data.get("user_id_to_descriptions_templated", {})
        user_documents = data.get("user_id_to_documents", {})
        for user in users:
            agents[user] = {
                "name": user,
                "description": user_descriptions.get(user, ""),
                "documents": user_documents.get(user, []),
                "is_primary_user": user == primary_user,
            }

    initialize_agents(
        agents, primary_agent_model_client, other_agents_model_client, enable_handoffs=args.groupchat_type == "swarm"
    )

    # load question
    with open(args.data_dir / "experiment_0.json") as f:
        data = json.load(f)
        # TODO: add the original question to the config file
        question = (
            data["participant_id_to_descriptions"][primary_user]
            .split("She is interested in knowing")[-1]
            .split("which may be available in her documents")[0]
            .strip()
        )

    termination_condition = MaxMessageTermination(30) | TextMentionTermination(
        text="Final Answer:",
        sources=[primary_user],
    )

    if args.groupchat_type == "swarm":
        # setup a swarm group chat with the primary agent as the initial agent
        # Extract the agent objects and ensure primary agent is first
        agent_list = []
        agent_list.append(agents[primary_user]["agent"])

        # Add other agents
        for agent_info in agents.values():
            if not agent_info.get("is_primary_user"):
                agent_list.append(agent_info["agent"])
        team = Swarm(
            participants=agent_list, termination_condition=termination_condition
        )

    elif args.groupchat_type == "graph":
        # setup a graph-based group chat with the primary agent as the initial agent
        graph_builder = DiGraphBuilder()
        for agent_info in agents.values():
            graph_builder.add_node(agent_info["agent"])
        for agent_name, agent_info in agents.items():
            if agent_name == primary_user:
                continue

            def make_trigger_condition(agent_name: str):
                def trigger_condition(message: BaseChatMessage) -> bool:
                    pattern = re.compile(
                        rf"Hello @{re.escape(agent_name)}($|[^a-zA-Z0-9_-])", re.IGNORECASE
                    )
                    return bool(pattern.search(message.to_model_text()))

                return trigger_condition

            graph_builder.add_edge(
                source=agents[primary_user]["agent"],
                target=agent_info["agent"],
                condition=make_trigger_condition(agent_name),
            )
            graph_builder.add_edge(
                source=agent_info["agent"],
                target=agents[primary_user]["agent"],
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
            source=agents[primary_user]["agent"],
            target=terminal_agent,
            condition=final_trigger_condition,
        )
        graph_builder.set_entry_point(agents[primary_user]["agent"])
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

    await Console(team.run_stream(task=question))

    # Print token usage summary
    print("\nToken usage summary:")
    print(f"Prompt tokens: {llm_usage.prompt_tokens}")
    print(f"Completion tokens: {llm_usage.completion_tokens}")
    print(f"Total tokens: {llm_usage.tokens}")
    print(f"LLM calls logged to: {args.log_llm_calls}")


if __name__ == "__main__":
    asyncio.run(main())
