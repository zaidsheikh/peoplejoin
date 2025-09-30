import json
from async_collab.orchestrator.datum import AsyncCollabDatum
from async_collab.orchestrator.prompt_builder import PromptBuilder
from async_collab.plugins.plugins.cot_plugin import SimpleReasoningPlugin
from async_collab.scenarios.all_exemplars import exemplar_by_id
from async_collab.plugins.plugin import Plugin

IS_MESSAGE_NONE_MODE: bool = False

# TODO: use litellm.utils.function_to_dict

system_plugin_tools = [
    {
        "type": "function",
        "function": {
            "name": "System__finish",
            "description": "Call this function to indicate that the current turn is complete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

enterprise_plugin_tools = [
    {
        "type": "function",
        "function": {
            "name": "Enterprise__send_message",
            "description": "Send a message to a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to send the message to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the message to send"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the message"
                    }
                },
                "required": ["user_id", "content"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Enterprise__send_session_completed",
            "description": "If the primary user indicates that they no longer need assistance, send a session completed message.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Enterprise__resolve_primary_user",
            "description": "Return the primary user details.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Enterprise__resolve_person",
            "description": "Find list of persons matching a given name and return details of the first match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person to search for"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            }
        }
    }
]

enterprise_search_plugin_tools = [
    {
        "type": "function",
        "function": {
            "name": "EnterpriseSearch__search_documents",
            "description": "Returns the list of relevant documents (including document content/records)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "EnterpriseSearch__search_relevant_people",
            "description": "Returns names of relevant person and any accompanying rationale",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant people"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]

reflection_plugin_tools = [
    {
        "type": "function",
        "function": {
            "name": "Reflection__thought",
            "description": "Assess the current state of the conversation and decide on the next action",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your assessment of the current state and planned next action"
                    }
                },
                "required": ["thought"],
                "additionalProperties": False
            }
        }
    }
]


class ChatCompletionPromptBuilder(PromptBuilder):
    prompt: list[dict]
    tools: list[dict]

    def __init__(self, plugins: list[Plugin], exemplar_ids: list[str]) -> None:
        super().__init__(plugins, exemplar_ids)
        self.tools = self.get_tools()
        self.reset()

    def get_tools(self) -> list[dict]:
        """Get all tools based on enabled plugins"""
        all_tools = system_plugin_tools + enterprise_plugin_tools + enterprise_search_plugin_tools
        
        # Check if reflection plugin is enabled
        plugin_ids = {plugin.plugin_id for plugin in self.plugins}
        is_reflection_plugin_on = SimpleReasoningPlugin.plugin_id in plugin_ids
        if is_reflection_plugin_on:
            all_tools += reflection_plugin_tools

        return all_tools

    def get_instruction_prompt(self) -> str:
        """
        Get the prompt for the instruction
        """
        prompt = "You are a clever and helpful assistant helping a user. To accomplish the user request, you must use the specified Python functions. Each function call should be within a single line. Only respond with tool calls. If no tool call is needed, return the tool call System__finish()\n"
        if IS_MESSAGE_NONE_MODE:
            prompt += "Do not send any messages to any user other than the primary user. If the primary insists to reach out to other users, tell the primary user that you are not allowed to do so.\n"
        return prompt

    def update_prompt(self, **kwargs):
        content = ""
        if "event" in kwargs:
            content += f"Event: {kwargs['event']}\n"
        if "action" in kwargs:
            content += f"Action: {kwargs['action']}\n"
        if "result" in kwargs:
            content += f"Result: {kwargs['result']}\n"
        for key, value in kwargs.items():
            if key not in ["event", "action", "result", "role", "content"]:
                self.prompt += [{"role": "user", "content": value}]
        if content:
            self.prompt += [{"role": "user", "content": content}]
        if "role" in kwargs and "content" in kwargs:
            self.prompt += [{"role": kwargs["role"], "content": kwargs["content"]}]


    def reset(self):
        self.prompt = [{"role": "system", "content": self.get_instruction_prompt()}] + self.get_exemplars_messages() # type: ignore

    def get_cur_event_repl(self):
        self.cur_event_repl = ""
        for message in self.prompt:
            if message["role"] != "system":
                self.cur_event_repl += message["content"] + "\n"
        return self.cur_event_repl

    def reset_cur_event_repl(self):
        self.cur_event_repl = ""

    def get_exemplars_messages(self) -> list[dict]:
        """
        Get the prompt for the examples
        """
        return []

    def get_exemplars_prompt(self) -> str:
        """
        Not implemented
        """
        raise NotImplementedError

    def get_plugin_prompts(self) -> str:
        """
        Not implemented
        """
        raise NotImplementedError

    def init_test_exemplar(self):
        """
        Not implemented
        """
        raise NotImplementedError