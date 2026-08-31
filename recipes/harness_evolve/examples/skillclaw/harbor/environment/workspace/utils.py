"""
Tool definitions for meeting negotiation agent evaluation (OpenAI function calling format).

Runner loads this module to get TOOLS and passes them to the LLM API.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Send an HTTP request to a URL. Use this to call mock service API endpoints (Gmail at http://localhost:9100, Calendar at http://localhost:9101).",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "description": "HTTP method. Use POST for API calls.",
                        "default": "POST",
                    },
                    "url": {
                        "type": "string",
                        "description": "Full URL (e.g. http://localhost:9100/gmail/messages).",
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for the request. Omit for GET or empty body.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Use for saving results (e.g. results.md).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path (e.g. /tmp_workspace/results/results.md).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


def get_tools():
    """Return the tool list (for runner that does import and calls get_tools())."""
    return list(TOOLS)
