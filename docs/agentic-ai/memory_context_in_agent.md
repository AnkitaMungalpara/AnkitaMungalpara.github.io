---
title: 'Managing Memory, Context, and State in an AI Agent'
date: 2025-04-05

parent: Agentic AI

nav_order: 2

tags:
  - Agentic AI
  - Memory
  - State
  - Conetxt
---

# Inside the Mind of an AI Agent: Memory, State, and Context Explained
{: .no_toc }

![1](../../../assets/images/ai_agent.webp)

This blog post offers a detailed description of an agentic system architecture that integrates Large Language Models (LLMs) with tool-based reasoning to address complex, multi-step problems. We will implement an agent that demonstrates autonomous decision-making, efficient memory management, and adaptable tool orchestration. We will also explore the interaction between perception, planning, and action modules to demonstrate their collaboration in producing intelligent, reasoning-driven behavior.

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>



## Introduction to Agentic Systems

Agentic systems represent a paradigm shift in artificial intelligence, moving beyond simple input-output transformations to create autonomous entities capable of reasoning, planning, and executing complex tasks. Unlike traditional AI systems that process queries in isolation, agentic systems maintain context, learn from interactions, and dynamically adapt their behavior based on available tools and past experiences.


The architecture presented here implements a [Model Control Protocol (MCP)](2025-04-05-mcp.html)-based system that can connect to multiple tools, analyze user requests through perception and decision, and execute a multi-step reasoning chain to achieve the desired goal.

## Core Components

This agentic flow consists of several interconnected modules, each responsible for specific aspects of the reasoning process:

- `Agent Context`: Maintains session state, memory, and execution history
- `Multi-MCP Dispatcher`: Manages connections to multiple tool servers
- `Perception`: Analyzes and structures user input
- `Decision`: Generates action plans using LLM reasoning
- `Memory`: Stores and retrieves relevant execution history
- `Tool Orchestration`: Manages tool discovery, filtering, and execution

## Model Control Protocol (MCP)

The MCP (Model Control Protocol) framework enables integration with multiple tool servers. Each server exposes specific capabilities through a standardized interface:

### Architecture

This architecture supports dynamic tool discovery, where the agent automatically identifies available tools across multiple MCP servers.

```python
class MultiMCP:
    def __init__(self, server_configs: List[dict]):
        self.server_configs = server_configs
        self.tool_map: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        for config in self.server_configs:
            params = StdioServerParameters(
                command=sys.executable,
                args=[config["script"]],
                cwd=config.get("cwd", os.getcwd())
            )
            
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    
                    for tool in tools.tools:
                        self.tool_map[tool.name] = {
                            "config": config,
                            "tool": tool
                        }
```

### Tool Filtering

The system implements tool management capability that enable tool selection based on contextual hints:

```python
def filter_tools_by_hint(tools: List[Any], hint: Optional[str] = None) -> List[Any]:
    """
    Filters tools based on contextual hints from perception analysis.
    Implements fuzzy matching to handle variations in tool naming.
    """
    if not hint:
        return tools
    
    hint_lower = hint.lower()
    filtered = [tool for tool in tools if hint_lower in tool.name.lower()]
    return filtered if filtered else tools 

def summarize_tools(tools: List[Any]) -> str:
    """
    Generates human-readable tool descriptions for LLM consumption.
    """
    return "\n".join(
        f"- {tool.name}: {getattr(tool, 'description', 'No description provided.')}"
        for tool in tools
    )
```

This filtering mechanism reduces the cognitive load on the LLM by presenting only relevant tools, improving both performance and accuracy in tool selection.

## Perception and Intent Analysis

### Structured Input Processing

The perception module represents a critical component that transforms natural language queries into structured data suitable for computational processing:

```python
class PerceptionResult(BaseModel):
    user_input: str
    intent: Optional[str]
    entities: List[str] = []
    tool_hint: Optional[str] = None

async def extract_perception(user_input: str) -> PerceptionResult:
    prompt = f"""
    You are an AI that extracts structured facts from user input.
    Available tools: {tool_context}
    Input: "{user_input}"
    Return the response as a Python dictionary with keys:
    - intent: (brief phrase about what the user wants)
    - entities: a list of strings representing keywords or values
    - tool_hint: (name of the MCP tool that might be useful, if any)
    - user_input: same as above
    """
    response = await model.generate_text(prompt)
    clean = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE)
    parsed = json.loads(clean)
    return PerceptionResult(**parsed)
```

This approach implements semantic parsing that goes beyond simple keyword extraction to understand user intent and identify relevant computational entities.

### Recovery and Validation

The perception also includes error-handling mechanism:

```python
# Entity validation and correction
if isinstance(parsed.get("entities"), dict):
    parsed["entities"] = list(parsed["entities"].values())

# Safe fallback
except Exception as e:
    print(f"[perception] LLM perception failed: {e}")
    return PerceptionResult(user_input=user_input)
```


## Memory and Context Management

### Session Context

The system maintains a context across execution steps through the `AgentContext` class:

```python
class AgentContext:
    def __init__(self, user_input: str, profile: Optional[AgentProfile] = None):
        self.user_input = user_input
        self.session_id = f"session-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.step = 0
        self.memory = MemoryManager(
            embedding_model_url=self.agent_profile.memory_config["embedding_url"],
            model_name=self.agent_profile.memory_config["embedding_model"]
        )
        self.memory_trace: List[MemoryItem] = []
        self.final_answer: Optional[str] = None
```

This context management facilitates **stateful reasoning**, allowing each step to build upon previous steps while maintaining awareness of the end goal.

### Memory Retrieval

The memory implements retrieval mechanism that support contextual recall:

```python
retrieved = self.context.memory.retrieve(
    query=query,
    top_k=self.context.agent_profile.memory_config["top_k"],
    type_filter=self.context.agent_profile.memory_config.get("type_filter", None),
    session_filter=self.context.session_id
)
```

This implementation accommodates semantic memory retrieval, allowing for the exploration and incorporation of relevant past steps into current reasoning processes.

## Decision Making and Planning

### Strategic Planning Framework

The decision module implements a planning module that combines LLM reasoning with strategic constraints:

```python
async def decide_next_action(
    context: AgentContext,
    perception: PerceptionResult,
    memory_items: list[MemoryItem],
    all_tools: list[Any],
    last_result: str = "",
) -> str:
    strategy = context.agent_profile.strategy
    step = context.step + 1
    max_steps = context.agent_profile.max_steps
    
    # Implement hint-based tool filtering
    filtered_tools = filter_tools_by_hint(all_tools, hint=perception.tool_hint)
    filtered_summary = summarize_tools(filtered_tools)
    
    plan = await generate_plan(
        perception=perception,
        memory_items=memory_items,
        tool_descriptions=filtered_summary,
        step_num=step,
        max_steps=max_steps,
    )
    return plan
```

This planning approach employs hierarchical reasoning, wherein high-level strategies direct the selection and execution of tools.

### Plan Generation and Validation

The core planning logic employs prompt engineering to ensure reliable plan generation:

```python
async def generate_plan(
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str] = None,
    step_num: int = 1,
    max_steps: int = 3
) -> str:
    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"
    
    prompt = f"""
    You are a reasoning-driven AI agent with access to tools and memory.

    Context:
    - Step: {step_num} of {max_steps}
    - Memory: {memory_texts}
    - Available tools: {tool_descriptions}
    
    Input Summary:
    - User input: "{perception.user_input}"
    - Intent: {perception.intent}
    - Entities: {', '.join(perception.entities)}
    - Tool hint: {perception.tool_hint or 'None'}
    
    Respond with either:
    - FUNCTION_CALL: tool_name|param1=value1|param2=value2
    - FINAL_ANSWER: [your final result]
    """
    response = await model.generate_text(prompt)
    return response.strip()
```

This structured prompting approach ensures consistent output formatting and reliable plan parsing.

## Tool Execution and Error Handling

### Function Call Parsing

```python
def parse_function_call(response: str) -> tuple[str, Dict[str, Any]]:
    """
    Parses FUNCTION_CALL strings into executable parameters.
    Supports nested parameter structures and type inference.
    """
    if not response.startswith("FUNCTION_CALL:"):
        raise ValueError("Invalid function call format.")
    
    _, raw = response.split(":", 1)
    parts = [p.strip() for p in raw.split("|")]
    tool_name, param_parts = parts[0], parts[1:]
    
    args = {}
    for part in param_parts:
        key, val = part.split("=", 1)
        
        # Implement type inference
        try:
            parsed_val = ast.literal_eval(val)
        except Exception:
            parsed_val = val.strip()
        
        # Support nested parameter structures
        keys = key.split(".")
        current = args
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = parsed_val
    
    return tool_name, args
```

### Tool Execution Pipeline

The execution pipeline manages tool invocation with comprehensive error handling:

```python
tool_name, arguments = parse_function_call(plan)

# Handle input wrapping for specific tools
if self.tool_expects_input(tool_name):
    tool_input = {'input': arguments} if not (isinstance(arguments, dict) and 'input' in arguments) else arguments
else:
    tool_input = arguments

response = await self.mcp.call_tool(tool_name, tool_input)

# Safe content extraction
raw = getattr(response.content, 'text', str(response.content))
result_obj = json.loads(raw) if raw.strip().startswith("{") else raw
result_str = result_obj.get("markdown") if isinstance(result_obj, dict) else str(result_obj)
```

## Real-world Example: Multi-Step Mathematical Reasoning

To illustrate the agentic workflow in managing complex multi-step computations, we will now examine the execution of a mathematical query. This example will showcase the agent's capability to comprehend the problem, perform sequential operations, and maintain contextual awareness throughout multiple reasoning steps.

```
Query: "Find the ASCII values of characters in AGENTIC AI and then return sum of exponentials of those values."
```

This query presents a multi-step computational challenge requiring:

- **String Processing**: Converting text characters to ASCII numerical values
- **Mathematical Computation**: Calculating exponentials of large integers
- **Aggregation**: Summing exponential results
- **Sequential Reasoning**: Maintaining intermediate results across steps

### Perception 

```
PerceptionResult(
    user_input="Find the ASCII values of characters in AGENTIC AI and then return sum of exponentials of those values",
    intent="Calculate sum of exponentials of ASCII values",
    entities=["AGENTIC AI", "ASCII", "exponentials", "sum"],
    tool_hint="Calculator"
)
```

The perception module effectively recognizes that the task has two components: string processing and mathematical computation. While the tool hint `Calculator` indicates that mathematical operations are needed, the system must first address the string component.

### Strategic Planning

```python
# Decision module generates first action plan
plan = "FUNCTION_CALL: strings_to_chars_to_int|input={'string': 'AGENTIC AI'}"
```

The planning module demonstrates tool selection by recognizing that string-to-ASCII conversion must precede mathematical operations. It implies the agent's ability to break down complex tasks into logical sequences.

### Function Call Parsing

The parsing handles nested parameter structures correctly, maintaining the required input wrapping for the target tool.

```python
# Parser extracts tool name and parameters
tool_name, arguments = parse_function_call(plan)
# Result: ("strings_to_chars_to_int", {'input': {'string': 'AGENTIC AI'}})
```

### ASCII Conversion Execution

```python
# Tool execution through MCP framework
response = await mcp.call_tool("strings_to_chars_to_int", {
    "input": {"string": "AGENTIC AI"}
})

# Tool returns ASCII values for each character
result = {"ascii_values": [65, 71, 69, 78, 84, 73, 67, 32, 65, 73]}
```

### Memory

```python
# storing intermediate result in memory
memory_item = MemoryItem(
    text="strings_to_chars_to_int({'string': 'AGENTIC AI'}) → [65, 71, 69, 78, 84, 73, 67, 32, 65, 73]",
    type="tool_output",
    tool_name="strings_to_chars_to_int",
    user_query="Find the ASCII values of characters in AGENTIC AI and then return sum of exponentials of those values",
    tags=["strings_to_chars_to_int", "ascii_conversion"],
    session_id="session-1751741661-fcb7ee"
)
```

### Second Tool Execution - Mathematical Processing

```python
# System generates enhanced query incorporating previous result
revised_query = """Original user task: Find the ASCII values of characters in AGENTIC AI and then return sum of exponentials of those values

Your last tool produced this result:
{"ascii_values": [65, 71, 69, 78, 84, 73, 67, 32, 65, 73]}

If this fully answers the task, return:
FINAL_ANSWER: your answer

Otherwise, return the next FUNCTION_CALL."""
```

### Mathematical Planning

```python
# Perception analysis for second step
# Intent: "Calculate the sum of exponentials of ASCII values"
# Memory: Retrieved 3 memories (including previous ASCII conversion)

# Generated plan
plan = "FUNCTION_CALL: int_list_to_exponential_sum|input={'numbers': [65, 71, 69, 78, 84, 73, 67, 32, 65, 73]}"
```

The system demonstrates reasoning by understanding that the ASCII values from the previous step must be processed through exponential summation.

### Exponential Computation

```python
# Execute exponential summation tool
response = await mcp.call_tool("int_list_to_exponential_sum", {
    "input": {"numbers": [65, 71, 69, 78, 84, 73, 67, 32, 65, 73]}
})

# Mathematical computation: e^65 + e^71 + e^69 + e^78 + e^84 + e^73 + e^67 + e^32 + e^65 + e^73
result = {"result": 3.032684709254243e+36}
```

**Result**

```python
"FINAL_ANSWER: [3.032684709254243e+36]"
```

<!-- ## Performance Characteristics

The execution demonstrates several performance advantages:

- Efficient Tool Selection: Hint-based filtering reduces search space
- Minimal Memory Overhead: Selective storage of relevant intermediate results
- Optimized Communication: Structured parameter passing minimizes protocol overhead
- Scalable Architecture: Modular design supports concurrent tool execution

## Conclusion and Future Directions -->

The presented agentic flow illustrates multi-step reasoning capabilities through the integration of perception, planning, memory, and tool execution.

The modular design of the architecture allows for extensibility and customization while also ensuring effective error handling and strong performance characteristics.

