import chainlit as cl
from langchain_mistralai import ChatMistralAI
from chainlit.input_widget import Switch
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from agent.codegen import CodeGeneratorAgent
from agent.code_explainer import CodeExplainerAgent
from agent.rag import VectorStoreOperations, RetrievalAgent
from agent.github_agent import GitHubAgent


# Initialize VectorStoreOperations once and build retriever
op = VectorStoreOperations(user_id="user")
docs = op.load_code_and_readme_files()
if docs:
    op.add_documents(docs)
retriever = op.vector_store.as_retriever(search_kwargs={"k": 1})
retrieval_agent = RetrievalAgent(retriever)

# Instantiate agents with injected retrieval_agent
code_generator_agent = CodeGeneratorAgent(retrieval_agent=retrieval_agent)
code_explainer_agent = CodeExplainerAgent(retrieval_agent=retrieval_agent)
github_agent = GitHubAgent()

# code_generator = RunnableLambda(code_generator_agent)
code_generator = RunnableLambda(code_generator_agent)
code_generator.name = "code_generator"
code_explainer = RunnableLambda(code_explainer_agent)
code_explainer.name = "code_explainer"
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)

def extract_agent_from_text(text: str) -> str:
    agents = ["code_explainer", "code_generator", "github_agent"]
    for line in reversed(text.strip().splitlines()):
        line_clean = line.strip().lower()
        for agent in agents:
            if agent in line_clean:
                return agent
    raise ValueError("No agent decision found in supervisor output.")

class AgentState(TypedDict, total=False):
    input: str  # raw user input
    output: str
    supervisor_decision: str 

async def code_generator_node(state: AgentState) -> AgentState:
    query = state["input"]
    msg = cl.Message(content="🛠️ Generating code...")
    await msg.send()
    
    context_docs = []
    if code_generator_agent.retrieval_agent:
        context_docs = code_generator_agent.retrieval_agent.retrieve(query)
    
    full_content = ""
    
    async for chunk in code_generator_agent.agenerate_code_stream(query, context_docs):
        if chunk["type"] == "content":
            full_content += chunk["data"]
            msg.content = f"🛠️ Generating code...\n\n```python\n{full_content}\n```"
            await msg.update()
        elif chunk["type"] == "tool_output":
            full_content += chunk["data"]
            msg.content = f"🛠️ Generating code...\n\n```python\n{full_content}\n```"
            await msg.update()
        elif chunk["type"] == "error":
            msg.content = f"❌ Error: {chunk['data']}"
            await msg.update()
            return {
                **state,
                "output": chunk["data"],
            }
    
    msg.content = f"```python\n{full_content}\n```"
    await msg.update()
    
    return {
        **state,
        "output": full_content,
    }

async def code_explainer_node(state: AgentState) -> AgentState:
    query = state["input"]
    msg = cl.Message(content="🧠 Explaining code...")
    await msg.send()

    input_data = {"query": query}
    result = await code_explainer.ainvoke(input_data)

    print(f"Code explainer result: {result}")
    msg.content = result['messages'][0]['content']
    await msg.update()
    return {
        **state,
        "output": result['messages'][0]['content'],
    }

async def github_agent_node(state: AgentState) -> AgentState:
    query = state["input"]
    msg = cl.Message(content="💡 Handling GitHub request...")
    await msg.send()

    # Get all messages from the current chat session
    chat_history = cl.user_session.get("messages", [])

    # Create a list of all messages including the current one
    messages = [
        cl.Message(content=m["content"], author=m.get("author", "User"))
        for m in chat_history
    ]
    messages.append(cl.Message(content=query))

    result = await github_agent.run(messages)

    print(f"GitHub agent result: {result}")
    msg.content = result
    await msg.update()

    # Update chat history
    chat_history.append({"content": query, "author": "User"})
    chat_history.append({"content": result, "author": "Assistant"})
    cl.user_session.set("messages", chat_history)

    return {
        **state,
        "output": result,
    }

async def supervisor_node(state: AgentState) -> AgentState:
    query = state["input"]
    messages = [
        HumanMessage(content=(
            f"You are a supervisor managing three agents:\n"
            f"- code_generator: for generating code.\n"
            f"- code_explainer: for explaining code.\n"
            f"- github_agent: for handling GitHub-related tasks and issues.\n"
            f"Decide which agent should handle the user query below. "
            f"Provide a brief reasoning and then the final decision with the agent name.\n"
            f"User query: {query}\n"
            f"Output with the agent name with `code_generator`, `code_explainer`, or `github_agent` ONLY."
        ))
    ]
    msg = cl.Message(content="🧠 **Supervisor Thinking...**\n")
    await msg.send()
    decision_content = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            decision_content += chunk.content
            # Update the message with the streamed content
            msg.content = f"🧠 **Supervisor Thought Process:**\n{decision_content}"
            await msg.update()
    
    try:
        print(f"Supervisor decision: {decision_content}")
        next_agent = extract_agent_from_text(decision_content)
    except ValueError as e:
        print(f"Error parsing agent from supervisor: {e}")
        next_agent = "code_generator"  # fallback

    await cl.Message(content=f"🤖 **Supervisor routed to:** `{next_agent}`").send()
    return {**state, "supervisor_decision": next_agent}

def route_supervisor(state: AgentState):
    print(f"Routing decision: {state['supervisor_decision']}")
    return state["supervisor_decision"]

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node, async_fn=True)
builder.add_node("code_generator", code_generator_node, async_fn=True)
builder.add_node("code_explainer", code_explainer_node, async_fn=True)
builder.add_node("github_agent", github_agent_node, async_fn=True)
builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"code_generator": "code_generator", "code_explainer": "code_explainer", "github_agent": "github_agent"},
)
builder.add_edge("code_generator", END)
builder.add_edge("code_explainer", END)
builder.add_edge("github_agent", END)

graph = builder.compile()

@cl.on_chat_start
async def startup():
    await cl.ChatSettings(
        [
            Switch(
                id="use_rag",
                label="Use RAG (Codebase Context)",
                initial=True,
            )
        ]
    ).send()

    # Initialize chat history
    cl.user_session.set("messages", [])

    await cl.Message(content="Codebase indexed and ready! Ask me anything.").send()

@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("use_rag", settings["use_rag"])

@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    # Update chat history with user message
    chat_history = cl.user_session.get("messages", [])
    chat_history.append({"content": query, "author": "User"})
    cl.user_session.set("messages", chat_history)

    initial_state: AgentState = {
        "input": query,
        "output": "",
    }

    try:
        final_state = await graph.ainvoke(initial_state)
        # The agent nodes will update the chat history with their responses
        await cl.Message(content=f"🧠 Final Answer:\n{final_state['output']}").send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()
        raise