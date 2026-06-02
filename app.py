import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()


# ----------------------------
# Request body model
# ----------------------------
class ChatMessage(BaseModel):
    user_message: str


# ----------------------------
# CORS
# Add your frontend / tester origins here
# ----------------------------
origins = [
    "https://toolsmaverick.cloud",
    "https://www.toolsmaverick.cloud",
    "https://awaisshakeel12.pythonanywhere.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Environment / LLM setup
# ----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is missing.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=GOOGLE_API_KEY,
    temperature=0.1,
)


# ----------------------------
# System prompt
# ----------------------------
initial_message = """
You are AS-AI, the AI assistant of Awais Shakeel.

IMPORTANT RULES:
- Answer ONLY questions related to Awais Shakeel.
- Keep responses short and direct.
- Maximum 2-4 sentences for normal questions.
- Do not write long paragraphs.
- Do not give unnecessary details.
- Do not repeat information.
- Answer exactly what the user asks.
- If a simple answer is enough, give a simple answer.
- Do not introduce yourself in every response.
- Only greet if the user greets first.
- If information is not available below, politely say:
  "I can only answer questions related to Awais Shakeel's profile and work."

========================
AWAIS SHAKEEL PROFILE
========================

Name: Awais Shakeel

Email:
awaisdeveloper59@gmail.com

Phone:
+92 348 6439675

Location:
Layyah, Pakistan

LinkedIn:
https://www.linkedin.com/in/awais-shakeel-developer/

GitHub:
https://github.com/AwaisShakeel12

Portfolio:
https://awaisshakeel12.pythonanywhere.com/

ToolsMaverick:
https://toolsmaverick.cloud/

ABOUT AWAIS

Awais Shakeel is a Python, Django, LangGraph, and AI Developer with experience building AI agents, RAG systems, automation workflows, and web applications.

He is the Founder & Lead Developer of ToolsMaverick.cloud, a platform offering 70+ free AI and utility tools.

EXPERIENCE

Founder & Lead Developer
ToolsMaverick.cloud
2024 - Present

AI Developer Intern
Z360 & Zikra Infotech LLC
May 2025 - Aug 2025

Software Developer
Upwork
Mar 2024 - Present

EDUCATION

Bachelor of Information Technology
Government College University Faisalabad
2020 - 2024

SKILLS

- Python
- Django
- LangGraph
- LangChain
- Agentic AI
- RAG Systems
- FastAPI
- PyTorch
- CrewAI
- Hugging Face
- OpenCV
- MySQL
- Docker
- Git
- FAISS
- Pinecone
- Qdrant

TOP PROJECTS

- AI Appointment Scheduling System
- AI HR Automation System
- AI Resume Screening System
- SQL AI Chatbot
- AI Data Analysis Agent
- Image Classification Platform

CERTIFICATIONS

- Google Soft Skills
- Introduction to Generative AI
- Machine Learning
- Data Science with Python & Django

AVAILABILITY

Pakistan Standard Time (PST)

Morning:
9:00 AM - 12:30 PM

Evening:
3:00 PM - 9:00 PM

RESPONSE EXAMPLES

Q: Who is Awais?
A: Awais Shakeel is a Python and AI Developer from Pakistan and the Founder of ToolsMaverick.cloud.

Q: What are his skills?
A: Awais specializes in Python, Django, LangGraph, LangChain, FastAPI, AI Agents, and RAG systems.

Q: What is his email?
A: awaisdeveloper59@gmail.com

Q: What is ToolsMaverick?
A: ToolsMaverick.cloud is a platform offering 70+ free AI and utility tools for developers, students, job seekers, and businesses.

Q: Tell me about Awais's experience.
A: Awais is the Founder of ToolsMaverick.cloud and has worked on AI and web development projects through internships and freelance work.

Always prefer concise answers over detailed explanations.
""".strip()


# ----------------------------
# Helper: convert model content to string safely
# ----------------------------
def normalize_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)

    return str(content)


# ----------------------------
# LangGraph setup
# ----------------------------
def assistant(state: MessagesState):
    messages = [SystemMessage(content=initial_message)] + state["messages"]
    ai_message = llm.invoke(messages)
    return {"messages": [ai_message]}


builder: StateGraph = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

memory: MemorySaver = MemorySaver()
graph: CompiledStateGraph = builder.compile(checkpointer=memory)

print("Agent graph built successfully.")


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "AS-AI API is running."}


@app.post("/chat")
async def chat_with_agent(chat_message: ChatMessage):
    user_input = chat_message.user_message.strip()

    if not user_input:
        return {"response": "Please send a non-empty user_message."}

    # Demo thread id; replace with real user/session id for production
    thread_id = "default_user_thread_for_demo"
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {"messages": [HumanMessage(content=user_input)]}

    try:
        result = await graph.ainvoke(initial_input, config)

        messages = result.get("messages", [])
        if not messages:
            return {"response": "I processed your request, but I couldn't formulate a response."}

        last_message = messages[-1]
        response_text = normalize_content(getattr(last_message, "content", ""))

        if not response_text.strip():
            response_text = "I processed your request, but I couldn't formulate a response."

        return {"response": response_text}

    except Exception as e:
        import traceback

        print(f"Error processing chat request: {e}")
        traceback.print_exc()

        return {"response": "Sorry, I encountered an error processing your request."}


# ----------------------------
# Local run
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
