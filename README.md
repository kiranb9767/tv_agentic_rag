# AI-Powered User Manual Assistant with Agentic RAG Workflow

An advanced **Agentic RAG (Retrieval-Augmented Generation)** application built using **LangChain, LangGraph, Pinecone, Groq, MongoDB, Tavily Search, and WebSockets**.

This project acts as an intelligent **Philips TV User Manual Assistant** capable of:

- Understanding user questions
- Retrieving relevant manual content using vector search
- Searching the web dynamically when required
- Streaming AI responses in real time
- Maintaining conversation history
- Using intelligent AI workflow routing with LangGraph

-------------------------------

# Demo Features

✅ Agentic AI Workflow  
✅ Hybrid RAG + Web Search  
✅ Real-Time Token Streaming  
✅ Pinecone Vector Search  
✅ LangGraph State Machine  
✅ Context Compression Retriever  
✅ WebSocket Communication  
✅ MongoDB Chat History  
✅ Guardrails & Fallback Handling

---------------------

# Tech Stack

## Backend
- Node.js
- Express.js
- LangChain
- LangGraph
- Groq LLM
- Pinecone
- MongoDB
- Tavily Search API
- WebSockets

## Frontend
- HTML
- CSS
- Vanilla JavaScript

--------------------------

# AI Workflow Architecture

text
User Question
      ↓
Input Guard
      ↓
Planner Agent
 ┌───────────────┐
 │               │
RAG           WEB SEARCH
 │               │
 └──── Merge ────┘
        ↓
Answer Generator
        ↓
Guard Validation
        ↓
Real-Time Streaming Response

----------------------------------------

# Key AI Concepts Implemented

- Agentic AI
- Retrieval-Augmented Generation (RAG)
- Hybrid Retrieval
- Semantic Search
- Context Compression
- Streaming LLM Responses
- Multi-Step AI Workflow
- Vector Database Retrieval
- Real-Time AI Communication
- State-Based AI Routing

----------------------------------------

# Project Structure

text
tv_agentic_rag/
│
├── backend/
│   ├── server.js
│   ├── package.json
│   ├── .env
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│
├── document/
│   └── UserManual/
│       ├── html/
│       └── EN.xml
│
└── README.md

------------------------------------

# Environment Variables

Create a `.env` file inside the backend folder.

```env
OPENAI_API_KEY=your_openai_api_key

GROQ_API_KEY=your_groq_api_key

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index

TAVILY_API_KEY=your_tavily_api_key

MONGODB_URI=your_mongodb_connection
MONGODB_DB_NAME=your_database_name
```

------------------------------------------------

# Installation

## Clone Repository

```bash
git clone https://github.com/kiranb9767/tv_agentic_rag.git

cd tv_agentic_rag
```

---

## Install Dependencies

```bash
npm install
```

-------------------------------------------------

# Run Project

## Start Server

```bash
node server.js
```

Server runs at:

http://localhost:3000

-----------------------------------------------------------

# Upload User Manual Documents

Inside backend code:

```js
await uploadDocuments();
```

Uncomment this once to upload documents into Pinecone.

After successful upload, comment it again.

----------------------------------------------------------

# How It Works

## 1. Document Processing

- Reads HTML manual files
- Cleans HTML content
- Splits into chunks
- Creates embeddings
- Uploads vectors into Pinecone

------------------------------------------------

## 2. Planner Agent

The planner intelligently decides:

- `RAG_ONLY`
- `WEB_ONLY`
- `RAG_AND_WEB`

based on user intent.

------------------------------------------------

## 3. Retrieval Layer

Uses:

- Pinecone Vector Database
- MMR Retrieval
- Context Compression Retriever

to fetch highly relevant context.

----------------------------------------------------

## 4. Web Search Layer

Uses Tavily Search API for:

- latest/public information
- internet-based answers
- dynamic external knowledge

----------------------------------------------

## 5. Streaming Layer

AI responses are streamed token-by-token using WebSockets for real-time UX.

Example:

```json
{
  "type": "stream",
  "token": "Hello"
}
```

----------------------------------------------------------------------

# Frontend Features

- Modern Dark UI
- Streaming Chat Interface
- Responsive Layout
- Custom Scrollbars
- Remote Focus Friendly UI
- AI/User Chat Bubbles

------------------------------------------------------------

# Future Improvements

- Voice Assistant Integration
- Multi-language Support
- Redis Caching
- Authentication System
- Docker Deployment

----------------------------------------------------

# Why LangGraph?

LangGraph is used to create a state-driven AI workflow system where the application dynamically routes between:

- RAG Retrieval
- Web Search
- Merge Logic
- Guardrails
- Fallback Handling

This makes the application more reliable and scalable compared to a basic chatbot.

------------------------------------------------------

# APIs & Services Used

- OpenAI Embeddings API
- Groq LLM API
- Pinecone Vector Database
- MongoDB Atlas
- Tavily Search API

--------------------------------------------------------------

# Example Use Cases

- Smart TV Assistant
- Customer Support AI
- Product Documentation Assistant
- Enterprise Knowledge Base
- Technical Support Chatbot

---------------------------------------------------------------

# Author

Kiran Burle

GitHub:
https://github.com/kiranb9767

---------------------------------------------------

# GitHub Topics

agentic-ai
rag
langchain
langgraph
pinecone
groq
mongodb
websocket
semantic-search
vector-database
retrieval-augmented-generation
ai-chatbot
