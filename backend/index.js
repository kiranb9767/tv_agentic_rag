import dotenv from "dotenv";
import fs from "fs";
import readline from "readline";

import rateLimit from "express-rate-limit";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { Document } from "@langchain/core/documents";

import { ChatGroq } from "@langchain/groq";
import { OpenAIEmbeddings } from "@langchain/openai";

import { WebSocketServer } from "ws";
import { TavilySearch } from "@langchain/tavily";
import { StateGraph, Annotation } from "@langchain/langgraph";

import { MongoClient } from "mongodb";
import express from "express";
import http from "http";

import path from "path";
import { fileURLToPath } from "url";

import { ContextualCompressionRetriever } from "@langchain/classic/retrievers/contextual_compression";

import { LLMChainExtractor } from "@langchain/classic/retrievers/document_compressors/chain_extract";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const exapp = express();

exapp.use(express.static(path.join(__dirname, "../frontend")));

const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  message: "Too many requests",
});

exapp.use(limiter);

const server = http.createServer(exapp);

const wss = new WebSocketServer({
  noServer: true,
});

server.on("upgrade", (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit("connection", ws, request);
  });
});

server.listen(3000, "0.0.0.0", () => {
  console.log("Server running on network");
});

let current_lang_iso = "en";

const mongoClient = new MongoClient(process.env.MONGODB_URI);

await mongoClient.connect();

console.log("Connected to MongoDB");

const db = mongoClient.db(process.env.MONGODB_DB_NAME);

const conversationHistory = db.collection("conversation_history");

const GraphState = Annotation.Root({
  Question: Annotation,
  Answer: Annotation,
  Confidence: Annotation,
  error: Annotation,
  strategy: Annotation,
  ragQuery: Annotation,
  webQuery: Annotation,
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex: index,
});

const groqLlm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama-3.1-8b-instant",
  temperature: 0,
  streaming: true,
  maxTokens: 500,
}).withRetry({
    stopAfterAttempt: 3,
  }).withConfig({
    timeout: 8000,
  });

const webSearchTool = new TavilySearch({
  apiKey: process.env.TAVILY_API_KEY,
  maxResults: 3,
});

function xmlParseToString() {
  const xmlData = fs.readFileSync(
    `../document/UserManual/${current_lang_iso.toUpperCase()}.xml`,
    "utf8",
  );

  const result = {};
  const chapters = xmlData.match(/<chapter[^>]*>[\s\S]*?<\/chapter>/g) || [];
  chapters.forEach((chapter) => {
    const title = chapter.match(/title="([^"]+)"/)?.[1]?.trim() || "";
    const models = [...chapter.matchAll(/<diversity>(.*?)<\/diversity>/g)].map(
      (m) => m[1],
    );

    const src = chapter.match(/src="([^"]+)"/)?.[1]?.trim() || "";
    const fileName = src.split("/").pop();

    result[fileName] = {
      title,
      models,
    };
  });

  return result;
}

function extractTextFromHtml(html) {
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<\/h[1-6]>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/li>/gi, "\n")
    .replace(/&#\d+;/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&gt;/g, ">")
    .replace(/&lt;/g, "<")
    .replace(/&amp;/g, "&")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function uploadDocuments() {
  try {
    const folderPath = `../document/UserManual/html/${current_lang_iso.toUpperCase()}`;
    const htmlFiles = fs.readdirSync(folderPath);
    const xmlData = xmlParseToString();
    const documents = [];

    for (const file of htmlFiles) {
      const filePath = `${folderPath}/${file}`;
      const htmlContent = fs.readFileSync(filePath, "utf8");
      const text = extractTextFromHtml(htmlContent);

      documents.push(
        new Document({pageContent: text
          ,
          metadata: {
            source: file,
            language: current_lang_iso.toUpperCase(),
            models: (xmlData[file]?.models || []).join(","),
            title: xmlData[file]?.title || "",
          },
        }),
      );
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 400,
      chunkOverlap: 80,
    });

    const chunks = await textSplitter.splitDocuments(documents);

    await index.deleteAll();
    await vectorStore.addDocuments(chunks);

    console.log("Upload successful!");
  } catch (e) {
    console.log("UPLOAD ERROR:", e);
  }
}

/*
await uploadDocuments();
*/

const baseRetriever = vectorStore.asRetriever({
  searchType: "mmr",
  k: 8,
});

const compressor = LLMChainExtractor.fromLLM(groqLlm);

const compressionRetriever = new ContextualCompressionRetriever({
  baseRetriever,
  baseCompressor: compressor,
});

async function retriveRelevantDocuments(query) {
  const docs = await compressionRetriever.invoke(query);

  return docs.slice(0, 5);
}

async function compressContext(question, docs) {
  const compressed = await groqLlm.invoke([
    new SystemMessage(`
      Extract ONLY useful information related to the user question.
      Remove unrelated details.
      Keep concise.`),

    new HumanMessage(`
      Question:
        ${question}
      Documents:
        ${docs.map((d) => d.pageContent).join("\n")}`),
  ]);
  return compressed.content;
}

const BAD_WORDS = ["fuck", "shit", "idiot", "bastard", "asshole"];

function inputGuardNode(state) {
  const abusive = BAD_WORDS.some((word) =>
    state.Question.toLowerCase().includes(word),
  );
  if (abusive) {
    return {
      ...state,
      error: true,
      Answer: "Please use respectful language.",
      Confidence: 1,
    };
  }
  return state;
}

async function plannerNode(state) {
  try {
    const result = await groqLlm.invoke([
      new SystemMessage(`
You are a routing planner.

Your job is to decide whether the question should use:

1. RAG_ONLY
- when internal knowledge/documents are enough

2. WEB_ONLY
- when external/public/latest internet information is required

3. RAG_AND_WEB
- when both internal docs and internet information are needed

Return ONLY valid JSON.

Format:
{
 "strategy":"RAG_ONLY|WEB_ONLY|RAG_AND_WEB",
 "ragQuery":"",
 "webQuery":""
}

Rules:
- Questions about uploaded manuals, setup, troubleshooting, settings, device features usually use RAG_ONLY
- Questions requiring latest/public/general internet information use WEB_ONLY
- If both sources are useful use RAG_AND_WEB
- Do not explain anything
- Return JSON only
`),

      new HumanMessage(`
Question:
${state.Question}
`),
    ]);

    let parsed;

    try {
      const cleaned = result.content
        .replace(/```json/g, "")
        .replace(/```/g, "")
        .trim();

      parsed = JSON.parse(cleaned);
    } catch {
      parsed = {
        strategy: "RAG_ONLY",

        ragQuery: state.Question,

        webQuery: state.Question,
      };
    }

    return {
      ...state,

      strategy: parsed.strategy || "RAG_ONLY",

      ragQuery: parsed.ragQuery || state.Question,

      webQuery: parsed.webQuery || state.Question,
    };
  } catch {
    return {
      ...state,

      strategy: "RAG_ONLY",

      ragQuery: state.Question,

      webQuery: state.Question,
    };
  }
}

async function ragNode(state) {
  if (state.strategy === "WEB_ONLY") {
    return state;
  }

  try {
    const docs = await retriveRelevantDocuments(state.ragQuery);

    return {
      ...state,
      RagContext: docs.map((d) => d.pageContent).join("\n"),
    };
  } catch {
    return {
      ...state,
      RagContext: "",
    };
  }
}

async function webNode(state, config) {
  if (state.strategy === "RAG_ONLY") {
    return state;
  }

  const ws = config?.configurable?.ws;

  let finalAnswer = "";

  try {
    const webResult = await webSearchTool.invoke({
      query: state.webQuery || state.Question,
    });

    console.log("Web Search Result:", webResult);

    let webContext = "";

    if (Array.isArray(webResult?.results)) {
      webContext = webResult.results
        .map(
          (r) => `
content: ${r.content || ""}
url: ${r.url || ""}
`,
        )
        .join("\n");
    } else {
      webContext = JSON.stringify(webResult);
    }

    const result = await groqLlm.stream([
      new SystemMessage(`
Answer the user's question using the provided web information.

Rules:
1. Give direct answer
2. Use available URLs if relevant
3. If URL available format as HTML link
4. Keep answer concise
5. Do not hallucinate

Example:
<a href="https://www.netflix.com">Watch Here</a>
`),

      new HumanMessage(`
Question:
${state.Question}

Web information:
${webContext}
`),
    ]);

    for await (const chunk of result) {
      finalAnswer += chunk.content || "";

      if (ws) {
        ws.send(
          JSON.stringify({
            type: "stream",
            token: chunk.content || "",
          }),
        );
      }
    }

    if (ws) {
      ws.send(
        JSON.stringify({
          type: "done",
        }),
      );
    }
  } catch {
    finalAnswer = "NOT_SURE";
  }

  console.log("Web Node Result:", finalAnswer);

  return {
    ...state,

    WebContext: finalAnswer,

    Answer: finalAnswer,

    Confidence: finalAnswer.includes("NOT_SURE") ? 0.3 : 0.9,
  };
}

function mergeNode(state) {
  return {
    ...state,

    Context: `
RAG:
${state.RagContext || ""}

WEB:
${state.WebContext || ""}
`,
  };
}

async function answerNode(state, config) {
  if (state.strategy === "WEB_ONLY") {
    return {
      ...state,
    };
  }

  const ws = config?.configurable?.ws;

  let finalAnswer = "";

  try {
    const result = await groqLlm.stream([
      new SystemMessage(`
You are the User Manual Assistant for Philips TV.

STRICT RULES:
1. Answer ONLY from context
2. Prefer RAG information first
3. Use WEB only if needed
4. Keep concise
5. No hallucination
`),

      new HumanMessage(`
Question:
${state.Question}

Context:
${state.Context}
`),
    ]);

    for await (const chunk of result) {
      if (chunk.content) {
        finalAnswer += chunk.content;

        if (ws) {
          ws.send(
            JSON.stringify({
              type: "stream",
              token: chunk.content,
            }),
          );
        }
      }
    }

    if (ws) {
      ws.send(
        JSON.stringify({
          type: "done",
        }),
      );
    }
  } catch {
    finalAnswer = "NOT_SURE";
  }

  return {
    ...state,

    Answer: finalAnswer,

    Confidence: finalAnswer.includes("NOT_SURE") ? 0.3 : 0.9,
  };
}

function guardNode(state) {
  if (!state.Answer) {
    return {
      ...state,
      error: true,
    };
  }

  if (state.Confidence && state.Confidence < 0.6) {
    return {
      ...state,
      error: true,
    };
  }

  return {
    ...state,
    error: false,
  };
}

function fallbackNode(state) {
  return {
    ...state,

    Answer: "Unable to answer safely.",
  };
}

const graph = new StateGraph({
  channels: GraphState.spec,
});

graph
  .addNode("INPUT_GUARD", inputGuardNode)

  .addNode("PLANNER", plannerNode)

  .addNode("RAG", ragNode)

  .addNode("WEB", webNode)

  .addNode("MERGE", mergeNode)

  .addNode("ANSWER", answerNode)

  .addNode("GUARD", guardNode)

  .addNode("FALLBACK", fallbackNode);

graph.setEntryPoint("INPUT_GUARD");

graph.addConditionalEdges(
  "INPUT_GUARD",
  (state) => (state.error ? "FALLBACK" : "PLANNER"),
  {
    FALLBACK: "FALLBACK",

    PLANNER: "PLANNER",
  },
);

graph.addConditionalEdges("PLANNER", (state) => state.strategy, {
  RAG_ONLY: "RAG",

  WEB_ONLY: "WEB",

  RAG_AND_WEB: "RAG",
});

graph.addConditionalEdges(
  "RAG",
  (state) => (state.strategy === "RAG_AND_WEB" ? "WEB" : "MERGE"),
  {
    WEB: "WEB",

    MERGE: "MERGE",
  },
);

graph.addEdge("WEB", "MERGE");

graph.addEdge("MERGE", "ANSWER");

graph.addEdge("ANSWER", "GUARD");

graph.addConditionalEdges(
  "GUARD",
  (state) => (state.error ? "FALLBACK" : "__end__"),
  {
    FALLBACK: "FALLBACK",

    __end__: "__end__",
  },
);

graph.addEdge("FALLBACK", "__end__");

const app = graph.compile();

/////////  ws server to handle msg and send rsp////////////////////
wss.on("connection", async (ws) => {
  console.log("Web soc server connected......");

  const history = await conversationHistory
    .find({})
    .sort({ timestamp: 1 })
    .toArray();

  console.log("Conversation History:", history);

  ws.send(JSON.stringify({ type: "history", message: history }));

  ws.on("message", async (message) => {
    console.log("Web soc on msg receive......");
    const data = JSON.parse(message.toString());

    if (data.type === "clear_history") {
      await conversationHistory.deleteMany({});
      console.log("Conversation history cleared.");
      return;
    }

    const userQuestion = data.message;

    console.log("User Question: " + userQuestion);

    await conversationHistory.insertOne({
      role: "user",
      content: userQuestion,
      timestamp: new Date(),
    });

    const docs = await retriveRelevantDocuments(userQuestion);

    const context = docs.map((d) => d.pageContent).join("\n");
    console.log("Retrieved Context:", context);

    const result = await app.invoke(
      {
        Question: userQuestion,
        Context: context,
      },
      {
        configurable: {
          ws: ws,
        },
      },
    );
    console.log("Final Graph Result:", result);
    await conversationHistory.insertOne({
      role: "ai",
      content: result.Answer,
      timestamp: new Date(),
    });

    console.log("AI Answer : " + result.Answer);
  });
});

/*
Terminal testing

async function userInput() {

  return new Promise((resolve) => {

    const readLine =
      readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });

    readLine.on(
      "line",
      async (input) => {
        resolve(input);
        readLine.close();
      }
    );
  });
}
*/
