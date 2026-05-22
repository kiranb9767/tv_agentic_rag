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
import { MemorySaver } from "@langchain/langgraph";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";

import { MongoClient } from "mongodb";
import express from "express";
import http from "http";

import path from "path";
import { fileURLToPath } from "url";

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

const checkpointer =
  new MongoDBSaver({
    client: mongoClient,
    dbName: process.env.MONGODB_DB_NAME,
  });

const conversationHistory = db.collection("conversation_history");


const GraphState = Annotation.Root({
  Question: Annotation,
  RagContext: Annotation,
  WebContext: Annotation,
  FinalContext: Annotation,
  Answer: Annotation,
  Confidence: Annotation,
  error: Annotation,
  strategy: Annotation,
  ragQuery: Annotation,
  webQuery: Annotation,
  RetrievalScore: Annotation,
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
})
  .withRetry({
    stopAfterAttempt: 3,
  })
  .withConfig({
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

    const models = [...chapter.matchAll(/<diversity>(.*?)<\/diversity>/g)].map((m) => m[1],);

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
        new Document({
          pageContent: text,
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
      chunkSize: 1000,
      chunkOverlap: 150,
    });

    const chunks = await textSplitter.splitDocuments(documents);
    await index.deleteAll();
    await vectorStore.addDocuments(chunks);
    console.log("Upload successful!");
  } catch (e) {
    console.log("UPLOAD ERROR:", e);
  }
}

//await uploadDocuments();

async function retriveRelevantDocuments(query) {

  const results = await vectorStore.similaritySearchWithScore(query,5);
  const docs = results.map((r) => r[0]);
  const scores = results.map((r) => r[1]);

  console.log("RETRIEVAL SCORES:",scores);

  const topScore = scores[0] || 0;

  return {
    docs,
    topScore,
  };
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

    const retrievalScore = state.RetrievalScore || 0;

    const ragContext = state.RagContext || "";

    console.log("Planner Retrieval Score:",retrievalScore);

    const plannerResult =
      await groqLlm.invoke([

        new SystemMessage(`
          You are a smart query routing planner.

          Decide the BEST strategy.

          Available strategies:

          1. SMALL_TALK
             Use when user is:
             - greeting
             - introducing themselves
             - casual chatting
             - thanking
             - conversational reply
             - identity statement

             Examples:
             - hi
             - hello
             - my name is kiran
             - thanks
             - nice to meet you
             - how are you

          2. RAG_ONLY
             Use when:
             - Philips TV manual
             - settings
             - troubleshooting  
             - installation
             - device features
             - remote issues
             - HDMI/WiFi/display/audio
             - internal documentation enough

          3. WEB_ONLY
             Use when:
             - latest news
             - realtime/current info
             - public internet knowledge
             - weather
             - stock/news/sports

          4. RAG_AND_WEB
             Use when both manual docs and web help.

          IMPORTANT RULES:
          - Prefer SMALL_TALK for conversational input
          - Prefer RAG_ONLY for device/manual questions
          - Be conservative with WEB
          - Low retrieval score alone DOES NOT mean WEB_ONLY
          - Use retrieval score as supporting signal only

          Return ONLY valid JSON.

          Format:
          {"strategy":"SMALL_TALK|RAG_ONLY|WEB_ONLY|RAG_AND_WEB"}
        `),

        new HumanMessage(`
          USER QUESTION:${state.Question}
          RETRIEVAL SCORE:${retrievalScore}
          RETRIEVED MANUAL CONTEXT:${ragContext}`),
      ]);

    let parsed;
    try {
      const cleaned =
        plannerResult.content
          .replace(/```json/g, "")
          .replace(/```/g, "")
          .trim();

      parsed = JSON.parse(cleaned);
      console.log("Planner Parsed Result:",parsed);
    } catch (e) {
      console.log("Planner Parse Error:",e);
      parsed = {strategy: "RAG_ONLY",};
    }

    const finalStrategy =parsed.strategy?.trim() || "RAG_ONLY";

    console.log("Final Planner Strategy:",finalStrategy);
    return {
      ...state,
      strategy:finalStrategy,
      ragQuery:state.Question,
      webQuery:state.Question,
      RagContext:ragContext,
    };
  } catch (e) {
    console.log("PLANNER NODE ERROR:",e);
    return {
      ...state,
      strategy:"RAG_ONLY",
      ragQuery:state.Question,
      webQuery:state.Question,
      RagContext:state.RagContext || "",
    };
  }
}

async function smallTalkNode(state, config) {
  const ws = config?.configurable?.ws;

  let finalAnswer = "";

  try {
    const result = await groqLlm.stream([
      new SystemMessage(`
          You are a helpful assistant for Philips TV users.
          Answer the user's question in a friendly manner.
          Rules:
            1. Be friendly and helpful
            2. Keep answers concise
            3. Do not hallucinate`),
      new HumanMessage(`
          Question:${state.Question}`),
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
    finalAnswer = "Sorry, I'm having trouble answering that right now.";
  }

  return {
    ...state,
    Answer: finalAnswer,
    Confidence: 0.9,
  };
}

async function ragNode(state) {

  if (state.strategy === "WEB_ONLY") {
    return state;
  }
  try {
    console.log("RAG Node Invoked with RagQuery:",state.ragQuery);

    return {
      ...state,
      RagContext: state.RagContext || "",
    };
  } catch {
    console.log("RAG NODE ERROR:");
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
        query:
          state.webQuery ||
          state.Question,
      });

    console.log("Web Search Result:",webResult);

    let webContext = "";
    if (Array.isArray(webResult?.results)) {
      webContext =webResult.results.map((r) => `
                  content: ${r.content || ""}
                  url: ${r.url || ""}`).join("\n");
    } else {
      webContext = JSON.stringify(webResult);
    }
    if (state.strategy === "WEB_ONLY") {
      const result = await groqLlm.stream([
        new SystemMessage(`
            You are a helpful assistant.
            Answer the user's question using ONLY the provided web results.
            Rules:
              - Keep answer concise
              If URLs are available:
              - convert them into clickable HTML links
              Format:
              <a href="URL" target="_blank">Open Link</a>
            - Do not hallucinate
          - If answer unavailable say:"No reliable web information found."`),

        new HumanMessage(`
            Question:${state.Question}
            Web Results:${webContext}`),
      ]);

      for await (const chunk of result) {
        if (chunk.content) {
          finalAnswer +=chunk.content;

          if (ws) {
            ws.send(
              JSON.stringify({
                type: "stream",
                token:chunk.content,
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

      return {
        ...state,
        WebContext:webContext,
        Answer:finalAnswer,
        Confidence: 0.9,
      };
    }

    return {
      ...state,
      WebContext:webContext,
    };
  } catch (e) {
    console.log("WEB NODE ERROR:",e);

    return {
      ...state,
      WebContext: "",
      Answer:"Unable to fetch web information.",
      Confidence: 0.3,
    };
  }
}


function mergeNode(state) {

  return {
    ...state,
    FinalContext: 
    `MANUAL_CONTEXT:${state.RagContext || ""}
    WEB_CONTEXT:${state.WebContext || ""}`,
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
          You are a Philips TV manual assistant.
          You MUST answer using ONLY the provided context.

          IMPORTANT RULES:
          - Do NOT use your own knowledge
          - Do NOT ignore provided context
          - Treat context as source of truth
          - Keep concise and user-friendly

          If answer exists:
          - explain naturally
          If answer unavailable:
          - reply:
         "Information not available in the provided manual."`),

      new HumanMessage(`
          Question:${state.Question}
          Context:${state.FinalContext}`),
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
    Confidence:finalAnswer.includes("NOT_SURE")? 0.3: 0.9,
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
  .addNode("SMALL_TALK", smallTalkNode)
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
  SMALL_TALK: "SMALL_TALK",
});

graph.addConditionalEdges(
  "RAG",
  (state) =>
    (state.strategy === "RAG_AND_WEB" ? "WEB": "MERGE"),
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

const app = graph.compile({
    checkpointer: checkpointer,
});


wss.on("connection", async (ws) => {

  console.log(
    "Web soc server connected......"
  );

  ws.on("message", async (message) => {

    try {

      console.log(
        "Web soc on msg receive......"
      );
      const data =
        JSON.parse(message.toString());

      const sessionId =
        data.sessionId;

      if (!sessionId) {

        console.log(
          "Session ID missing"
        );

        return;
      }


      if (!ws.historySent) {

        const history =
          await conversationHistory
            .find({
              sessionId:
                sessionId,
            })
            .sort({
              timestamp: 1,
            })
            .toArray();

        console.log(
          "Conversation History:",
          history
        );

        ws.send(
          JSON.stringify({
            type: "history",
            message: history,
          })
        );

        ws.historySent = true;
      }


      if (
        data.type ===
        "clear_history"
      ) {

        await conversationHistory
          .deleteMany({
            sessionId:
              sessionId,
          });

        console.log(
          "Conversation history cleared."
        );

        return;
      }



      if (
        data.type !==
        "question"
      ) {

        console.log(
          "Invalid message type"
        );

        return;
      }


      const userQuestion =
        data.message;

      if (!userQuestion) {

        console.log(
          "Empty question"
        );

        return;
      }

      console.log(
        "User Question:",
        userQuestion
      );



      await conversationHistory
        .insertOne({

          sessionId:
            sessionId,

          role: "user",

          content:
            userQuestion,

          timestamp:
            new Date(),
        });



      const retrievalResult =
        await retriveRelevantDocuments(
          userQuestion
        );

      const docs =
        retrievalResult.docs;

      const topScore =
        retrievalResult.topScore;

      const context =
        docs
          .map(
            (d) =>
              d.pageContent
          )
          .join("\n");

      console.log(
        "Retrieved Context:",
        context
      );

      console.log(
        "Top Retrieval Score:",
        topScore
      );

      const result =
        await app.invoke(
          {
            Question:
              userQuestion,

            RagContext:
              context,

            RetrievalScore:
              topScore,
          },
          {
           configurable: {
              thread_id: String(sessionId),
              checkpoint_ns: "chat",
              ws: ws,
        },
          }
        );

      console.log(
        "Final Graph Result:",
        result
      );


      await conversationHistory
        .insertOne({

          sessionId:
            sessionId,

          role: "ai",

          content:
            result.Answer,

          timestamp:
            new Date(),
        });

      console.log(
        "AI Answer:",
        result.Answer
      );

    } catch (e) {

      console.log(
        "WS MESSAGE ERROR:",
        e
      );

      try {

        ws.send(
          JSON.stringify({
            type: "stream",
            token:
              "Something went wrong.",
          })
        );

        ws.send(
          JSON.stringify({
            type: "done",
          })
        );

      } catch {}
    }
  });
});