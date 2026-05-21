const chat = document.getElementById("chat");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");

const clearHistoryBtn = document.getElementById("clearHistoryBtn");
// WebSocket connection

const ws = new WebSocket(
  window.location.origin.replace("http", "ws")
);

ws.onopen = () => {
  console.log("Connected to AI backend");
};

//Ws on msg receive
let aiMessageDiv = null;
ws.onmessage = (event) => {
  console.log("On msg receive from AI......");
  const data = JSON.parse(event.data);
  if (data.type === "answer") {
    addMessage("AI", data.message, "ai");
  } else if (data.type === "history") {
    data.message.forEach((msg) => {
      const sender = msg.role === "user" ? "User" : "AI";
      const type = msg.role === "user" ? "user" : "ai";
      addMessage(sender, msg.content, type);
    });
  } else if (data.type === "stream") {
    if (!aiMessageDiv) {
      aiMessageDiv = document.createElement("div");
      aiMessageDiv.className = "message ai";
      aiMessageDiv.innerHTML = "AI: ";
      chat.appendChild(aiMessageDiv);
    }

    aiMessageDiv.textContent += data.token;

    chat.scrollTop = chat.scrollHeight;
  } else if(data.type === "done"){
    if(aiMessageDiv){
        aiMessageDiv.innerHTML = aiMessageDiv.textContent;
    }
    aiMessageDiv = null;
}
};


sendBtn.addEventListener("click", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();  
    }
});
sendBtn.addEventListener("enter", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();  
    }
});
sendBtn.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();   
    }
});
clearHistoryBtn.addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "clear_history" }));
  chat.innerHTML = "";
});




function sendMessage() {
  const text = input.value.trim();
  addMessage("User", text, "user");
  ws.send(JSON.stringify({ type: "question", message: text }));
  input.value = "";
}

// Display AI response

function addMessage(sender, text, type) {
  const div = document.createElement("div");
  div.className = "message " + type;
  div.innerHTML = sender + ": " + text;
  div.querySelectorAll("a").forEach(a => {
    a.setAttribute("tabindex", "-1");
  });
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}
