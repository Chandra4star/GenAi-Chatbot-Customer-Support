# AI Customer Support Chatbot — Full Prototype (Java Spring Boot + Python RAG + React)

This single-document project contains a minimal, runnable prototype with these parts:

* `backend/` — Spring Boot (Java) API that receives chat messages and proxies to the RAG microservice.
* `rag-service/` — Python Flask microservice that performs simple retrieval (file-based) + OpenAI-powered generation.
* `frontend/` — Minimal React + HTML chat widget to talk to the backend.
* `docker-compose.yml` and README with run instructions.

> **Important:** Before running, set environment variables (see README). This prototype is intentionally simple and meant as a starting point. Do not use in production without security, rate-limiting, PII handling, and proper secrets management.

---

## Project structure

```
ai-support-prototype/
├── backend/
│   ├── pom.xml
│   └── src/main/java/com/example/chat/ (Java files)
├── rag-service/
│   ├── requirements.txt
│   └── app.py
├── frontend/
│   └── index.html
└── docker-compose.yml
```

---

## 1) Backend (Spring Boot) — `backend/`

### `pom.xml`

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>chat-backend</artifactId>
  <version>0.0.1-SNAPSHOT</version>
  <packaging>jar</packaging>
  <properties>
    <java.version>17</java.version>
    <spring.boot.version>3.2.0</spring.boot.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter</artifactId>
    </dependency>
    <dependency>
      <groupId>com.fasterxml.jackson.core</groupId>
      <artifactId>jackson-databind</artifactId>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-maven-plugin</artifactId>
      </plugin>
    </plugins>
  </build>
</project>
```

### `src/main/java/com/example/chat/ChatApplication.java`

```java
package com.example.chat;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ChatApplication {
    public static void main(String[] args) {
        SpringApplication.run(ChatApplication.class, args);
    }
}
```

### `src/main/java/com/example/chat/controller/ChatController.java`

```java
package com.example.chat.controller;

import com.example.chat.model.ChatRequest;
import com.example.chat.model.ChatResponse;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String ragUrl = System.getenv().getOrDefault("RAG_SERVICE_URL", "http://localhost:5000/rag");

    @PostMapping("/message")
    public ResponseEntity<ChatResponse> message(@RequestBody ChatRequest req) {
        // forward to RAG microservice
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<ChatRequest> entity = new HttpEntity<>(req, headers);

        ResponseEntity<ChatResponse> resp = restTemplate.postForEntity(ragUrl, entity, ChatResponse.class);
        return ResponseEntity.ok(resp.getBody());
    }
}
```

### `src/main/java/com/example/chat/model/ChatRequest.java`

```java
package com.example.chat.model;

public class ChatRequest {
    private String sessionId;
    private String message;

    public ChatRequest() {}

    public ChatRequest(String sessionId, String message) {
        this.sessionId = sessionId;
        this.message = message;
    }

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}
```

### `src/main/java/com/example/chat/model/ChatResponse.java`

```java
package com.example.chat.model;

public class ChatResponse {
    private String reply;
    private double confidence;

    public ChatResponse() {}

    public ChatResponse(String reply, double confidence) {
        this.reply = reply;
        this.confidence = confidence;
    }

    public String getReply() { return reply; }
    public void setReply(String reply) { this.reply = reply; }
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
}
```

---

## 2) RAG microservice (Python) — `rag-service/`

This microservice is intentionally small: it loads a few text files as a naive KB, computes embeddings (OpenAI embeddings), finds top matches by cosine similarity, and asks the OpenAI completions/Chat API to produce an answer using only the retrieved context.

> **Set env var** `OPENAI_API_KEY` before running.

### `requirements.txt`

```
flask
openai
numpy
scipy
requests
```

### `app.py`

```python
from flask import Flask, request, jsonify
import os
import openai
import numpy as np
from numpy.linalg import norm

openai.api_key = os.environ.get('OPENAI_API_KEY')

app = Flask(__name__)

# --- tiny file-based KB: load all .txt in ./kb/
KB_DIR = os.path.join(os.path.dirname(__file__), 'kb')
kb_texts = []
kb_ids = []

if os.path.isdir(KB_DIR):
    for fname in os.listdir(KB_DIR):
        if fname.endswith('.txt'):
            path = os.path.join(KB_DIR, fname)
            with open(path, 'r', encoding='utf-8') as f:
                kb_texts.append(f.read())
                kb_ids.append(fname)

# compute embeddings for KB
print('KB docs loaded:', len(kb_texts))
kb_embeddings = []

if kb_texts:
    for doc in kb_texts:
        resp = openai.Embedding.create(input=doc, model='text-embedding-3-small')
        emb = np.array(resp['data'][0]['embedding'], dtype=float)
        kb_embeddings.append(emb)


def top_k_docs(query, k=3):
    # embed query
    resp = openai.Embedding.create(input=query, model='text-embedding-3-small')
    q_emb = np.array(resp['data'][0]['embedding'], dtype=float)
    if not kb_embeddings:
        return []
    sims = [float(np.dot(q_emb, d) / (norm(q_emb) * norm(d))) for d in kb_embeddings]
    idx = np.argsort(sims)[::-1][:k]
    results = [{'id': kb_ids[i], 'text': kb_texts[i], 'score': float(sims[i])} for i in idx]
    return results


@app.route('/rag', methods=['POST'])
def rag():
    data = request.json
    query = data.get('message', '')

    top_docs = top_k_docs(query, k=3)
    context = '\n\n'.join([f"[DOC:{d['id']}]\n{d['text']}" for d in top_docs])

    system_prompt = (
        "You are a helpful customer support assistant. Use only the provided documents to answer. "
        "If the answer is not contained, say 'I don't know; please contact support.' Keep answers short."
    )

    user_msg = f"Context:\n{context}\n\nUser question: {query}\nAnswer concisely and cite the doc ids used."

    # call Chat Completions
    chat_resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=300,
        temperature=0.0
    )

    text = chat_resp['choices'][0]['message']['content']

    # create a simple confidence heuristic: average of top doc scores (0..1)
    confidence = 0.0
    if top_docs:
        confidence = sum([d['score'] for d in top_docs]) / len(top_docs)

    return jsonify({'reply': text, 'confidence': confidence})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
```

### `rag-service/kb/sample-faq.txt`

```
How do I reset my password?
To reset your password go to Settings > Security > Reset password. We will send a verification link to your registered email.

What is the refund policy?
You may request a refund within 14 days of purchase if you have not downloaded the product.
```

---

## 3) Frontend — `frontend/index.html`

A minimal static HTML + React widget (no build step) that talks to the backend `/api/chat/message`.

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Support Widget - Prototype</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; }
    .widget { width: 420px; border: 1px solid #ddd; padding: 12px; border-radius: 8px; }
    .messages { height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 8px; }
    .msg-user { text-align: right; color: #111; }
    .msg-bot { text-align: left; color: #0b5; }
  </style>
</head>
<body>
  <div id="root"></div>

  <script>
    const e = React.createElement;

    function App(){
      const [messages, setMessages] = React.useState([]);
      const [text, setText] = React.useState('');

      function send(){
        if(!text.trim()) return;
        const userMsg = {from: 'user', text};
        setMessages(prev => [...prev, userMsg]);
        const payload = { sessionId: 'demo-session', message: text };
        setText('');

        fetch('/api/chat/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        }).then(r => r.json()).then(data => {
          setMessages(prev => [...prev, {from:'bot', text: data.reply || 'No response'}]);
        }).catch(err => {
          setMessages(prev => [...prev, {from:'bot', text: 'Error: '+err.message}]);
        })
      }

      return e('div', {className:'widget'},
        e('h3', null, 'AI Support (Prototype)'),
        e('div', {className:'messages'}, messages.map((m,i) => e('div', {key:i, className: m.from === 'user' ? 'msg-user' : 'msg-bot'}, m.text))),
        e('div', null,
          e('input', {value:text, onChange: (e)=>setText(e.target.value), style: {width: '78%'}}),
          e('button', {onClick: send, style: {width:'20%'}}, 'Send')
        )
      )
    }

    ReactDOM.render(e(App), document.getElementById('root'));
  </script>
</body>
</html>
```

---

## 4) `docker-compose.yml`

```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - RAG_SERVICE_URL=http://rag-service:5000/rag
    depends_on:
      - rag-service

  rag-service:
    build:
      context: ./rag-service
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  # frontend is static, you can serve with any static file server or open index.html directly
```

Note: The `backend` service assumes there's a Dockerfile in `backend/` and `rag-service` has one too. For a quick local run you can start the Python rag-service directly and run Spring Boot from your IDE.

---

## 5) README — run steps (quickstart)

1. Set OpenAI API key (Linux/Mac):

```bash
export OPENAI_API_KEY="sk-..."
```

2. Start the RAG microservice (quick, without docker):

```bash
cd rag-service
pip install -r requirements.txt
# create a folder rag-service/kb and place sample-faq.txt there (already included in this doc). Then run:
python app.py
```

3. Start the Spring Boot backend (from IDE or Maven):

```bash
cd backend
./mvnw spring-boot:run
# or with your IDE run ChatApplication
```

4. Open `frontend/index.html` in your browser (or host it via a static server). Type messages; backend will call rag service which calls OpenAI.

---

## Notes, limitations, and next steps

* This prototype uses OpenAI and sends doc contents to the LLM; for production you must scrub PII or use private models.
* The KB is file-based and small — replace with a vector DB (Pinecone/Milvus/Chroma) for scale.
* Add authentication, rate-limiting, caching, and monitoring.
* Improve prompts, add citation formatting, and build an admin UI to maintain KB.

---

If you'd like, I can now:

* convert this into a Git repo with files (I can create a zip file via python\_user\_visible), OR
* provide a full Dockerfile for backend and rag-service, OR
* replace the OpenAI calls with a placeholder so you can run offline. Tell me which next step you want.
