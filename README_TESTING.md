# Testing the MCP AI Agent

## Prerequisites

1. Make sure Redis is running:
   ```bash
   redis-server
   # Or with Docker:
   docker run -d -p 6379:6379 redis
   ```

2. Make sure you have your OpenAI API key in `.env`:
   ```bash
   OPENAI_API_KEY=your_actual_key_here
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Running the Server

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

You can verify it's running by visiting:
- API docs: http://localhost:8000/docs
- MCP endpoint: http://localhost:8000/agent/mcp

## Running Tests

In a **new terminal window** (keep the server running), activate the venv and run:

```bash
source .venv/bin/activate
python test.py
```

### Test Modes

**1. Automated Test Suite**
- Runs 5 predefined test queries
- Tests Wikipedia news tool
- Tests country details tool
- Tests agent's reasoning capabilities

**2. Interactive Mode**
- Ask your own questions
- See real-time responses
- Type `exit` or `quit` to stop

## Example Test Queries

Try these in interactive mode:

```
What is the latest news about climate change?
Tell me about Germany
What's the capital of Italy and give me details about the country?
Give me news about space exploration
What are the details about Canada?
```

## Expected Behavior

The agent should:
1. Understand your question
2. Choose the appropriate tool(s):
   - `global_news` for news/Wikipedia queries
   - `get_countries_details` for country information
3. Execute the tool(s)
4. Return a formatted response

## Troubleshooting

**Connection Error**
```
Error: Cannot connect to the server
```
→ Make sure the server is running (`uvicorn main:app --reload`)

**Timeout Error**
```
Error: Request timed out
```
→ The LLM might be taking time. Check your OpenAI API key and quota.

**Redis Error**
```
Connection refused to Redis
```
→ Start Redis: `redis-server` or use Docker

**OpenAI API Error**
```
Error: Invalid API key
```
→ Check your `.env` file has the correct `OPENAI_API_KEY`

## Manual Testing with curl

You can also test directly with curl:

```bash
curl -X POST "http://localhost:8000/workflow?message=Tell%20me%20about%20France"
```

Or use the interactive API docs at http://localhost:8000/docs
