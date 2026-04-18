# agentforge-worker-service

Python worker for the AgentForge platform. A stateless executor of LLM tasks for role-based agents (planner, coder, reviewer, ...) that consumes tasks from RabbitMQ, runs them through a local LLM (Ollama) via LangChain, and publishes results back to RabbitMQ. Conversation state and routing between agents are owned by the orchestrator — the worker never talks to a database directly.

## Architecture

- **RabbitMQ is the only interface.** The worker holds a persistent AMQP connection. It does not accept HTTP.
- **Stateless.** Nothing is kept between tasks. Run N replicas and RabbitMQ fans messages out via Competing Consumers.
- **Inter-agent communication** happens *through the orchestrator*: agent A's output is placed into the next task's `context` for agent B. Each context entry carries `from_agent_role`, so the LLM can tell who said what.
- **Prompts arrive in the message.** The worker does not store role templates or build prompts itself — `system_prompt`, `user_prompt`, and `context` all come from the task message.

## Integration contract

The worker consumes from `agent-tasks-queue` and publishes to `agent-results-queue` / `agent-errors-queue`. All messages are JSON, UTF-8, `content_type: application/json`, persistent delivery.

### Queues

| Queue                  | Direction             | Purpose                                 |
| ---------------------- | --------------------- | --------------------------------------- |
| `agent-tasks-queue`    | producer → **worker** | Task requests for agents                |
| `agent-results-queue`  | **worker** → consumer | Successful task completions             |
| `agent-errors-queue`   | **worker** → consumer | Failures (LLM errors, parse errors)     |

All three queues are declared `durable=true` by the worker on startup, so the producer side does not need to pre-declare them.

### Input: `AgentTaskRequested` (publish to `agent-tasks-queue`)

```json
{
  "task_id": "11111111-1111-1111-1111-111111111111",
  "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  "agent_role": "planner",
  "system_prompt": "You are a planner. Break the task into 3 concrete steps.",
  "user_prompt": "Build a login form",
  "context": [],
  "model": "llama3.1:8b",
  "temperature": 0.2
}
```

| Field              | Type                       | Required | Notes                                                                                   |
| ------------------ | -------------------------- | -------- | --------------------------------------------------------------------------------------- |
| `task_id`          | UUID                       | yes      | Unique per task. Echoed back in results/errors — use it to correlate.                   |
| `conversation_id`  | UUID                       | yes      | Groups tasks of a multi-agent conversation. Echoed back; used to stitch steps.          |
| `agent_role`       | string                     | yes      | Informational. Echoed back. The worker does not branch on it.                           |
| `system_prompt`    | string                     | yes      | System instruction for the current agent.                                               |
| `user_prompt`      | string                     | yes      | Current user message.                                                                   |
| `context`          | array of `ContextEntry`    | no       | Previous steps from other agents. See below.                                            |
| `model`            | string                     | no       | Ollama model tag (e.g. `llama3.1:8b`). Defaults to `DEFAULT_MODEL` env var.             |
| `temperature`      | float                      | no       | Defaults to `DEFAULT_TEMPERATURE` env var.                                              |

`ContextEntry`:

```json
{
  "from_agent_role": "planner",
  "content": "Step 1: ...\nStep 2: ...",
  "message_type": "assistant"
}
```

`message_type` is one of `user`, `assistant`, `system`. The worker prefixes every context entry with `[<from_agent_role>]: ` before feeding it to the LLM, so downstream agents can tell handoffs apart.

### Output: `TaskCompletedEvent` (from `agent-results-queue`)

```json
{
  "task_id": "11111111-1111-1111-1111-111111111111",
  "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  "agent_role": "planner",
  "output": "Step 1: ...\nStep 2: ...\nStep 3: ...",
  "model_used": "llama3.1:8b",
  "tokens_in": 128,
  "tokens_out": 256,
  "completed_at": "2026-04-18T12:34:56.789Z"
}
```

`tokens_in` / `tokens_out` may be `null` if the underlying model does not report usage.

### Output: `TaskFailedEvent` (from `agent-errors-queue`)

```json
{
  "task_id": "11111111-1111-1111-1111-111111111111",
  "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  "agent_role": "planner",
  "error_type": "llm_timeout",
  "error_message": "TimeoutError: ...",
  "failed_at": "2026-04-18T12:34:56.789Z"
}
```

`error_type` values: `llm_timeout`, `llm_connection_error`, `llm_api_error`, `parse_error`. For `parse_error` (an unparseable task body) `task_id` / `conversation_id` will be the zero UUID `00000000-0000-0000-0000-000000000000`.

### Delivery semantics

- Tasks that produce a result or a classified error are ACKed.
- Unexpected exceptions are NACKed with `requeue=false` to avoid poison-message loops.
- The worker runs with `prefetch_count=1` by default, so tasks distribute fairly across replicas.

## Running the worker locally

```bash
cp .env.example .env
docker compose up --build -d
# First run only: pull the default model into the Ollama volume.
docker exec -it agentforge-ollama ollama pull llama3.1:8b
```

The stack includes RabbitMQ (`localhost:5672`, management UI at `localhost:15672`, `guest` / `guest`), Ollama (`localhost:11434`), and two worker replicas.

> `docker-compose.yml` declares `deploy.replicas: 2`. If your Compose version ignores that directive for `up`, scale explicitly: `docker compose up --build --scale worker=2 -d`.

Follow logs: `docker compose logs -f worker`.

## Configuration

Everything is env-driven. See `.env.example`:

| Variable                | Default                              | Purpose                        |
| ----------------------- | ------------------------------------ | ------------------------------ |
| `RABBITMQ_URL`          | `amqp://guest:guest@rabbitmq:5672/`  | AMQP URL                       |
| `TASKS_QUEUE`           | `agent-tasks-queue`                  | Input queue                    |
| `RESULTS_QUEUE`         | `agent-results-queue`                | Success output queue           |
| `ERRORS_QUEUE`          | `agent-errors-queue`                 | Failure output queue           |
| `OLLAMA_BASE_URL`       | `http://ollama:11434`                | Ollama HTTP endpoint           |
| `DEFAULT_MODEL`         | `llama3.1:8b`                        | Fallback model tag             |
| `DEFAULT_TEMPERATURE`   | `0.2`                                | Fallback temperature           |
| `LLM_TIMEOUT_SECONDS`   | `120`                                | Per-invocation LLM timeout     |
| `LOG_LEVEL`             | `INFO`                               | structlog level                |
| `PREFETCH_COUNT`        | `1`                                  | AMQP `basic.qos`               |

## End-to-end smoke test

### Step 1 — `planner` agent

In the RabbitMQ management UI (`localhost:15672` → Queues → `agent-tasks-queue` → Publish message), set `content_type: application/json` and publish:

```json
{
  "task_id": "11111111-1111-1111-1111-111111111111",
  "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  "agent_role": "planner",
  "system_prompt": "You are a planner. Break the task into 3 concrete steps.",
  "user_prompt": "Build a login form",
  "context": []
}
```

Then Get Message from `agent-results-queue` — a `TaskCompletedEvent` with the same `task_id` and `conversation_id` should be waiting.

### Step 2 — `coder` agent, receiving the planner's output as context

```json
{
  "task_id": "22222222-2222-2222-2222-222222222222",
  "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  "agent_role": "coder",
  "system_prompt": "You are a coder. Implement the first step from the plan.",
  "user_prompt": "Implement step 1",
  "context": [
    {
      "from_agent_role": "planner",
      "content": "<paste the planner's output from step 1 here>",
      "message_type": "assistant"
    }
  ]
}
```

Worker logs will show the context line rendered as `[planner]: ...`, confirming the handoff.

### Verifying fan-out

Publish 4+ tasks back to back — logs from both worker replicas will show them interleaving.

### Verifying error path

Publish a task with `"model": "nonexistent:1b"` — a `TaskFailedEvent` lands in `agent-errors-queue`.

## Project layout

```
src/agentforge_worker/
├── config.py                # Pydantic Settings (env vars)
├── logging_config.py        # structlog → JSON
├── main.py / __main__.py    # entrypoint
├── contracts/               # AgentTaskRequested, TaskCompletedEvent, TaskFailedEvent
├── llm/client.py            # LangChain + ChatOllama
├── messaging/consumer.py    # pika consumer (ack/nack, reconnect)
├── messaging/producer.py    # event publisher
└── handlers/task_handler.py # parse → LLM → publish
```

## Not yet implemented

Retry policies / DLQ, streaming responses, tool / function calling, OpenTelemetry metrics, unit tests, alternative LLM providers (OpenAI / Anthropic).
