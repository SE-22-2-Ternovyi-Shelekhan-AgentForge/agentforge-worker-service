"""
Sends a test AgentSessionRequested message to agent-sessions-queue
and listens for results/events on agent-results-queue and agent-events-queue.

Usage:
    python send_test_message.py
"""
import json
import uuid
import time
import pika

RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
SESSIONS_QUEUE = "agent-sessions-queue"
RESULTS_QUEUE  = "agent-results-queue"
EVENTS_QUEUE   = "agent-events-queue"
ERRORS_QUEUE   = "agent-errors-queue"

session_id      = str(uuid.uuid4())
conversation_id = str(uuid.uuid4())

message = {
    "session_id":      session_id,
    "conversation_id": conversation_id,
    "user_prompt":     "What is 2 + 2? Reply briefly.",
    "history":         [],
    "team": {
        "supervisor_prompt": "You are a supervisor. Delegate the task to the assistant.",
        "agents": [
            {
                "role": "assistant",
                "system_prompt": "You are a helpful assistant. Answer concisely.",
                "tools": []
            }
        ],
        "max_iterations": 5
    }
}

connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
channel = connection.channel()

for q in [SESSIONS_QUEUE, RESULTS_QUEUE, EVENTS_QUEUE, ERRORS_QUEUE]:
    channel.queue_declare(queue=q, durable=True)

channel.basic_publish(
    exchange="",
    routing_key=SESSIONS_QUEUE,
    body=json.dumps(message),
    properties=pika.BasicProperties(delivery_mode=2),
)
print(f"[>] Sent session_id={session_id}")
print(f"[>] Waiting for response (Ctrl+C to stop)...\n")

received = {"count": 0}

def on_message(ch, method, props, body):
    received["count"] += 1
    data = json.loads(body)
    event = data.get("event_type") or data.get("event") or "message"
    print(f"[<] {method.routing_key}: {event} — {json.dumps(data, indent=2)[:300]}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(RESULTS_QUEUE, on_message)
channel.basic_consume(EVENTS_QUEUE,  on_message)
channel.basic_consume(ERRORS_QUEUE,  on_message)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    print(f"\n[i] Stopped. Received {received['count']} message(s).")
    channel.stop_consuming()

connection.close()
