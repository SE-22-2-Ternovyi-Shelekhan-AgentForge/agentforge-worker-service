import pika
from pika.adapters.blocking_connection import BlockingChannel

from agentforge_worker.config import settings
from agentforge_worker.contracts import (
    AgentEventOccurred,
    AgentSessionCompleted,
    AgentSessionFailed,
)


def _publish(channel: BlockingChannel, queue: str, body: bytes) -> None:
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=body,
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=pika.DeliveryMode.Persistent,
        ),
    )


def publish_event(channel: BlockingChannel, evt: AgentEventOccurred) -> None:
    _publish(channel, settings.events_queue, evt.model_dump_json().encode("utf-8"))


def publish_completed(channel: BlockingChannel, completed: AgentSessionCompleted) -> None:
    _publish(channel, settings.results_queue, completed.model_dump_json().encode("utf-8"))


def publish_failed(channel: BlockingChannel, failed: AgentSessionFailed) -> None:
    _publish(channel, settings.errors_queue, failed.model_dump_json().encode("utf-8"))
