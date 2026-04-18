import pika
from pika.adapters.blocking_connection import BlockingChannel
from pydantic import BaseModel


def publish_event(channel: BlockingChannel, queue: str, event: BaseModel) -> None:
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=event.model_dump_json().encode("utf-8"),
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=pika.DeliveryMode.Persistent,
        ),
    )
