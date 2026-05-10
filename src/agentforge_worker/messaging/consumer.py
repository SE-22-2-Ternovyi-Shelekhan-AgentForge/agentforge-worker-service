import time

import pika
import structlog
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import AMQPConnectionError

from agentforge_worker.config import settings
from agentforge_worker.handlers.session_handler import handle_session

log = structlog.get_logger(__name__)


def _declare_queues(channel: BlockingChannel) -> None:
    for queue in (
        settings.sessions_queue,
        settings.events_queue,
        settings.results_queue,
        settings.errors_queue,
    ):
        channel.queue_declare(queue=queue, durable=True)


def _on_message(channel: BlockingChannel, method, properties, body: bytes) -> None:  # noqa: ARG001
    try:
        handle_session(channel, body)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as exc:  # noqa: BLE001
        # handle_session already publishes AgentSessionFailed for recognized errors. This catch is
        # the last-resort net for unexpected bugs: NACK without requeue so we don't spin on a
        # poison message, and rely on the surrounding log entry for triage.
        log.exception("consumer.unhandled_error", error=str(exc))
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def _run_once() -> None:
    params = pika.URLParameters(settings.rabbitmq_url)
    connection = pika.BlockingConnection(params)
    try:
        channel = connection.channel()
        _declare_queues(channel)
        channel.basic_qos(prefetch_count=settings.prefetch_count)
        channel.basic_consume(
            queue=settings.sessions_queue,
            on_message_callback=_on_message,
            auto_ack=False,
        )
        log.info(
            "consumer.started",
            queue=settings.sessions_queue,
            prefetch=settings.prefetch_count,
            ollama=settings.ollama_base_url,
            default_model=settings.default_model,
            supervisor_model=settings.supervisor_model,
        )
        channel.start_consuming()
    finally:
        if connection.is_open:
            connection.close()


def start_consumer() -> None:
    backoff = 2.0
    while True:
        try:
            _run_once()
            return
        except AMQPConnectionError as exc:
            log.warning("consumer.connection_failed", error=str(exc), retry_in=backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        except KeyboardInterrupt:
            log.info("consumer.stopped")
            return
