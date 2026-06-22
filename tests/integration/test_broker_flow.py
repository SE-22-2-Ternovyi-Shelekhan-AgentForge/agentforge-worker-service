from uuid import uuid4

import pika
import pytest
from testcontainers.rabbitmq import RabbitMqContainer

from agentforge_worker.contracts import AgentEventOccurred
from agentforge_worker.messaging.publisher import publish_event

_EVENTS_Q = "agent-events-queue"
_RESULTS_Q = "agent-results-queue"
_ERRORS_Q = "agent-errors-queue"


@pytest.fixture(scope="module")
def rmq():
    with RabbitMqContainer("rabbitmq:3.13-management") as container:
        yield container


@pytest.fixture(scope="module")
def channel(rmq):
    conn = pika.BlockingConnection(pika.URLParameters(rmq.get_connection_url()))
    ch = conn.channel()
    for q in (_EVENTS_Q, _RESULTS_Q, _ERRORS_Q):
        ch.queue_declare(queue=q, durable=True)
    yield ch
    conn.close()


def test_publish_event_roundtrip(channel, monkeypatch):
    sid = uuid4()
    cid = uuid4()

    import agentforge_worker.messaging.publisher as pub_mod
    import agentforge_worker.config as cfg_mod

    monkeypatch.setattr(cfg_mod.settings, "events_queue", _EVENTS_Q)

    evt = AgentEventOccurred(
        session_id=sid,
        conversation_id=cid,
        event_type="supervisor_routed",
        payload={"next": "researcher", "reasoning": "needs research"},
    )
    publish_event(channel, evt)

    method, _props, body = channel.basic_get(queue=_EVENTS_Q, auto_ack=True)
    assert method is not None, "No message received from queue"

    decoded = AgentEventOccurred.model_validate_json(body)
    assert decoded.session_id == sid
    assert decoded.conversation_id == cid
    assert decoded.event_type == "supervisor_routed"
    assert decoded.payload["next"] == "researcher"
    assert decoded.payload["reasoning"] == "needs research"
