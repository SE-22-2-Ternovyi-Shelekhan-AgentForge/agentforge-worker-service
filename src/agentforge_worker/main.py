from agentforge_worker.config import settings
from agentforge_worker.logging_config import configure_logging
from agentforge_worker.messaging.consumer import start_consumer


def main() -> None:
    configure_logging(settings.log_level)
    start_consumer()


if __name__ == "__main__":
    main()
