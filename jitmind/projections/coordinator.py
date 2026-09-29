"""Host-owned worker; no worker actions are invoked by projection readers."""

from .models import ProjectionError, integer, selection


class ProjectionCoordinator:
    def __init__(self, authority, source, consumers):
        if type(consumers) not in (tuple, list) or not 1 <= len(consumers) <= 16:
            raise ProjectionError("invalid_consumers")
        if len({consumer.path for consumer in consumers}) != len(consumers):
            raise ProjectionError("duplicate_consumer")
        self.authority, self.source, self.consumers = (
            authority,
            source,
            tuple(consumers),
        )

    def deliver(self, scope, event, *, snapshots=()):
        selection(self.authority, scope, snapshots)
        outcomes = {}
        for consumer in self.consumers:
            try:
                consumer.deliver(
                    self.authority, scope, self.source, event, snapshots=snapshots
                )
                outcomes[str(consumer.path)] = {"state": "done"}
            except ProjectionError as exc:
                outcomes[str(consumer.path)] = {"state": "retry", "reason": exc.code}
        return outcomes

    def drain(self, scope, *, snapshots=(), limit=100):
        selection(self.authority, scope, snapshots)
        integer(limit, 1, 1000)
        outcomes = {}
        for consumer in self.consumers:
            try:
                state = consumer.status(
                    self.authority, scope, self.source, snapshots=snapshots
                )
                events = self.source.events(
                    self.authority,
                    scope,
                    after_revision=state["watermark"],
                    limit=limit,
                )
                for event in events:
                    consumer.deliver(
                        self.authority, scope, self.source, event, snapshots=snapshots
                    )
                outcomes[str(consumer.path)] = consumer.status(
                    self.authority, scope, self.source, snapshots=snapshots
                )
            except ProjectionError as exc:
                outcomes[str(consumer.path)] = {
                    "status": "unavailable",
                    "reasons": (exc.code,),
                }
        return outcomes

    def rebuild(self, scope, *, snapshots=(), limit=1000):
        selection(self.authority, scope, snapshots)
        results = {}
        for consumer in self.consumers:
            try:
                consumer.rebuild(
                    self.authority, scope, self.source, snapshots=snapshots, limit=limit
                )
                results[str(consumer.path)] = consumer.status(
                    self.authority, scope, self.source, snapshots=snapshots
                )
            except ProjectionError as exc:
                results[str(consumer.path)] = {
                    "status": "unavailable",
                    "reasons": (exc.code,),
                }
        return results
