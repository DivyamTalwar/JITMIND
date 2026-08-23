"""Measure the in-process cost of JITMIND's dependency-free trace recorder."""

from time import perf_counter

from jitmind.observability import TraceRecorder


def main(events: int = 10_000) -> None:
    recorder = TraceRecorder("benchmark query")
    started = perf_counter()
    for index in range(events):
        recorder.record(
            "retrieval.fusion",
            input_count=20,
            output_count=5,
            attributes={"iteration": index},
        )
    elapsed_ms = (perf_counter() - started) * 1_000
    recorder.finish()
    print(
        f"events={events} elapsed_ms={elapsed_ms:.3f} "
        f"microseconds_per_event={elapsed_ms * 1000 / events:.3f}"
    )


if __name__ == "__main__":
    main()

