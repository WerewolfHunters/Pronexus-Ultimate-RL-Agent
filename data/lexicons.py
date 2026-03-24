"""Domain lexicons used by signal scoring."""

HEDGE_PHRASES = [
    "maybe",
    "possibly",
    "generally",
    "typically",
    "it depends",
    "in many cases",
    "often",
    "usually",
    "can be",
    "might",
]

FAILURE_INDICATORS = [
    "failed",
    "failure",
    "mistake",
    "rollback",
    "outage",
    "incident",
    "postmortem",
    "rca",
    "broken",
    "downtime",
    "learned",
]

TEMPORAL_ANCHORS = [
    "yesterday",
    "last year",
    "last month",
    "q1",
    "q2",
    "q3",
    "q4",
    "sprint",
    "release",
    "version",
    "on-call",
]

VAGUE_TEMPORAL_PHRASES = [
    "recently",
    "at some point",
    "in the past",
    "over time",
    "eventually",
    "sometime",
    "previously",
]

TRIBAL_VOCAB = {
    "data_engineering": [
        "idempotent",
        "backfill",
        "late-arriving",
        "watermark",
        "schema evolution",
        "partition pruning",
        "exactly-once",
        "dead letter queue",
        "offset",
        "compaction",
    ],
    "backend": [
        "p99",
        "idempotency key",
        "retry storm",
        "thundering herd",
        "circuit breaker",
        "bulkhead",
        "cold start",
        "eventual consistency",
        "read replica",
        "distributed lock",
    ],
    "devops": [
        "blue green",
        "canary",
        "pod disruption budget",
        "liveness probe",
        "readiness probe",
        "hpa",
        "slo",
        "slis",
        "error budget",
        "node affinity",
    ],
}

KNOWN_TOOLS = [
    "kafka",
    "rabbitmq",
    "redis",
    "kubernetes",
    "docker",
    "spark",
    "airflow",
    "dbt",
    "postgres",
    "terraform",
]
