"""Central experiment configuration. Every run uses identical settings for all policies."""
SEED = 42
SIZES = [50, 200, 500]          # >=3 independent workload sizes (processes per scenario)
SCENARIOS_PER_SIZE = 300        # number of independent workload scenarios per size
CS_COST = 0.5                   # context-switch overhead (time units), identical for all policies
CV_FOLDS = 5
CV_REPEATS = 3

# Candidate scheduling policies the ML model chooses between: name -> (algorithm, quantum)
POLICIES = {
    "FCFS":  ("FCFS", None),
    "SJF":   ("SJF", None),
    "RR_q2": ("RR", 2),
    "RR_q5": ("RR", 5),
    "RR_q10": ("RR", 10),
    "RR_q25": ("RR", 25),
    "RR_q50": ("RR", 50),
}

# Policies the ML model may choose between (FCFS = infinite quantum, RR_q* = time-quantum choice).
# SJF and RR_q2 are simulated too but kept OUT of the label set: SJF needs advance knowledge of burst
# lengths and dominates mean waiting, which would make the prediction task trivial. They are reported
# as fixed baselines / reference oracle instead.
LABEL_POLICIES = ["FCFS", "RR_q5", "RR_q10", "RR_q25", "RR_q50"]

# Cost used to decide the best policy for a scenario (lower = better): efficiency + responsiveness + tail fairness
COST_WEIGHTS = {"avg_wait": 1.0, "avg_resp": 1.0, "p95_wait": 1.0}

# Fixed rule-based baselines the ML selector is compared with
FIXED_BASELINES = ["FCFS", "SJF", "RR_q10"]
