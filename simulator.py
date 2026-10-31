"""Event-driven single-CPU scheduling simulator with alternating CPU / I/O bursts.
Supports FCFS, SJF (non-preemptive, by next CPU burst) and Round Robin (quantum q)."""
import heapq
from collections import deque
import numpy as np


def simulate(procs, algo, quantum=None, cs_cost=0.5):
    n = len(procs)
    cpu = [p["cpu"] for p in procs]
    io = [p["io"] for p in procs]
    arr = [p["arrival"] for p in procs]
    idx = [0] * n
    rem = [cpu[i][0] for i in range(n)]
    first = [-1.0] * n
    finish = [0.0] * n
    events = [(arr[i], i) for i in range(n)]
    heapq.heapify(events)
    fifo, sjf_heap, seq = deque(), [], 0

    def add_ready(pid):
        nonlocal seq
        if algo == "SJF":
            seq += 1
            heapq.heappush(sjf_heap, (rem[pid], seq, pid))
        else:
            fifo.append(pid)

    def admit(upto):
        while events and events[0][0] <= upto:
            add_ready(heapq.heappop(events)[1])

    t, last, done, ctx, busy = 0.0, -1, 0, 0, 0.0
    while done < n:
        admit(t)
        if not (sjf_heap if algo == "SJF" else fifo):
            t = events[0][0]            # CPU idle until next arrival / I/O completion
            continue
        pid = heapq.heappop(sjf_heap)[2] if algo == "SJF" else fifo.popleft()
        if first[pid] < 0:
            first[pid] = t
        if last != pid:
            t += cs_cost
            ctx += 1
        run = min(quantum, rem[pid]) if algo == "RR" else rem[pid]
        t += run
        busy += run
        rem[pid] -= run
        last = pid
        admit(t)
        if rem[pid] <= 1e-9:
            idx[pid] += 1
            if idx[pid] == len(cpu[pid]):
                finish[pid] = t
                done += 1
            else:
                heapq.heappush(events, (t + io[pid][idx[pid] - 1], pid))
                rem[pid] = cpu[pid][idx[pid]]
        else:
            add_ready(pid)

    arr_a, fin_a = np.array(arr), np.array(finish)
    tot = np.array([sum(c) + sum(i) for c, i in zip(cpu, io)])
    turn = fin_a - arr_a
    wait = turn - tot                    # time spent in ready queue (incl. switch overhead)
    resp = np.array(first) - arr_a
    return {
        "avg_wait": wait.mean(), "p95_wait": np.percentile(wait, 95),
        "avg_turn": turn.mean(), "avg_resp": resp.mean(),
        "throughput": n / t, "cpu_util": busy / t, "ctx_switches": ctx, "makespan": t,
    }
