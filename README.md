# CS2 – Machine-Learning-Assisted CPU Scheduling

## Run
    pip install -r requirements.txt
    python run_all.py          # ~2-3 min: builds dataset, trains models, writes results + figures

## Layout
- config.py              all experiment settings (sizes, seed, context-switch cost, policies, cost weights)
- workload_generator.py  5 workload types (cpu_bound, io_bound, interactive, heavy_tail, mixed) + feature extraction
- simulator.py           event-driven single-CPU simulator (FCFS, SJF, Round Robin q) with CPU/I-O bursts
- build_dataset.py       generates scenarios, simulates every policy, labels the best one -> data/raw/*.csv
- train_evaluate.py      Decision Tree, Random Forest, Logistic Regression baseline; CV; OS-level regret; stats; plots
- results/               CSV summaries + figures (fig1..fig6)

## Method in one paragraph
Each scenario is a batch of N processes (N = 50, 200, 500) with alternating CPU/I-O bursts. Every candidate
policy is simulated on the identical workload with identical settings. The label is the policy with the lowest
cost = avg waiting + avg response + p95 waiting. Models predict the label from workload features only; predicted
policies are then looked up in the simulated results to compute real scheduling performance (regret vs oracle).
