# Expert Verification Engine

[Click Here for Video demo](https://www.loom.com/share/54c6a5ef38294e6cb857de83621fa470)

## 1) What This Project Solves

This project detects a specific fraud pattern in hiring workflows: candidates using AI tools to produce polished expert answers that do not reflect real hands-on experience.

Traditional identity checks verify who a person is. This system is focused on verifying what they likely know.

Core idea:
AI often follows instructions too perfectly and writes with a repeatable style. Human experts are usually more uneven, opinionated, and experience-anchored.

## 2) High-Level System Design

The engine combines three parts:

1. Behavioral Tripwires (Layer 1)
- T1: Word-count obedience trap
- T2: Submission velocity trap

2. Content Signals (Layer 2)
- S1 to S8 linguistic/statistical signals scored between 0 and 1

3. Decisioning
- Weighted Fraud Probability Score (FPS)
- Optional RL policy (PASS / FLAG / FOLLOW_UP)

The Streamlit app provides:
- Screening flow
- Recruiter-friendly results dashboard
- RL training monitor

## 3) Fraud Scoring Logic

### 3.1 Tripwires

T1 (Word Count Trap):
- Checks whether an answer is too close to instructed word limit (default tolerance: plus/minus 2 words).

T2 (Submission Speed Trap):
- Computes seconds per word.
- Fires if speed is faster than 1.5 seconds/word.

Tripwire aggregation:
- BOTH fired => multiplier 1.5
- ONE fired => multiplier 1.2
- NONE fired => multiplier 1.0

### 3.2 Signals S1-S8

Signals are produced by detection/signals.py:

- S1 Hedging Density
- S2 Absence of Failure Narratives
- S3 Temporal Vagueness
- S4 Low Tribal Vocabulary (domain insider language deficit)
- S5 Neutral Tool Opinion Polarity
- S6 Answer Length Uniformity
- S7 Structural Symmetry
- S8 Wide-but-Shallow Coverage

Each signal is normalized to [0, 1], where higher means more fraud-like behavior.

### 3.3 Final FPS Formula

Weighted base score:
```bash
(S1*8 + S2*9 + S3*8 + S4*9 + S5*7 + S6*7 + S7*6 + S8*6) / 60
```

Then:
```bash
FPS = min(base_score * tripwire_multiplier, 1.0)
```

Risk bands:
- CLEAN: < 0.35
- BORDERLINE: 0.35 to < 0.60
- HIGH_SUSPICION: 0.60 to < 0.80
- FRAUD: >= 0.80

## 4) RL Component

RL is implemented as a Gymnasium environment plus a sparse Q-learning agent.

Action space:
- 0 PASS
- 1 FLAG
- 2 FOLLOW_UP

Observation space (13 features):
- T1, T2
- S1..S8
- FPS
- question_index_norm
- follow_ups_used_norm

Reward design:
- FLAG FRAUD: +1.0
- PASS GENUINE: +0.5
- PASS FRAUD: -1.0
- FLAG GENUINE: -2.0
- FOLLOW_UP within limit: -0.1
- FOLLOW_UP over limit: -0.3

Reasoning:
False positives are penalized most heavily to avoid unfairly flagging genuine experts.

## 5) Data Generation

Synthetic candidate generation is in data/generator.py.

It creates FRAUD and GENUINE profiles with different probability distributions for tripwires and signal values.

Outputs:
- candidate_id
- label
- per-question tripwire states
- per-question signal vector
- per-question FPS

## 6) Dashboard UX (What You See)

The app has 3 tabs:

1. Screening
- Question-by-question response capture
- Live word count and elapsed time
- Stores answer text + duration

2. Results
- Verdict banner with plain language explanation
- Fraud Probability gauge
- Signal breakdown chart
- Top weighted drivers table
- Per-question diagnostics table (word count, sec/word, T1/T2 flags)
- "How to read this" guidance section for recruiters

3. RL Training
- Train episodes and dataset size controls
- Reward trend plot
- Fraud caught vs false positives chart

## 7) Project Structure

- app.py
- detection/
  - tripwires.py
  - signals.py
  - scorer.py
- data/
  - lexicons.py
  - generator.py
  - sample_candidates.json
- rl/
  - environment.py
  - agent.py
  - trainer.py
- scripts/
  - generate_sample_data.py
  - train_agent.py
- tests/
  - test_tripwires.py
  - test_signals.py
  - test_scorer.py
- requirements.txt

## 8) Installation and Setup (Windows PowerShell)

Step 1: Create virtual environment
```bash
python -m venv .venv
```

Step 2: Activate environment
```bash
.\.venv\Scripts\Activate.ps1
```

Step 3: Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Step 4: Download language assets
```bash
python -c "import nltk; nltk.download('punkt')"
python -m spacy download en_core_web_sm
```

Note:
If spaCy model download is blocked by network policy, the code falls back to a blank English pipeline automatically.

## 9) Run the System

Generate sample data:
```bash
python scripts/generate_sample_data.py
```

Train RL agent:
```bash
python scripts/train_agent.py
```

Start UI:
```bash
streamlit run app.py
```

## 10) Run Tests

pytest -q

## 11) End-to-End Runtime Flow

1. Candidate submits answers in Screening tab.
2. System records answer times and computes T1/T2 per question.
3. System computes S1..S8 across answer set.
4. FPS is calculated using weighted base and tripwire multiplier.
5. Dashboard presents verdict and explainability outputs.
6. Optional RL module can be trained and used for decision policy improvement.

## 12) Production Notes

- Explainability first: every verdict includes component-level contribution visibility.
- Conservative policy: false-positive penalty is intentionally highest.
- Modular architecture: tripwires, signals, scorer, RL, and UI are isolated for easier audits.
- Deterministic interfaces: each module has clean function-level inputs/outputs suitable for service extraction.

## 13) Limits and Expected Behavior

- This is a PoC with synthetic training data by default.
- Signals are heuristic and should be calibrated with real review feedback.
- RL policy quality depends on representative labels and retraining cadence.

## 14) Next Recommended Steps

1. Add persistent storage for recruiter review outcomes.
2. Add model/version tracking for scorer and RL artifacts.
3. Add CI checks (tests + style + type checks).
4. Add role-based access and audit logs for enterprise deployment.
