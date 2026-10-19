# Q-learning in CliffWalking

CPSC 4420/6420 Artificial Intelligence Assignment 2. This project implements
tabular Q-learning directly with NumPy and uses Gymnasium only for the environment.
It includes sections A–H, including the graduate stochasticity experiment.

Public repository: https://github.com/NimaZafarmomen/Gymnasium-

Watch the trained agent: [videos/trained_agent.mp4](videos/trained_agent.mp4).

## Main findings

- All seven deterministic configurations learned the optimal 13-step route,
  with return **−13** and **100% success** in final greedy testing.
- The selected configuration was alpha **0.5**, gamma **0.90**, epsilon decay
  **0.995**. Its final performance is tied with the baseline, not superior.
- Training with 10% action replacement improved mean noisy test return from
  **−67.006** to **−37.214** on the robustness test batch.
- At 25% replacement, the same learning settings were unstable: noise-trained
  agents achieved **62.6%** success and mean return **−410.350**. The original
  zero-noise agents achieved 100% success and mean return −164.246 in this sample.
  No claim of reliable high-noise convergence is made.

## Download the complete project

The [complete project archive](CliffWalking_Code_Results_Video.zip) contains all
511 project files, including every per-run CSV and trained Q-table. The code,
summary results, figures, and video are also available for browsing in this
repository. To reproduce or audit the supplied results, download the archive,
extract it, and open the `cliffwalking_assignment` directory. From a clone:

```bash
python -m zipfile -e CliffWalking_Code_Results_Video.zip .
cd cliffwalking_assignment
```

Then follow the setup and run instructions below.

## Setup

The recorded runs used Python 3.12.14. Use Python 3.12 and install FFmpeg
(`ffmpeg` and `ffprobe` must be on PATH) for video creation and verification.
On Ubuntu, FFmpeg is available through `sudo apt install ffmpeg`. Other systems
can use the official installers at https://ffmpeg.org/download.html.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.
Times New Roman is used for the supplied figures. Plot regeneration uses
DejaVu Serif if Times New Roman is not installed; the numerical results do not
depend on fonts. The script automatically uses common installed Times New Roman
fonts when available.

## Run

Run these commands from the project directory:

```bash
python -m unittest -v test_q_learning.py
python q_learning.py --workers 5
python make_outputs.py
python verify_results.py
```

`--workers 1` runs sequentially and produces the same experiment data. Training
uses separate process-local random generators. A full run overwrites files in
the specified results directory. To preserve the supplied results, use:

```bash
python q_learning.py --workers 5 --output results_reproduced
python make_outputs.py --results results_reproduced
python verify_results.py --results results_reproduced
```

Plot/video generation writes to the `plots/` and `videos/` folders beside the
chosen results directory. If FFmpeg is unavailable, regenerate plots with
`python make_outputs.py --skip-video` and retain the supplied MP4.

For a small execution check, use `--episodes 200 --seeds 1 --output results_smoke`.
This is not the full report protocol. The summary and verification scripts are
designed for the full 5,000-episode, five-seed protocol, not this smoke run.

## Environment and algorithm

- Gymnasium 1.2.1: `CliffWalking-v1`, `is_slippery=False`.
- Grid: 4 rows × 12 columns. Start 36; terminal goal 47.
- Observation: `12 * row + column`, `Discrete(48)`; action space `Discrete(4)`.
- Actions: 0 up, 1 right, 2 down, 3 left.
- Ordinary reward −1; cliff reward −100. A cliff transition returns to the start
  **without terminating** the episode.
- The added 500-step `TimeLimit` is external truncation.
- Initialize all 48 × 4 Q-values to zero; float64 arithmetic.
- Training uses epsilon-greedy action selection with uniform random choices
  among exact maximizing actions. Evaluation uses deterministic `argmax`.
- Target: `reward` on true termination; otherwise
  `reward + gamma * max(Q[next_state])`, including at time-limit truncation.
- Update: `Q[state, action] += alpha * (target - Q[state, action])`.

Hyperparameters are exposed in the `Config` dataclass near the top of
`q_learning.py`. The alpha, gamma, and decay comparisons are listed in
`deterministic_configs`. No external RL implementation is used.

## Experimental protocol

Every configuration uses 5,000 episodes and seeds 0–4. Epsilon starts at 1.0 and
decays once per episode according to `max(0.05, decay ** e)` for zero-based `e`.
All comparisons change one parameter relative to the baseline.

| Experiment | Values | Other fixed settings |
|---|---|---|
| Baseline | alpha 0.5, gamma 0.99, decay 0.995 | 5,000 episodes, cap 500 |
| Learning rate | 0.1, 0.5, 0.9 | gamma 0.99, decay 0.995 |
| Discount factor | 0.50, 0.90, 0.99 | alpha 0.5, decay 0.995 |
| Exploration decay | 0.950, 0.995, 0.999 | alpha 0.5, gamma 0.99 |
| Action replacement | p = 0, 0.10, 0.25 | selected deterministic settings |

The shared baseline is reused, giving **seven unique deterministic
configurations**, not nine. There are 35 deterministic full training runs and
10 additional noisy runs (225,000 main training episodes in total).

With probability p, `ActionNoise` executes one of the **three different valid
actions**, each with conditional probability 1/3. The Q-update learns the
**intended** action. The wrapper has a separate RNG from epsilon-greedy action
selection. Noise is also active during greedy evaluation. Built-in slipperiness
stays off. The p = 0 results are reused from the selected deterministic run.

## Evaluation, model selection, and statistics

Final validation and test each contain 100 frozen greedy episodes per training
seed. The report uses 500 episodes per configuration. Selection is ranked by:

1. Highest final validation mean return.
2. Highest mean validation return at checkpoints 50, 100, ..., 1000.
3. Highest overall mean training return.
4. Lexicographic configuration name if still tied.

The final test is never used for selection. No hyperparameters are tuned on
the noisy results. Fine-resolution diagnostic checkpoints are excluded from
configuration selection.

After selection, the first 200 episodes of every deterministic run are replayed
with frozen-policy checks every five episodes. Every replayed training prefix
is compared for exact equality with the original run. The sustained-optimal
checkpoint is the first observed −13 checkpoint with no later observed
nonoptimal checkpoint, combining these diagnostics with 50-episode checkpoints
after episode 200. This is an observed-checkpoint measure, not a guarantee for
unobserved episodes between checkpoints.

For training seed `s` and zero-based evaluation episode `j`:

| Phase | Environment seed |
|---|---|
| Checkpoint | `100000 + 1000*s + j` |
| Final validation | `200000 + 1000*s + j` |
| Final test (report G) | `300000 + 1000*s + j` |
| Robustness comparison (report H) | `400000 + 1000*s + j` |

The same robustness episode seeds are used for the original and noise-trained
policies at a given test noise level. Their trajectories can differ, so this is
a common-seed comparison, not identical realized state/action trajectories.
Deterministic checkpoint evaluation uses one rollout; noisy checkpoints use
30. Final evaluation has no exploration and never updates Q.

Reported ± values and figure bands generally show **sample SD across the five
seed-level means**, not SEM or a confidence interval. Within-run late-training
SD is explicitly labeled separately. SD bands may extend beyond physically
feasible return or success bounds because they are descriptive dispersion,
not probability intervals. All returns are undiscounted.

Repeated evaluation in the deterministic environment does not provide
independent environmental variation: the policies start at the same state and
use deterministic argmax. The five training seeds are the independent training
replicates. All episode-level data are retained.

## Files

| Path | Contents |
|---|---|
| `q_learning.py` | Agent, action-noise wrapper, full experiment runner |
| `test_q_learning.py` | Nine behavioral correctness tests |
| `make_outputs.py` | Summaries, eight report figures, native rendered video |
| `verify_results.py` | Independent numeric and video checks |
| `results/protocol.json` | Version information, seeds and full settings |
| `results/runs/<config>/seed_<n>/` | Per-run CSVs, config and Q-table |
| `results/training.csv` | All 225,000 main training episode records |
| `results/early_checkpoints.csv` | Fine-resolution early diagnostics |
| `results/selection_ranking.csv` | Auditable validation-based ranking |
| `results/summary.csv` | Deterministic and noise-trained summaries |
| `results/robustness.csv` | All held-out transfer/comparison episodes |
| `results/robustness_summary.csv` | Statistics used in report H |
| `plots/` | Eight figures as 300-dpi PNGs |
| `videos/trained_agent.mp4` | 10-second, H.264 native Gymnasium rendering |
| `videos/video_metadata.json` | Video policy, seed, route and score |
| `verification.json` | Results of the independent audit |

## Submission

The assignment requires a **PDF report** in Canvas and an accessible **public
GitHub repository** containing this code, README, results, plots, and MP4.
The editable Times New Roman Word report and its PDF are supplied separately.
The repository URL is https://github.com/NimaZafarmomen/Gymnasium-.
Submit the PDF report in Canvas. Do not submit individual code files to Canvas.

## References

1. Farama Foundation. [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/).
2. Farama Foundation. [Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
3. Watkins, C. J. C. H., and Dayan, P. (1992). [Q-learning](https://doi.org/10.1007/BF00992698).
   Machine Learning, 8, 279–292.
