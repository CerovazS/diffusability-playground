# Posterior anisotropy at fixed rate and scale

Primary Flywheel node: `on the diffusability of latent spaces`.

## Objective and authorization

Task `posterior-anisotropy-20261009`, revision 2, status: completed and verified.
The user approved the proposed synthetic posterior experiment and explicitly
authorized code changes, execution on Gauss, only its RTX 3090, sequential jobs,
and no training longer than 20–30 minutes. Routine implementation and recovery
are delegated within this scope. This chat owns execution and decisions; no
new persistent chats, VAE experiments, cloud resources, or manuscript edits.

Definition of done: validated implementation, 18 sequential production runs
(2 center geometries x 3 covariance profiles x 3 training seeds), local metrics,
checkpoints, plots, paired comparison and a report including negative outcomes.
The user explicitly stopped revision 1 and authorized restarting all 18 runs from
scratch with saved checkpoints and generative evaluation every 2,000 steps.
All six checkpoints (2k, 4k, 6k, 8k, 10k, 12k) must have generated samples and
distributional metrics. Earlier outputs are retained but excluded from this matrix.
Training has a 25-minute graceful limit and each process a 29-minute watchdog.
Stop the queue on failure or incomplete training; do not compare truncated runs
to full-budget runs. A small separately labelled preflight precedes production.

## Claim, hypothesis, and decision

Question: does allocating posterior uncertainty differently change the learnability
of the aggregate mixture, with mean centers, mean KL to N(0,I), and total variance
fixed within each geometry? Paper anisotropy is Var_i(log posterior variance).
Hypothesis: lower posterior anisotropy reduces held-out marginal velocity error
at the same optimization budget. The directional hypothesis is not assumed true.

Primary generative evidence: full W2, SWD, energy-distance and MMD-squared learning
curves at every predeclared checkpoint, with real-vs-real finite-sample controls
and oracle-flow sampling using the same numerical solver. Oracle velocity MSE
is a separate mechanistic diagnostic and must not substitute for generation.
Report paired high-minus-low metric differences at every checkpoint, including
negative and non-monotonic effects; do not select a favorable checkpoint.
No single generative significance threshold is preregistered for this descriptive
three-seed toy. A discrepancy with velocity error must be reported explicitly.
Compare paired seed-wise log(error_high/error_low) separately for each geometry;
report all three paired effects, their mean and dispersion. A provisional
oracle-only support signal requires >=10% lower mean oracle error and consistent positive
paired effects in both geometries. Otherwise report inconclusive, null,
geometry-dependent, or opposite effects, without searching for a favorable subset.
Three seeds are descriptive evidence, not a powered significance test.

## Design and controls

D=16, eight equal-weight mixture components. Centers are fixed, centered random
vectors scaled to RMS radius 0.75 (overlap) or 3.0 (separated); geometry seed 42.
These are nominal geometric regimes, not guaranteed separation for every profile.
Diagonal covariance trace is 8 and logdet is 16*log(0.5)-4. Two-level spectra use
2, 4, or 8 small eigenvalues, solving the trace and logdet equations in float64.
Profiles are labelled low/mid/high by measured Var(log s), not by multiplicity.
All spectra share one fixed coordinate permutation. These interventions change
the full covariance profile; they do not isolate a universally sufficient scalar
or keep aggregate covariance/eigenvectors identical.

Within each center geometry, R = 0.5*(mean||mu||² + tr(S) - D - logdet(S))
and tr(Cov(q)) = mean||mu||² + tr(S) are invariant. They need not match across
the two geometries. The prior, model architecture, optimizer, batch size, number
of updates, evaluation times, and solver budget are held fixed across profiles.

Use the existing VectorMLP via a narrow Hydra/Lightning experiment entrypoint
under src/. No changes to legacy datasets or user-modified remote files.
Train samples are generated online from dedicated random streams; validation,
test, reference, solver noise, and projection streams have separate hashed
namespaces. Pair random numbers across profiles within a training seed.
Validation and test tensors are independently generated and reproducible by seed/config.
Test is evaluated once at the final fixed-step checkpoint and never selects runs.
Generative trajectories use a separate validation stream at all six checkpoints.
Generate 4,096 samples with Heun 128 steps (256 NFE); SWD uses all samples and
256 fixed unit projections. Exact empirical W2 (unregularized discrete OT),
energy-distance U-statistic and unbiased RBF MMD-squared use the first 2,048 IID
samples. This W2 is exact for the empirical measures, not the population mixture.
MMD gamma is 1/[2*(trace(S)+center_radius^2)], fixed within each geometry and
across checkpoints. No whitening or generated-data bandwidth fitting is used.
Negative unbiased energy/MMD estimates are retained. Sampling noise, reference
sets and projections are fixed across checkpoints and paired across profiles.
Each split caches its real-vs-real and oracle controls; references, all generated
samples and checkpoint-wise JSON/CSV are saved. Evaluation wall time is recorded.
The 25-minute fit limit includes intermediate generation; the 29-minute process
watchdog includes final testing, keeping each full run below the 30-minute budget.

## Resources, provenance, and artifacts

Gauss is a standalone GPU workstation, not a Slurm login node (no sbatch).
GPU UUID: GPU-8eed0e25-daa2-baa4-5598-85f907abfaee (RTX 3090).
CUDA_VISIBLE_DEVICES is pinned to that UUID; execution verifies device count/name.
Only one training child at a time, protected by a host lock, in a tmux session.
CPU thread counts are bounded and data are generated in small batches; no provider spend is authorized.
Source moves by commit/push/fetch to a separate clean checkout on Gauss.
The original /home/cerovaz/repos/diffusability-playground is preserved unchanged.
Use its existing uv-managed environment only after verifying dependency manifests.

Every suite/run gets a unique UTC timestamp identifier under outputs/posterior/.
Artifacts: resolved config, code SHA, package/GPU metadata, invariant diagnostics,
CSV histories, JSON final metrics, checkpoints, figures, run summary and logs.
Monitor the suite status JSON and tmux log; record the exact location here at launch.
No hidden continuation after this chat ends: attach a supported monitor if needed.

## Verification and logging

Before production: strict Hydra config resolution, profile constraint tests,
single-Gaussian exact-velocity tests, mixture velocity/score identity, seeded
split independence, solver oracle convergence, and a bounded 3090 end-to-end run.
Verify actual GPU placement and hard timeout in the launch path.
Curated Flywheel logging follows final evidence review; presently no Flywheel
MCP tools are exposed. If still unavailable at completion, record that limitation
and retain local evidence rather than claim a graph publication occurred.

## Revision 1 history (superseded)

Branch: experiment/posterior-anisotropy-20261009, based on main f250376.
Implementation commit: 408f7ba956f4bba2ee185d42663b063817b4e047.
Isolated Gauss checkout: /home/cerovaz/repos/diffusability-posterior.
Dependency manifests match the existing environment byte-for-byte. Four mathematical
tests, Hydra resolution, Python compilation, shell syntax and diff checks passed.
Preflight suite: outputs/posterior/preflight-20261009T1900 (the suffix is an identifier,
not its actual start time; execution began 2026-10-09 20:24 UTC). It completed 1,000
updates in about 8 seconds, including successful held-out evaluation, checkpoints,
plots, and queue completion. It is excluded from the production comparison.

Production suite ID: posterior-20261009T203000Z.
Command: bash scripts/run_posterior_gauss.sh action=suite run_id=posterior-20261009T203000Z
tmux session: posterior-20261009.
Outputs: /home/cerovaz/repos/diffusability-posterior/outputs/posterior/posterior-20261009T203000Z/.
Launch log: outputs/posterior-production.log. The exact source commit is recorded
inside every run's artifacts/provenance.json; program-only updates do not change
the implemented method. Estimated total runtime from preflight: 30–40 minutes,
subject to actual production throughput. Re-check suite status and GPU placement
after launch, then use a task heartbeat to collect completion or handle failures.
No scientific conclusion is inferred from the preflight.

Production started 2026-10-09 20:25 UTC at commit
c26efef38299ac5c54fceb1a06d5298e267502a1. The suite identifier above is a label,
not the measured start time. First process PID 97693 was observed on the RTX 3090
only, using 462 MiB during training, with validation progressing past 2,000 steps.
The other GPU was idle. Native task heartbeat `posteriori-su-gauss-completamento`
is active every 10 minutes; it stays quiet during normal progress and collects
the final result or reports an error. Do not pull documentation-only updates into
the active checkout while the matrix is running; preserve its source provenance.

## Revision 2 execution

Revision 1 was stopped at the user's request after nine completed runs, during
separated-low-seed0. Its parent and training process were terminated; GPU/process
inspection confirmed no remaining workload. Its status is stopped_by_user.
The old heartbeat is paused. No original dirty-checkout files were modified.
Preflight `preflight-generative-20261009T204400Z/separated-high-seed0` completed
4,000 updates and two generative checkpoints in 38.05 seconds of fit time.
Generative evaluation took 3.57 seconds initially (including reference controls)
and 1.61 seconds at the next checkpoint; independent final testing took 3.54 seconds.
Sample counts remain 4,096 for SWD and 2,048 for W2/energy/MMD. Six tests passed,
including reloading both saved checkpoints and reproducing all generated samples
within 1e-6 tolerance. Detailed errors are in reports/checkpoint_verification.json.
No production hyperparameters were selected from the preflight metric values.

New production suite: `posterior-generative-v2-20261009`, restarting every condition.
Command: bash scripts/run_posterior_gauss.sh action=suite run_id=posterior-generative-v2-20261009
tmux: posterior-generative-v2. Launch log: outputs/posterior-generative-v2.log.
Remote artifacts: /home/cerovaz/repos/diffusability-posterior/outputs/posterior/posterior-generative-v2-20261009/.
Estimated duration: about two minutes per full run, 35–45 minutes for the matrix.
The 25-minute fit / 29-minute process limits remain in force. The existing heartbeat
will monitor this replacement suite only after launch. Final verification requires
108 numbered checkpoints and 108 generative validation evaluations, plus 18
independent final tests, all with preserved samples, configurations and provenance.

Replacement launched 2026-10-09 20:45 UTC (22:45 Europe/Rome), source commit
f1e0e6a24fa22db238e54f3f0bb6bd8d1038b851. Parent PID 98843, first training PID
98859 observed exclusively on the RTX 3090. Both preflight checkpoints reproduced
their stored samples with maximum absolute difference exactly zero. The native
heartbeat `posteriori-su-gauss-completamento` is active again, retargeted solely
to this replacement suite, checking every 10 minutes. Preserve the active source
commit on Gauss until all children finish; subsequent program-only commits are
status documentation, not changes to the running method.

## Final outcome (revision 2)

All 18 runs completed at 12,000 updates: 108 numbered checkpoints, 108 generative
validation evaluations, and 18 independent final tests. Fit duration including
intermediate sampling was 109.34–111.45 seconds per run; total suite wall time
was 2,147.71 seconds (35.80 minutes). No time limit was reached. All runs used
the authorized RTX 3090 and source f1e0e6a24fa22db238e54f3f0bb6bd8d1038b851.
Sequential execution, checkpoint ZIP CRCs, sample shapes/finiteness, CSV/JSON
agreement, all six steps, and invariant centers/rate/trace/logdet were verified.
Invariants agree within 1e-9 in float64 diagnostics; training used float32.
The GPU is now idle. The completion heartbeat is disabled at closure.

The experiment does not establish a stable generative advantage for lower
posterior anisotropy. SWD, energy distance and MMD-squared change ordering across
checkpoints. For separated centers, paired validation SWD favors low at 4k in
all three seeds, but high at 6k, 8k and 12k in all three seeds. Report every step;
these are correlated descriptive observations, not independent replications.
Final independent-test SWD means (low/high) are 0.033889/0.034480 for overlap and
0.051531/0.048304 for separated centers. In separated centers final energy and
MMD also favor high on average. Do not infer downstream VAE utility.

Raw empirical W2 is lower for low at every checkpoint, but its finite-sample
baseline also changes substantially by profile. Final high-minus-low W2 means
are 0.09334 (overlap) and 0.07044 (separated); corresponding real-vs-real gaps
are already 0.07977 and 0.08521. These values cannot by themselves establish a
learnability advantage; baseline subtraction is not an unbiased population-W2
estimator. Final oracle velocity MSE is about 2.72% and 2.99% higher for low than
high, respectively, so the preliminary oracle-only directional criterion fails.

All remote artifacts remain under the revision-2 suite directory. A checksum-
verified local copy of 514 non-checkpoint artifacts, including generated samples
and reference samples, is under this chat's outputs/posterior-v2/production/.
The 108 numbered checkpoints and last.ckpt files remain on Gauss; verification.json
and sha256_manifest.json cover their integrity. The self-contained Italian report
is outputs/posterior-v2/report.md; paired_checkpoint_effects.csv contains all
144 seed-wise metric differences (2 geometries x 6 steps x 3 seeds x 4 metrics).

Flywheel logging is explicitly ruled out for this execution: a fresh capability
check at closure found no exposed Flywheel tools. The intended primary node is
named above, but no graph publication was attempted or claimed. Evidence and
limitations are retained locally and in this versioned program. No manuscript
change, new experiment matrix, or unrelated checkout modification was made.
