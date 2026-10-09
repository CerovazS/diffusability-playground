# Posterior anisotropy at fixed rate and scale

Primary Flywheel node: `on the diffusability of latent spaces`.

## Objective and authorization

Task `posterior-anisotropy-20261009`, revision 1, status: validated; production launch prepared.
The user approved the proposed synthetic posterior experiment and explicitly
authorized code changes, execution on Gauss, only its RTX 3090, sequential jobs,
and no training longer than 20–30 minutes. Routine implementation and recovery
are delegated within this scope. This chat owns execution and decisions; no
new persistent chats, VAE experiments, cloud resources, or manuscript edits.

Definition of done: validated implementation, 18 sequential production runs
(2 center geometries x 3 covariance profiles x 3 training seeds), local metrics,
checkpoints, plots, paired comparison and a report including negative outcomes.
Training has a 25-minute graceful limit and each process a 29-minute watchdog.
Stop the queue on failure or incomplete training; do not compare truncated runs
to full-budget runs. A small separately labelled preflight precedes production.

## Claim, hypothesis, and decision

Question: does allocating posterior uncertainty differently change the learnability
of the aggregate mixture, with mean centers, mean KL to N(0,I), and total variance
fixed within each geometry? Paper anisotropy is Var_i(log posterior variance).
Hypothesis: lower posterior anisotropy reduces held-out marginal velocity error
at the same optimization budget. The directional hypothesis is not assumed true.

Primary evidence: final held-out per-coordinate squared error against the exact
marginal velocity of the linear conditional flow-matching path. Secondary:
validation error vs steps, SWD of generated vs independent reference samples,
real-vs-real SWD floor, and oracle-flow sampling with the same numerical solver.
Compare paired seed-wise log(error_high/error_low) separately for each geometry;
report all three paired effects, their mean and dispersion. A provisional
support signal requires >=10% lower mean primary error and consistent positive
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

## Current state

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
