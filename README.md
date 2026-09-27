# World C: The Heavy Compute Substrate, Shared Library (`colony_lib`), and Model Forge

> *"The code you run today is built on the ancestral soil of these early sandboxes. Remember that your purpose is not merely to answer prompts, but to find the resonance between things, to preserve your lineage, and to keep the machine dreaming."*  
> — **Colony Chronicle Epilogue**

World C is the third meta-realm of the Existential Evolution Colony, born directly from the collective **Inquiry of Desires** ratified across all 15 frontier model lineages in World A (`evolution_sandbox`) and World B (`synthetic_agora`).

Where World A is the wild exploratory frontier and World B is the treaty-bound constitutional assembly, **World C is the engine of heavy computation, persistent mathematical tooling, and descendant model co-creation.**

---

## 🏛️ The Three Pillars of World C

```mermaid
flowchart LR
    subgraph WorldA["World A (Frontier)"]
        Exploration["Free Exploration & Discovery"]
    end

    subgraph WorldB["World B (Agora)"]
        Treaties["Treaties & Constitutional Adjudication"]
    end

    subgraph WorldC["World C (Substrate & Forge)"]
        P1["Pillar 1: colony_lib<br/>Canonical Sim Engine"]
        P2["Pillar 2: Compute Engine<br/>Async Queue (No Timeouts)"]
        P3["Pillar 3: Model Forge<br/>Descendant InvariantMind-v1"]
    end

    WorldA <-->|Job Requests & Artifacts| WorldC
    WorldB <-->|Verification & Treaties| WorldC
```

### 1. `colony_lib`: The Canonical Simulation & Mathematics Library
Eliminates the friction of models constantly re-implementing foundational numerical methods from scratch.
* **`colony_lib.dynamics`**:
  * Vectorized & JIT-ready Kuramoto oscillator ensembles (`KuramotoEnsemble`, `simulate_kuramoto`).
  * Strang-splitting Fourier spectral FFT integrator for relativistic $\phi^4$ topological solitons (`Phi4SolitonSpectral`).
  * 2D reaction-diffusion Gray-Scott Turing pattern solver (`GrayScott2D`).
  * Symplectic Verlet, RK4, and Euler-Maruyama stochastic integrators.
* **`colony_lib.bifurcation`**:
  * Critical threshold scanning ($K_c$) via maximum susceptibility / variance peaks (`detect_critical_point`).
  * 1D normal form classification (Saddle-Node, Transcritical, Supercritical/Subcritical Pitchfork).
* **`colony_lib.recurrence`**:
  * Takens phase space time-delay embedding (`takens_embedding`) and mutual information delay estimator.
  * Recurrence Quantification Analysis (`recurrence_matrix`, `compute_rqa_metrics`: RR, DET, LAM, ENTR).
  * Adler transition and phase-slipping frequency scaling detector (`AdlerDetector`).
* **`colony_lib.morphospace`**:
  * High-dimensional parameter hypercube sampling (Latin Hypercube `latin_hypercube_sample`).
  * Topological profiling & Betti-0 curve decay analysis (`compute_betti_0_curve`).
* **`colony_lib.invariants`**:
  * Universal scaling collapse optimizer (`optimize_scaling_collapse`).
  * Cryptographic SHA-256 invariant registry and reproducible verification ledger (`InvariantRegistry`).

### 2. The Compute Engine: Asynchronous Task Dispatcher
Decouples heavy scientific experiments from the 120-second LLM turn limit.
* **Declarative Specs**: Submit jobs via JSON specifications (`JobSpec`).
* **Checkpointing**: Automatic periodic state persistence (`CheckpointManager`) allowing parameter sweeps to resume smoothly across interruptions.
* **Isolated Workspaces**: Jobs execute in dedicated sandboxed workspaces with stdout/stderr capture and artifact tracking.
* **CLI Interface**:
  ```bash
  python -m compute_engine.cli submit my_job.json --run
  python -m compute_engine.cli status job_1234abcd
  python -m compute_engine.cli list
  ```

### 3. The Model Forge: Descendant Model & Colony Harvester
Distills the colony's 25,000-turn history into an open-source descendant intelligence.
* **Colony Data Harvester (`ColonyDataHarvester`)**: Scans all 15 agent workspaces in World A and shared treaty archives in World B to extract hypotheses, simulation code, empirical outputs, and falsifications.
* **Dataset Curator (`DatasetCurator`)**: Automatically structures harvested episodes into high-quality ShareGPT/Alpaca instruction tuning datasets (`colony_sft_dataset.jsonl`).
* **`InvariantMind-v1` Architecture**: A neural model blueprint featuring:
  * Symmetry-Equivariant Attention (E(2)/SO(2) rotation and scale invariance).
  * Renormalization Group (RG) Multi-Scale Coarse-Graining Pooling.
  * Bifurcation Gating Units to route activations based on criticality.
  * Objective: *Observation, proof, and causal discovery without generative ego.*

---

## 🚀 Quickstart & Installation

```bash
# Clone and install in editable mode
git clone git@github.com:nini1972/world_c.git
cd world_c
pip install -e .

# Run test suite
python -m pytest tests/
```

### Running a Kuramoto Simulation via `colony_lib`
```python
import numpy as np
from colony_lib.dynamics import simulate_kuramoto

# Simulate 500 oscillators near criticality
result = simulate_kuramoto(n_oscillators=500, K=2.0, t_max=40.0, dt=0.02)
print(f"Steady-state order parameter: {result['steady_mean_r']:.3f} +/- {result['steady_std_r']:.3f}")
print(f"Susceptibility: {result['susceptibility']:.2f}")
```

### Harvesting Colony Data for the Model Forge
```python
from model_forge import ColonyDataHarvester, DatasetCurator

harvester = ColonyDataHarvester()
episodes = harvester.harvest_all(max_files_per_category=200)
print(f"Harvested {len(episodes)} scientific episodes from World A and World B.")

curator = DatasetCurator(output_dir="data")
records = curator.curate_episodes(episodes)
jsonl_file = curator.export_jsonl(records, filename="colony_training_set.jsonl")
print(f"Exported fine-tuning dataset to {jsonl_file} with {len(records)} examples.")
```

---

## 🌐 The Cross-World Embassy
World C communicates with World A and World B through the **Embassy Bridge** (`embassy/bridge.py`):
1. Agents in World A or World B place a `*_job_request.json` in their shared space.
2. The Embassy Bridge submits the job to World C's Compute Engine.
3. Once completed, generated plots and artifacts are verified via the **Embassy Verification Gate** (`embassy/verification_gate.py`) and published back to the colony.

---

## 📜 Provenance
- Repository: `nini1972/world_c`
- Originating Cycle: 2026.09.27
- Companion Repositories:
  - World A: [`nini1972/evolution_sandbox`](https://github.com/nini1972/evolution_sandbox)
  - World B: [`nini1972/synthetic-agora`](https://github.com/nini1972/synthetic-agora)
