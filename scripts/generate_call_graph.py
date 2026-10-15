"""
Generates call graph and dependency graph artifacts for the active simulation pipeline.
"""

from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files in the active simulation pipeline
ACTIVE_PIPELINE_FILES = [
    "main.py",
    "ui/optimization_widget.py",
    "core/optimisation_engine.py",
    "core/engine_factory.py",
    "core/engine_surrogate/surrogate_engine.py",
    "core/engine_surrogate/analytical_models.py",
    "core/engine_surrogate/profile_generator_fast.py",
    "evaluation/mmp.py",
    "core/objectives/wrapper.py",
    "core/objectives/economic.py",
    "core/objectives/storage.py",
    "core/data_models.py"
]

def main():
    pyan_exe = REPO_ROOT / ".venv" / "Scripts" / "pyan3.exe"
    if not pyan_exe.exists():
        pyan_exe = "pyan3"
        
    cmd = [
        str(pyan_exe),
        *[str(REPO_ROOT / f) for f in ACTIVE_PIPELINE_FILES],
        "--uses",
        "--defines",
        "--colored",
        "--grouped",
        "--dot"
    ]
    
    print("Running Pyan3 on active simulation pipeline...")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    
    dot_content = res.stdout
    
    # Save DOT to audit and agent_wiki
    call_graph_audit = REPO_ROOT / "audit" / "architecture" / "call_graph.txt"
    call_graph_wiki = REPO_ROOT / "agent_wiki" / "architecture" / "call_graph.txt"
    dep_graph_audit = REPO_ROOT / "audit" / "architecture" / "dependency_graph.txt"
    dep_graph_wiki = REPO_ROOT / "agent_wiki" / "architecture" / "dependency_graph.txt"
    
    with open(call_graph_audit, "w", encoding="utf-8") as f:
        f.write(dot_content)
    with open(call_graph_wiki, "w", encoding="utf-8") as f:
        f.write(dot_content)
        
    # Structured human-readable call sequence
    human_readable_trace = """# Active Simulation Pipeline Call Trace

```mermaid
sequenceDiagram
    autonumber
    participant UI as ui/optimization_widget.py
    participant Opt as core/optimisation_engine.py
    participant Factory as core/engine_factory.py
    participant Wrap as SurrogateEngineWrapper
    participant Surv as core/engine_surrogate/surrogate_engine.py
    participant MMP as evaluation/mmp.py
    participant Hybrid as PhDHybridRecoveryModel
    participant PGF as core/engine_surrogate/profile_generator_fast.py
    participant ODE as scipy.integrate.solve_ivp
    participant Obj as core/objectives/wrapper.py

    UI->>Opt: run_optimization(config, reservoir_data)
    loop Every Generation / Trial
        Opt->>Factory: create_engine("surrogate")
        Factory-->>Opt: SurrogateEngineWrapper instance
        Opt->>Wrap: evaluate_scenario(params, reservoir_data)
        Wrap->>Surv: evaluate_scenario(...)
        Surv->>MMP: calculate_mmp(pvt_props, method="auto")
        MMP-->>Surv: MMP value (psi)
        Surv->>Hybrid: calculate_recovery_factor(params, mmp, p_avg)
        Hybrid-->>Surv: Ultimate Recovery Factor (RF)
        Surv->>PGF: generate_profiles(eor_params, rf, operational_params)
        PGF-->>Surv: oil, water, gas rate profiles
        Surv->>ODE: solve_ivp(tank_pressure_ode, y0=p_init)
        ODE-->>Surv: pressure_profile (psi)
        Surv->>Surv: _calculate_co2_purchased_recycled(injection, gas_prod)
        Surv->>Surv: _calculate_engine_npv(profiles, econ_params)
        Surv-->>Wrap: SimulationResults(profiles, npv, rf, stored_co2)
        Wrap-->>Opt: SimulationResults
        Opt->>Obj: evaluate_fitness(results, constraints)
        Obj-->>Opt: Objective Score (Fitness)
    end
    Opt-->>UI: Best Candidate & Pareto Front
```
"""
    with open(dep_graph_audit, "w", encoding="utf-8") as f:
        f.write(human_readable_trace)
    with open(dep_graph_wiki, "w", encoding="utf-8") as f:
        f.write(human_readable_trace)
        
    print("Call graph and dependency traces written successfully!")

if __name__ == "__main__":
    main()
