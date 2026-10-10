# Execution Plans

Use an ExecPlan for work expected to span multiple modules, change research semantics, add a new benchmark, or require multiple experiment stages.

An ExecPlan is a living Markdown document under `plans/` with:

1. **Goal and non-goals**
2. **Scientific hypothesis** (when applicable)
3. **Current behavior/evidence**
4. **Design and compatibility decisions**
5. **Implementation steps**
6. **Tests and falsification checks**
7. **Experiment/rollout plan**
8. **Risks and rollback**
9. **Progress log** with dated updates
10. **Outcome** documenting what actually happened

A new contributor should be able to execute the plan using only the repository and the plan itself.
