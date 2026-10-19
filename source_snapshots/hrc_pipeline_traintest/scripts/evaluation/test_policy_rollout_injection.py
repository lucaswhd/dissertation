import ast
from pathlib import Path


DEMO = Path(__file__).resolve().parents[1] / "collection/hrc_multitask_demo.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_policy_rollout_is_default_off_and_single_step_is_causal_env_step():
    source = DEMO.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_policy_client"
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant) and assignments[0].value.value is None

    policy_step = _function(tree, "_policy_step")
    rendered = ast.unparse(policy_step)
    assert "env.step(action" in rendered
    assert "validate_r1pro_joint_targets" in rendered
    assert rendered.index("validate_r1pro_joint_targets") < rendered.index("env.step(action")
    assert "load_state" not in rendered
    assert "recorder.record" in rendered


def test_all_active_phase_entry_points_have_policy_branch():
    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    functions = (
        "run_go_to_task_phase_smooth",
        "run_return_phase_smooth",
        "run_face_human_phase_smooth",
        "run_face_yaw_phase",
    )
    for name in functions:
        rendered = ast.unparse(_function(tree, name))
        assert "if _policy_client is not None:" in rendered, name
        assert "run_policy_phase" in rendered, name
        if name in {"run_go_to_task_phase_smooth", "run_return_phase_smooth"}:
            assert "_record_policy_arrival" in rendered, name


def test_relocate_uses_the_exact_follow_instruction_from_training():
    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "FOLLOW_POLICY_INSTRUCTION" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value == "Follow the person and maintain a safe distance."
    rendered = ast.unparse(_function(tree, "run_relocation_phase"))
    assert "task_instruction=FOLLOW_POLICY_INSTRUCTION" in rendered


def test_expert_operation_replay_calls_are_only_in_policy_else_branches():
    tree = ast.parse(DEMO.read_text(encoding="utf-8"))
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    replay_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in {"playback_episode_with_callback", "playback_external_segment"}:
            replay_calls.append(node)
    assert len(replay_calls) == 2

    for call in replay_calls:
        child = call
        parent = parents[child]
        guarded = False
        while parent is not None:
            if isinstance(parent, ast.If) and child in parent.orelse:
                if "_policy_client is not None" in ast.unparse(parent.test):
                    guarded = True
                    break
            child, parent = parent, parents.get(parent)
        assert guarded, ast.unparse(call)
