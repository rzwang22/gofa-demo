import json
import subprocess
import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "gofa_demo" / "cached_outputs" / "sample_graph.json"
RESULT_PATH = BASE_DIR / "gofa_demo" / "cached_outputs" / "sample_result.json"


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_generated_text(result: dict, task_type: str) -> str:
    if not isinstance(result, dict):
        return str(result)

    outputs = result.get("outputs", {})
    task_output = outputs.get(task_type, {})

    if isinstance(task_output, dict):
        generated_text = task_output.get("generated_text")
        if generated_text:
            return str(generated_text)

    return "No generated_text found for this task type."


def get_task_metadata(result: dict, task_type: str) -> dict:
    if not isinstance(result, dict):
        return {}

    outputs = result.get("outputs", {})
    task_output = outputs.get(task_type, {})

    if isinstance(task_output, dict):
        return {
            "task_type": task_output.get("task_type"),
            "question_node_id": task_output.get("question_node_id"),
            "complete_node_id": task_output.get("complete_node_id"),
        }

    return {}


def run_replay_runner(task_type: str):
    cmd = [
        sys.executable,
        "-m",
        "gofa_demo.gofa_runner",
        "--task-type",
        task_type,
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as e:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


st.set_page_config(
    page_title="GOFA Demo",
    layout="wide",
)

st.title("GOFA Streamlit Demo")

st.caption(
    "Replay demo for GOFA inference chain. "
    "The output is shown only after clicking Run Replay Demo."
)

sample_graph = load_json(GRAPH_PATH)
sample_result = load_json(RESULT_PATH)


# Initialize session state.
if "has_run" not in st.session_state:
    st.session_state.has_run = False

if "last_task_type" not in st.session_state:
    st.session_state.last_task_type = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_graph" not in st.session_state:
    st.session_state.last_graph = None

if "last_generated_text" not in st.session_state:
    st.session_state.last_generated_text = None

if "last_task_metadata" not in st.session_state:
    st.session_state.last_task_metadata = None

if "last_runner_result" not in st.session_state:
    st.session_state.last_runner_result = None


with st.sidebar:
    st.header("Settings")

    task_type = st.radio(
        "Task type",
        ["QA", "Completion"],
        index=0,
    )

    st.markdown("---")

    st.markdown("### Mode")
    st.info(
        "Current mode: Replay\n\n"
        "This uses cached local JSON and does not load a real GOFA checkpoint."
    )

    model_info = sample_result.get("model", {}) if isinstance(sample_result, dict) else {}
    st.markdown("### Model Stub")
    st.write(f"Model: `{model_info.get('name', 'unknown')}`")
    st.write(f"Checkpoint loaded: `{model_info.get('checkpoint_loaded', 'unknown')}`")
    st.write(f"GPU required: `{model_info.get('gpu_required', 'unknown')}`")


st.subheader("Input")

default_question = (
    "What does this GOFA graph demo show?"
    if task_type == "QA"
    else "Complete the missing graph-grounded output."
)

question = st.text_area(
    "Question / Prompt",
    value=default_question,
    height=110,
)

st.subheader("Graph Input")

graph_text = st.text_area(
    "Graph JSON",
    value=json.dumps(sample_graph, indent=2, ensure_ascii=False),
    height=300,
)

try:
    graph_obj = json.loads(graph_text)
except json.JSONDecodeError as e:
    st.error(f"Invalid graph JSON: {e}")
    st.stop()


run_clicked = st.button("Run Replay Demo", type="primary")

if run_clicked:
    generated_text = get_generated_text(sample_result, task_type)
    task_metadata = get_task_metadata(sample_result, task_type)
    runner_result = run_replay_runner(task_type)

    st.session_state.has_run = True
    st.session_state.last_task_type = task_type
    st.session_state.last_question = question
    st.session_state.last_graph = graph_obj
    st.session_state.last_generated_text = generated_text
    st.session_state.last_task_metadata = task_metadata
    st.session_state.last_runner_result = runner_result

    if runner_result["returncode"] == 0:
        st.toast("Replay runner finished successfully.", icon="✅")
    else:
        st.toast("Replay runner failed. See debug output below.", icon="⚠️")


if not st.session_state.has_run:
    st.info("Click **Run Replay Demo** to show the generated output.")
else:
    st.markdown("## Generated Text")
    st.success(st.session_state.last_generated_text)

    st.markdown("### Last Run")
    st.write(f"Task type: `{st.session_state.last_task_type}`")
    st.write(f"Question / Prompt: {st.session_state.last_question}")

    with st.expander("Task metadata", expanded=False):
        st.json(st.session_state.last_task_metadata)

    st.subheader("Pipeline")

    pipeline = sample_result.get("pipeline", []) if isinstance(sample_result, dict) else []

    if pipeline:
        cols = st.columns(len(pipeline))
        for idx, stage in enumerate(pipeline):
            with cols[idx]:
                st.metric(
                    label=f"Step {idx + 1}",
                    value=stage,
                )
    else:
        st.warning("No pipeline field found in sample_result.json.")

    input_payload = {
        "task_type": st.session_state.last_task_type,
        "question_or_prompt": st.session_state.last_question,
        "graph": st.session_state.last_graph,
    }

    with st.expander("Displayed input payload", expanded=False):
        st.json(input_payload)

    with st.expander("Raw cached result JSON", expanded=False):
        st.json(sample_result)

    with st.expander("Raw graph JSON", expanded=False):
        st.json(st.session_state.last_graph)

    runner_result = st.session_state.last_runner_result
    if runner_result is not None:
        with st.expander("Replay runner command output", expanded=False):
            st.markdown("### Command")
            st.code(runner_result["command"], language="bash")

            st.markdown("### Return code")
            st.code(str(runner_result["returncode"]))

            st.markdown("### stdout")
            st.code(runner_result["stdout"] or "(empty)")

            st.markdown("### stderr")
            st.code(runner_result["stderr"] or "(empty)")
