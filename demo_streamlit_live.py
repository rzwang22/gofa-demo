import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


GOFA_ROOT = Path("/mnt/sevenT/wrz_data/GOFA")
CHAT_GOFA = GOFA_ROOT / "chat_gofa.py"


def extract_generated_text(stdout: str) -> str:
    """
    Original chat_gofa.py prints the final output like:
    [' Sure! Let me explain ... </s>']

    This function scans stdout from bottom to top and extracts the last Python list/string output.
    """
    lines = stdout.splitlines()

    for line in reversed(lines):
        s = line.strip()

        if not s:
            continue

        # Skip debug/info lines.
        if s.startswith("[DEBUG") or s.startswith("[INFO"):
            continue

        # Try to parse the final list output.
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, list) and len(obj) > 0:
                    text = str(obj[0])
                    text = text.replace("</s>", "").strip()
                    return text
            except Exception:
                pass

        # Fallback: direct string-like output.
        if len(s) > 20 and not s.startswith("=") and "DEBUG" not in s:
            if "Sure!" in s or "Let me" in s:
                return s.replace("</s>", "").strip()

    return "Could not extract final generated text from chat_gofa.py output."


def extract_debug_summary(stdout: str) -> dict:
    summary = {}

    m = re.search(r"len\(g\.x\):\s*(\d+)", stdout)
    if m:
        summary["num_nodes"] = int(m.group(1))

    m = re.search(r"num edges:\s*(\d+)", stdout)
    if m:
        summary["num_edges"] = int(m.group(1))

    m = re.search(r"question\[0\]:\s*(.*)", stdout)
    if m:
        summary["question"] = m.group(1).strip()

    m = re.search(r"answer\[0\]:\s*(.*)", stdout)
    if m:
        summary["answer_field_in_graph"] = m.group(1).strip()

    return summary


def run_original_chat_gofa(cuda_visible_devices: str, timeout_sec: int):
    """
    Keep the original chat_gofa.py loading/generation logic.

    Important:
    - cwd is GOFA_ROOT so relative files like sample_graph_pyg.pth resolve exactly as the original script expects.
    - sys.executable uses the current conda env's Python.
    """
    env = os.environ.copy()

    if cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()

    cmd = [
        sys.executable,
        str(CHAT_GOFA),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(GOFA_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    return {
        "command": " ".join(cmd),
        "cwd": str(GOFA_ROOT),
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "generated_text": extract_generated_text(stdout),
        "debug_summary": extract_debug_summary(stdout),
    }


st.set_page_config(
    page_title="GOFA Live Demo",
    layout="wide",
)

st.title("GOFA Live Streamlit Demo")

st.caption(
    "This page calls the original GOFA `chat_gofa.py` script and displays the generated output. "
    "The sample graph and model paths are still controlled by the original script."
)

with st.sidebar:
    st.header("Live GOFA Settings")

    st.write("Original script:")
    st.code(str(CHAT_GOFA), language="bash")

    st.write("Working directory:")
    st.code(str(GOFA_ROOT), language="bash")

    cuda_visible_devices = st.text_input(
        "CUDA_VISIBLE_DEVICES",
        value="0",
        help="Use 0 for GPU 0, 1 for GPU 1, etc. Leave empty to use the default environment.",
    )

    timeout_sec = st.number_input(
        "Timeout seconds",
        min_value=60,
        max_value=1800,
        value=600,
        step=60,
    )

    st.warning(
        "Each click starts a fresh Python process and reloads the model. "
        "This is slower but preserves the original chat_gofa.py loading logic."
    )


st.subheader("Input Used by Original Script")

st.info(
    "The current original `chat_gofa.py` uses its hard-coded input, usually `sample_graph_pyg.pth`. "
    "The web page does not yet modify the graph/question."
)

if CHAT_GOFA.exists():
    st.success("Found original chat_gofa.py")
else:
    st.error(f"chat_gofa.py not found at: {CHAT_GOFA}")

run_clicked = st.button("Run Original GOFA Inference", type="primary")

if "live_result" not in st.session_state:
    st.session_state.live_result = None

if run_clicked:
    if not CHAT_GOFA.exists():
        st.error(f"Cannot run because chat_gofa.py was not found at: {CHAT_GOFA}")
        st.stop()

    with st.spinner("Running original GOFA inference. The model will be loaded in a subprocess."):
        try:
            result = run_original_chat_gofa(
                cuda_visible_devices=cuda_visible_devices,
                timeout_sec=int(timeout_sec),
            )
            st.session_state.live_result = result
        except subprocess.TimeoutExpired:
            st.error(f"chat_gofa.py timed out after {timeout_sec} seconds.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to run chat_gofa.py: {e}")
            st.stop()


result = st.session_state.live_result

if result is None:
    st.info("Click **Run Original GOFA Inference** to generate a live output.")
else:
    if result["returncode"] == 0:
        st.success("Original GOFA inference finished successfully.")
    else:
        st.error(f"chat_gofa.py exited with return code {result['returncode']}.")

    st.markdown("## Live GOFA Output")
    st.success(result["generated_text"])

    st.markdown("## Parsed Input Summary")
    summary = result.get("debug_summary", {})
    if summary:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Nodes", summary.get("num_nodes", "unknown"))
            st.metric("Edges", summary.get("num_edges", "unknown"))

        with col2:
            st.write("Question:")
            st.code(summary.get("question", "unknown"))

            st.write("Answer field in graph:")
            st.code(summary.get("answer_field_in_graph", "unknown"))
    else:
        st.warning("Could not parse graph/question summary from stdout.")

    with st.expander("Command", expanded=False):
        st.code(result["command"], language="bash")
        st.write("Working directory:")
        st.code(result["cwd"], language="bash")

    with st.expander("Raw stdout log", expanded=False):
        st.code(result["stdout"] or "(empty)")

    with st.expander("Raw stderr log", expanded=False):
        st.code(result["stderr"] or "(empty)")
