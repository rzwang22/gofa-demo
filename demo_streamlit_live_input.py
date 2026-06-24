import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
WEB_GOFA = BASE_DIR / "chat_gofa_web.py"


def extract_generated_text(stdout: str) -> str:
    lines = stdout.splitlines()

    # Prefer the line after [LIVE_GOFA_OUTPUT].
    for i, line in enumerate(lines):
        if line.strip() == "[LIVE_GOFA_OUTPUT]" and i + 1 < len(lines):
            s = lines[i + 1].strip()
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, list) and len(obj) > 0:
                    return str(obj[0]).replace("</s>", "").strip()
                return str(obj).replace("</s>", "").strip()
            except Exception:
                return s.replace("</s>", "").strip()

    # Fallback: scan from bottom.
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, list) and len(obj) > 0:
                    return str(obj[0]).replace("</s>", "").strip()
            except Exception:
                pass

    return "Could not extract final generated text from output."


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
        summary["question_seen_by_gofa"] = m.group(1).strip()

    m = re.search(r"node_text\[0\]:\s*(.*)", stdout)
    if m:
        summary["node_text_0"] = m.group(1).strip()

    return summary


def run_web_gofa(question: str, cuda_visible_devices: str, timeout_sec: int):
    env = os.environ.copy()

    if cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()

    cmd = [
        sys.executable,
        str(WEB_GOFA),
        "--question",
        question,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "generated_text": extract_generated_text(stdout),
        "debug_summary": extract_debug_summary(stdout),
    }


st.set_page_config(page_title="GOFA Live Input Demo", layout="wide")

st.title("GOFA Live Demo with Web Input")

st.caption(
    "This page calls `chat_gofa_web.py`, which keeps the original GOFA loading logic "
    "but lets the web page override the sample graph question."
)

with st.sidebar:
    st.header("Settings")

    cuda_visible_devices = st.text_input("CUDA_VISIBLE_DEVICES", value="0")

    timeout_sec = st.number_input(
        "Timeout seconds",
        min_value=60,
        max_value=1800,
        value=600,
        step=60,
    )

    st.warning(
        "Each run starts a new Python process and reloads the model. "
        "This is slow but safest for preserving the original code path."
    )

st.subheader("Web Input")

question = st.text_area(
    "Question / Prompt",
    value="Please explain this TypeScript wordFrequency function with clear inline comments.",
    height=120,
)

st.info(
    "The current version overrides the question/prompt in the sample PyG graph. "
    "The rest of the graph still comes from `/mnt/sevenT/wrz_data/GOFA/sample_graph_pyg.pth`."
)

if WEB_GOFA.exists():
    st.success("Found chat_gofa_web.py")
else:
    st.error(f"Missing script: {WEB_GOFA}")

run_clicked = st.button("Run GOFA with This Question", type="primary")

if "live_input_result" not in st.session_state:
    st.session_state.live_input_result = None

if run_clicked:
    if not WEB_GOFA.exists():
        st.error(f"Cannot run because script was not found: {WEB_GOFA}")
        st.stop()

    with st.spinner("Running GOFA with web input. This may take a while."):
        try:
            result = run_web_gofa(
                question=question,
                cuda_visible_devices=cuda_visible_devices,
                timeout_sec=int(timeout_sec),
            )
            st.session_state.live_input_result = result
        except subprocess.TimeoutExpired:
            st.error(f"GOFA timed out after {timeout_sec} seconds.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to run GOFA: {e}")
            st.stop()

result = st.session_state.live_input_result

if result is None:
    st.info("Click **Run GOFA with This Question** to generate an output.")
else:
    if result["returncode"] == 0:
        st.success("GOFA inference finished successfully.")
    else:
        st.error(f"GOFA exited with return code {result['returncode']}.")

    st.markdown("## Web Question")
    st.code(question)

    st.markdown("## Live GOFA Output")
    st.success(result["generated_text"])

    st.markdown("## Parsed Runtime Summary")
    summary = result.get("debug_summary", {})
    if summary:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Nodes", summary.get("num_nodes", "unknown"))
            st.metric("Edges", summary.get("num_edges", "unknown"))

        with col2:
            st.write("Question seen by GOFA:")
            st.code(summary.get("question_seen_by_gofa", "unknown"))

            st.write("node_text[0]:")
            st.code(summary.get("node_text_0", "unknown"))
    else:
        st.warning("Could not parse runtime summary from stdout.")

    with st.expander("Command", expanded=False):
        st.code(result["command"], language="bash")

    with st.expander("Raw stdout log", expanded=False):
        st.code(result["stdout"] or "(empty)")

    with st.expander("Raw stderr log", expanded=False):
        st.code(result["stderr"] or "(empty)")
