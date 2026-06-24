import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


GOFA_ROOT = Path("/mnt/sevenT/wrz_data/GOFA")
CONFIG_PATH = GOFA_ROOT / "configs" / "inference_config_demo.yaml"
RUN_GOFA = GOFA_ROOT / "run_gofa.py"


def extract_block(text: str, start_marker: str, end_marker: str | None = None) -> str:
    idx = text.find(start_marker)
    if idx < 0:
        return ""

    sub = text[idx + len(start_marker):]

    if end_marker:
        end = sub.find(end_marker)
        if end >= 0:
            sub = sub[:end]

    return sub.strip()


def parse_gofa_log(log_text: str) -> dict:
    result = {}

    result["skip_validation"] = "[DEMO_SKIP_VALIDATION]" in log_text

    m = re.search(r"\[DEBUG\] len\(g\.x\):\s*(\d+)", log_text)
    if m:
        result["num_nodes"] = int(m.group(1))

    m = re.search(r"\[DEBUG\] num edges:\s*(\d+)", log_text)
    if m:
        result["num_edges"] = int(m.group(1))

    m = re.search(r"target:\s*(.*?)\s+gen:\s*(.*?)(?:\n|$)", log_text)
    if m:
        result["target"] = m.group(1).strip()
        result["generation"] = m.group(2).strip().replace("</s>", "")

    m = re.search(r"cora_link_test/text_accuracy:([0-9.]+)±([0-9.]+)", log_text)
    if m:
        result["test_accuracy"] = float(m.group(1))
        result["test_accuracy_std"] = float(m.group(2))

    m = re.search(r"question:\s*(.*?)\n-{20,}", log_text, flags=re.S)
    if m:
        q = m.group(1)
        q = re.sub(r"\s+", " ", q).strip()
        result["question"] = q

    node_texts = re.findall(r"node_text\[(\d+)\]:\s*(.*)", log_text)
    result["node_texts"] = [
        {"idx": int(i), "text": t.strip()} for i, t in node_texts[:10]
    ]

    target_nodes = []
    pattern = r"graph node\s+(\d+)\s+->\s+feature\s+(\d+)\s*\n\s*g\.x\[(\d+)\]:\s*(.*)"
    for m in re.finditer(pattern, log_text):
        target_nodes.append(
            {
                "graph_node": m.group(1),
                "feature": m.group(2),
                "x_index": m.group(3),
                "text": m.group(4).strip(),
            }
        )

    # Usually includes question node and target nodes. Keep useful ones.
    result["mapped_nodes"] = target_nodes

    return result


def run_gofa(timeout_sec: int, cuda_visible_devices: str) -> dict:
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["WANDB_SILENT"] = "true"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    if cuda_visible_devices.strip():
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()

    cmd = [
        sys.executable,
        str(RUN_GOFA),
        "--override",
        str(CONFIG_PATH),
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
    log_text = stdout + "\n" + stderr

    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "parsed": parse_gofa_log(log_text),
    }


st.set_page_config(
    page_title="GOFA Normal Inference Demo",
    layout="wide",
)

st.title("GOFA Normal Inference Demo")

st.caption(
    "Runs the standard GOFA inference entrypoint on one sampled cora_link graph task."
)

with st.sidebar:
    st.header("Runtime")

    st.code(f"cd {GOFA_ROOT}\npython run_gofa.py --override {CONFIG_PATH}", language="bash")

    cuda_visible_devices = st.text_input("CUDA_VISIBLE_DEVICES", value="0")

    timeout_sec = st.number_input(
        "Timeout seconds",
        min_value=60,
        max_value=3600,
        value=900,
        step=60,
    )

    st.info(
        "This demo uses offline HuggingFace and offline wandb environment variables. "
        "Validation has been skipped in gp/lightning/training.py for demo speed."
    )

st.subheader("Demo Configuration")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Task", "cora_link")
with col2:
    st.metric("Test samples", "1")
with col3:
    st.metric("Mode", "generate")

with st.expander("Config file", expanded=False):
    if CONFIG_PATH.exists():
        st.code(CONFIG_PATH.read_text(encoding="utf-8"), language="yaml")
    else:
        st.error(f"Config not found: {CONFIG_PATH}")

run_clicked = st.button("Run GOFA Inference", type="primary")

if "gofa_result" not in st.session_state:
    st.session_state.gofa_result = None

if run_clicked:
    with st.spinner("Running GOFA inference. This loads the model and runs one test graph."):
        try:
            st.session_state.gofa_result = run_gofa(
                timeout_sec=int(timeout_sec),
                cuda_visible_devices=cuda_visible_devices,
            )
        except subprocess.TimeoutExpired:
            st.error(f"GOFA run timed out after {timeout_sec} seconds.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to run GOFA: {e}")
            st.stop()

result = st.session_state.gofa_result

if result is None:
    st.info("Click **Run GOFA Inference** to run one normal GOFA test case.")
else:
    parsed = result["parsed"]

    if result["returncode"] == 0:
        st.success("GOFA inference finished successfully.")
    else:
        st.error(f"GOFA exited with return code {result['returncode']}.")

    if parsed.get("skip_validation"):
        st.success("Validation skipped. The demo ran test only.")
    else:
        st.warning("Validation skip marker was not found in the log.")

    st.markdown("## Result")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Nodes", parsed.get("num_nodes", "unknown"))
    with c2:
        st.metric("Edges", parsed.get("num_edges", "unknown"))
    with c3:
        st.metric("Target", parsed.get("target", "unknown"))
    with c4:
        st.metric("Prediction", parsed.get("generation", "unknown"))

    if "test_accuracy" in parsed:
        st.metric("Test Accuracy", f"{parsed['test_accuracy']:.3f}")

    st.markdown("## Graph Question")

    if parsed.get("question"):
        st.info(parsed["question"])
    else:
        st.warning("Question was not parsed from the log.")

    st.markdown("## Model Output")

    st.success(parsed.get("generation", "No generation parsed."))

    st.markdown("## Target Answer")

    st.code(parsed.get("target", "No target parsed."))

    st.markdown("## Graph Nodes Preview")

    node_texts = parsed.get("node_texts", [])
    if node_texts:
        for item in node_texts:
            with st.expander(f"node_text[{item['idx']}]", expanded=False):
                st.write(item["text"])
    else:
        st.warning("No node_text entries parsed.")

    mapped_nodes = parsed.get("mapped_nodes", [])
    if mapped_nodes:
        with st.expander("Question / target node mapping", expanded=False):
            for item in mapped_nodes:
                st.markdown(
                    f"**graph node {item['graph_node']} → feature {item['feature']} → x[{item['x_index']}]**"
                )
                st.code(item["text"])

    with st.expander("Command", expanded=False):
        st.code(result["cmd"], language="bash")

    with st.expander("Raw stdout", expanded=False):
        st.code(result["stdout"] or "(empty)")

    with st.expander("Raw stderr", expanded=False):
        st.code(result["stderr"] or "(empty)")
