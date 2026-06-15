"""Streamlit demo for a lightweight GOFA inference replay."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from gofa_demo.gofa_runner import LiveGOFANotAvailable, run_live, run_replay


PAGE_TITLE = "GOFA Inference Demo"


def _node_role(node: dict[str, Any]) -> str:
    return str(node.get("role") or node.get("type") or "").lower()


def _node_color(node: dict[str, Any]) -> tuple[str, str]:
    role = _node_role(node)
    if role == "question":
        return "#2563eb", "#eff6ff"
    if role == "complete":
        return "#16a34a", "#f0fdf4"
    return "#475569", "#f8fafc"


def build_graph_svg(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    positions = {
        "paper": (90, 92),
        "method": (325, 92),
        "question": (560, 92),
        "evidence": (210, 270),
        "complete": (500, 270),
    }

    fallback_x = 110
    fallback_y = 90
    for index, node in enumerate(nodes):
        node_id = node["id"]
        positions.setdefault(node_id, (fallback_x + (index % 3) * 230, fallback_y + (index // 3) * 160))

    edge_markup = []
    for edge in edges:
        source = positions.get(edge["source"])
        target = positions.get(edge["target"])
        if not source or not target:
            continue
        x1, y1 = source
        x2, y2 = target
        label = html.escape(edge.get("label", ""))
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        edge_markup.append(
            f"""
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)" />
            <text x="{mx}" y="{my - 8}" text-anchor="middle" fill="#64748b" font-size="12">{label}</text>
            """
        )

    node_markup = []
    for node in nodes:
        node_id = node["id"]
        x, y = positions[node_id]
        stroke, fill = _node_color(node)
        label = html.escape(node.get("label", node_id))
        node_type = html.escape(node.get("type", "node"))
        node_markup.append(
            f"""
            <g>
              <rect x="{x - 76}" y="{y - 34}" width="152" height="68" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2" />
              <text x="{x}" y="{y - 5}" text-anchor="middle" fill="#0f172a" font-size="14" font-weight="700">{label}</text>
              <text x="{x}" y="{y + 18}" text-anchor="middle" fill="#475569" font-size="12">{node_type}</text>
            </g>
            """
        )

    return f"""
    <div style="width:100%; overflow-x:auto;">
      <svg viewBox="0 0 680 360" width="100%" height="360" role="img" aria-label="GOFA sample graph">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
          </marker>
        </defs>
        <rect x="0" y="0" width="680" height="360" fill="#ffffff" />
        {''.join(edge_markup)}
        {''.join(node_markup)}
      </svg>
    </div>
    """


def render_timeline(pipeline: list[dict[str, Any]]) -> None:
    cols = st.columns(len(pipeline))
    for col, stage in zip(cols, pipeline):
        with col:
            st.markdown(
                f"""
                <div class="timeline-step">
                  <div class="timeline-dot"></div>
                  <div class="timeline-name">{html.escape(stage['name'])}</div>
                  <div class="timeline-detail">{html.escape(stage['detail'])}</div>
                  <div class="timeline-time">{stage['duration_ms']:.2f} ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_runtime(runtime: dict[str, Any]) -> None:
    rows = [
        ("Mode", runtime["mode"]),
        ("Python", runtime["python"]),
        ("Platform", runtime["platform"]),
        ("GPU required", str(runtime["gpu_required"])),
        ("Checkpoint loaded", str(runtime["checkpoint_loaded"])),
        ("Checkpoint downloaded", str(runtime["checkpoint_downloaded"])),
        ("CWD", runtime["cwd"]),
    ]
    for key, value in rows:
        st.markdown(f"**{key}:** `{value}`")


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.5rem; }
          .timeline-step {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            min-height: 132px;
            background: #ffffff;
          }
          .timeline-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #0f766e;
            margin-bottom: 8px;
          }
          .timeline-name {
            color: #0f172a;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
            text-transform: capitalize;
          }
          .timeline-detail {
            color: #475569;
            font-size: 12px;
            min-height: 38px;
          }
          .timeline-time {
            color: #64748b;
            font-size: 12px;
            margin-top: 8px;
          }
          .legend-row {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin: 4px 0 12px 0;
          }
          .legend-item {
            display: flex;
            gap: 8px;
            align-items: center;
            color: #334155;
            font-size: 14px;
          }
          .legend-swatch {
            width: 14px;
            height: 14px;
            border-radius: 4px;
            display: inline-block;
            border: 2px solid currentColor;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(PAGE_TITLE)

    with st.sidebar:
        st.header("Run Settings")
        task_type = st.radio("Task type", ["QA", "Completion"], horizontal=True)
        live_mode = st.checkbox("Live GOFA mode", value=False)
        st.caption("Replay mode is the default and does not load a checkpoint or require GPU.")

    replay = run_replay(task_type)
    active_result = replay

    if live_mode:
        st.info("Live GOFA mode is enabled. The demo will probe the original entrypoint and fall back to replay output if it is unavailable.")
        try:
            live_result = run_live(task_type)
            st.success("Live GOFA entrypoint probe completed.")
            with st.expander("Live GOFA probe output", expanded=False):
                st.code(live_result.get("stdout") or "(no stdout)", language="text")
                if live_result.get("stderr"):
                    st.code(live_result["stderr"], language="text")
        except LiveGOFANotAvailable as exc:
            st.error(str(exc))
            st.caption("The page is still running with cached replay output.")
        except Exception as exc:
            st.error(f"Live GOFA mode failed gracefully: {exc}")
            st.caption("The page is still running with cached replay output.")

    graph = active_result["graph"]

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("sample_graph.json")
        st.code(json.dumps(graph, indent=2, ensure_ascii=False), language="json")

    with right:
        st.subheader("Graph Structure")
        st.markdown(
            """
            <div class="legend-row">
              <div class="legend-item"><span class="legend-swatch" style="color:#2563eb;background:#eff6ff"></span>Question node: user query anchor</div>
              <div class="legend-item"><span class="legend-swatch" style="color:#16a34a;background:#f0fdf4"></span>Complete node: generated/completed answer target</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        components.html(build_graph_svg(graph), height=390, scrolling=False)

        st.subheader("Task")
        st.markdown(f"**Task type:** `{task_type}`")
        st.markdown("**Question node:** `question` highlights where the query enters the graph.")
        st.markdown("**Complete node:** `complete` highlights where GOFA generation is decoded.")

    st.subheader("Pipeline Timeline")
    render_timeline(active_result["pipeline"])

    result_col, runtime_col = st.columns([1.15, 0.85])
    with result_col:
        st.subheader("GOFA Generated Result")
        st.markdown(active_result["output"]["text"])
        with st.expander("Converted input preview", expanded=False):
            st.code(active_result["converted_input"]["prompt_preview"], language="text")

    with runtime_col:
        st.subheader("Runtime")
        render_runtime(active_result["runtime"])


if __name__ == "__main__":
    main()
