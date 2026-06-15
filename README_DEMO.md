# GOFA Streamlit Demo

This demo shows a lightweight GOFA inference chain without touching GOFA core model code. The default mode is replay mode, which uses cached JSON output and does not require GPU, checkpoints, training, or downloads.

## Files

- `demo_streamlit.py`: Streamlit UI.
- `gofa_demo/gofa_runner.py`: Pure-Python replay runner and optional Live GOFA probe.
- `gofa_demo/cached_outputs/sample_graph.json`: Small graph fixture displayed by the UI.
- `gofa_demo/cached_outputs/sample_result.json`: Cached replay output.

## Start on a Server

Install Streamlit if it is not already available:

```bash
pip install streamlit
```

Start the demo:

```bash
streamlit run demo_streamlit.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://<server-ip>:8501` in a browser.

## Replay Mode

Replay mode is the default. It runs in a normal Python environment and uses only cached local files:

```bash
python -m gofa_demo.gofa_runner --task-type QA
python -m gofa_demo.gofa_runner --task-type Completion
```

Replay mode reports the full demo timeline:

1. load graph
2. validate graph
3. convert input
4. load model
5. generate
6. decode output

The `load model` stage is a replay stub and explicitly reports `checkpoint_loaded=false`.

## Live GOFA Mode

Live GOFA mode is optional. When the checkbox is enabled, the UI tries to probe the original `chat_gofa.py` entrypoint. If the file or dependencies are missing, the page shows the error and continues with replay output.

To point the demo at a configured live command, set:

```bash
export GOFA_DEMO_LIVE_COMMAND="python chat_gofa.py --help"
```

Do not use Live GOFA mode unless the GOFA environment, dependencies, and checkpoints are already prepared. This demo does not download checkpoints or run training.
