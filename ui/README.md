# Optional local simulator interface

Install and launch from the repository root:

```powershell
python -m pip install -e ".[ui]"
streamlit run ui/simulator_app.py
```

For IBM mode, install `.[ui,ibm]`. The interface offers ideal/normal, program
fault, synthetic hardware, mixed program-plus-synthetic, and explicit IBM
execution modes. Real execution requires separate explicit confirmation and
never falls back to synthetic mode. Normal and synthetic-only modes use the
reference as the submitted circuit to isolate sampling/noise behavior.
Circuits are entered as JSON operation lists using the same
gate names and operand order as the Python operation-list API. The UI shows
counts, evidence, missing evidence, and the result-specific math explanation.

The IBM key and optional instance value stay in Streamlit session memory for
the active browser session so reruns do not unexpectedly clear the entry. The
app has no code that writes either secret to files, environment variables, URLs,
results, or app-owned logs; a clear button removes them. Closing the session
discards them. This is not a promise about the operating system, a reverse
proxy, hosting provider, browser extensions, or IBM's own service. Run locally
on a trusted machine; a remote hosted UI necessarily receives the secret on its
server. Real mode never silently falls back to synthetic execution.

Noise sliders in synthetic modes create an explicit stress fixture. Those
values are not claimed to reproduce an IBM backend. See
[`../docs/algorithm_reference.md`](../docs/algorithm_reference.md) for the
model assumptions and mathematical limitations.
