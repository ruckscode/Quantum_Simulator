# Reproduction

Run the bounded inventory artifact builder from the repository root:

```powershell
.\.qbugs_env\Scripts\python.exe experiments\create_full_data_qml_v3_campaign.py
```

This command only inventories the local sources and indexes existing artifacts. It does not train. Before resuming QML training, ensure the idle system has at least 1 GiB available RAM, then rerun the memory measurement. Do not run any frozen-test evaluation during optimization.
