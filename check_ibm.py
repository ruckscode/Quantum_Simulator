from ibm_validation import _environment_runtime_service, discover_ibm_backends

service = _environment_runtime_service("ibm_quantum_platform")
result = discover_ibm_backends(service=service)

if result["error"]:
    raise RuntimeError(result["error"])

print("IBM service access: OK")
print("Backends:")
for backend in result["backends"]:
    print(f"{backend['name']} | qubits={backend['num_qubits']} | operational={backend['operational']}")
