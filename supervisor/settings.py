import os
from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("MODEL", "ollama:gpt-oss:20b-cloud")
DIAGNOSTICS_MODEL=os.getenv("DIAGNOSTICS_MODEL", "ollama:gpt-oss:20b-cloud") 


MCP_SERVERS = {
    "kubernetes": os.getenv("K8S_MCP_URL", "http://127.0.0.1:8080"),
}

ALLOWED_TOOLS = {
    "kubernetes": {
        "list_namespaces",
        "list_nodes",
        "list_pods",
        "pod_events",
        "pod_logs",
        "create_deployment",
        "scale_deployment",
        "delete_deployment",
        "delete_pod",
        "diagnose_cluster",
        "run_diagnostics",
        "exec_shell",
    }
}
