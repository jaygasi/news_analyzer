import json
import os
import platform

# Define the root directory (adjust if needed)
root_dir = "C:/Users/JayHy/OneDrive/Projects" if platform.system() == "Windows" else "/Users/JayHy/OneDrive/Projects"
workspace_file = os.path.join(root_dir, "Projects.code-workspace")

# Define the projects and their paths
projects = [
    {"name": "biomed_analyzer", "path": "news_analyzer/biomed_analyzer"},
    {"name": "earnings_analyzer", "path": "news_analyzer/earnings_analyzer"},
    {"name": "news_catalyst", "path": "news_analyzer/news_catalyst"}
]

# Determine the correct interpreter path based on OS
interpreter_path = "venv/Scripts/python.exe" if platform.system() == "Windows" else "venv/bin/python"

# Step 1: Create or update the Projects.code-workspace file
workspace_config = {
    "folders": [
        {"path": project["path"], "name": project["name"]} for project in projects
    ],
    "settings": {
        "python.terminal.activateEnvironment": True,
        "python.terminal.activateEnvInCurrentTerminal": True
    }
}

with open(workspace_file, "w") as f:
    json.dump(workspace_config, f, indent=4)

print(f"Updated {workspace_file}")

# Step 2: Create .vscode/settings.json for each project
for project in projects:
    project_dir = os.path.join(root_dir, project["path"])
    vscode_dir = os.path.join(project_dir, ".vscode")
    settings_file = os.path.join(vscode_dir, "settings.json")

    # Create .vscode directory if it doesn't exist
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)

    # Create settings.json with the interpreter path
    settings_config = {
        "python.defaultInterpreterPath": interpreter_path
    }

    with open(settings_file, "w") as f:
        json.dump(settings_config, f, indent=4)

    print(f"Created/Updated {settings_file}")