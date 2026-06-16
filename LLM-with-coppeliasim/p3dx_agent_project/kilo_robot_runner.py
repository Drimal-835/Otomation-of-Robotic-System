import json
import subprocess
import sys
from pathlib import Path

PREFIX = "coppeliasim:"


def is_mapping_prompt(prompt: str) -> bool:
    p = prompt.lower()
    return any(word in p for word in ["mapping", "map", "explore", "survey", "wall"])


def open_map_png():
    png = Path("map_graph.png")

    if not png.exists():
        print("Mapping finished, but map_graph.png was not found.")
        return

    print(f"Opening mapping result: {png.resolve()}")

    # Open inside VS Code
    subprocess.run(
        ["code", str(png)],
        check=False,
        capture_output=True,
        text=True,
    )


def main():
    if len(sys.argv) < 2:
        print('Usage: python kilo_robot_runner.py "coppeliasim: go to the goal"')
        return

    text = " ".join(sys.argv[1:]).strip()

    if not text.lower().startswith(PREFIX):
        print("Normal conversation detected. No robot command executed.")
        return

    prompt = text[len(PREFIX):].strip()

    if not prompt:
        print("Empty CoppeliaSim command.")
        return

    # Stop previous command first
    subprocess.run(
        [sys.executable, "agent_manager.py", "stop"],
        capture_output=True,
        text=True,
    )

    cmd = [
        sys.executable,
        "agent_manager.py",
        prompt,
    ]

    print("Running robot mission...")

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        data = json.loads(result.stdout)
        mission = data.get("mission", "")
        reason = data.get("reason", "done")

        if mission == "mapping" or is_mapping_prompt(prompt):
            print("Done: mapping completed.")
            open_map_png()
        else:
            print(f"Done: {reason}")

    except Exception:
        print("Done.")

        if is_mapping_prompt(prompt):
            open_map_png()


if __name__ == "__main__":
    main()