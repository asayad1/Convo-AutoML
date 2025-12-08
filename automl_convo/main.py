"""
This file is the entrypoint of the conversational AutoML agentic workflow.
Defines a chat based, shell like interface.
"""

from utils.drivers import ConversationalAutoMLRunner
import sys
import os

HELP_TEXT = """
Available commands:
  help                     Show this help message
  quit / exit              Exit the program
  set_dataset <path>       Load a new CSV and create a new AutoML runner
  reset                    Reset the conversation state but keep current dataset
  show_state               Print internal conversation state summary
  <any other text>         Will be treated as a natural-language AutoML question
"""

def load_runner(csv_path: str) -> ConversationalAutoMLRunner:
    """Creates a new runner with the given dataset."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found: {csv_path}")
        return None

    print(f"[INFO] Loading dataset: {csv_path}")
    return ConversationalAutoMLRunner(
        csv_path=csv_path,
        max_iterations=3,
        temp_dir="tmp_datasets",
    )


def main():
    # Default dataset
    default_csv = "data/titanic.csv"
    runner = load_runner(default_csv)

    if runner is None:
        print("Fatal error: Could not load default dataset.")
        sys.exit(1)

    print("\nConversational AutoML Shell")
    print("Type 'help' for commands.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        if user_input.startswith("set_dataset"):
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: set_dataset path/to/file.csv")
                continue

            new_path = parts[1].strip()
            new_runner = load_runner(new_path)
            if new_runner:
                runner = new_runner
                print("[INFO] Runner reset with new dataset.\n")
            continue

        if user_input.lower() == "reset":
            runner = load_runner(runner.csv_path)
            print("[INFO] Conversation state reset.\n")
            continue

        if user_input.lower() == "show_state":
            conv = runner.conv_state
            print("\n--- Conversation State ---")
            print(f"Last AutoML State: {conv.last_automl_state is not None}")
            print(f"Q&A History Count: {len(conv.qa_history)}")
            print("--------------------------\n")
            continue

        # Otherwise treat it as a question
        runner.ask(user_input)


if __name__ == "__main__":
    main()
