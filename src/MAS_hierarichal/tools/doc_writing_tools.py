from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Dict, Optional, List

from typing_extensions import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain.tools import tool

# ---------------- Safe Working Directory ----------------
# _TEMP_DIRECTORY = TemporaryDirectory()
# WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

_TEMP_DIRECTORY = mkdtemp()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY)
ALLOWED_EXTENSIONS = {".txt", ".md"}
MAX_FILE_SIZE = 10000  # max characters

def safe_path(file_name: str) -> Path:
    """Return a safe path inside WORKING_DIRECTORY, prevents directory traversal."""
    file_path = (WORKING_DIRECTORY / file_name).resolve()
    if not str(file_path).startswith(str(WORKING_DIRECTORY.resolve())):
        raise ValueError("Invalid file path")
    if file_path.suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("File type not allowed")
    return file_path

# ---------------- Document Tools ----------------

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    file_path = safe_path(file_name)
    content = "\n".join(f"{i+1}. {point}" for i, point in enumerate(points))
    if len(content) > MAX_FILE_SIZE:
        return "Error: outline too large"
    file_path.write_text(content)
    return f"Outline saved to {file_path.name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    file_path = safe_path(file_name)
    lines = file_path.read_text().splitlines()
    start = start or 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    file_path = safe_path(file_name)
    if len(content) > MAX_FILE_SIZE:
        return "Error: content too large"
    file_path.write_text(content)
    return f"Document saved to {file_path.name}"

@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    file_path = safe_path(file_name)
    lines = file_path.read_text().splitlines()

    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text)
        else:
            return f"Error: Line number {line_number} is out of range."

    new_content = "\n".join(lines)
    if len(new_content) > MAX_FILE_SIZE:
        return "Error: edited document too large"

    file_path.write_text(new_content)
    return f"Document edited and saved to {file_path.name}"

# ---------------- Safe Python REPL ----------------

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins

@tool
def python_repl_tool(
    code: Annotated[str, "Python code to execute (safe subset)."],
) -> str:
    """Execute Python code in a restricted environment."""
    try:
        byte_code = compile_restricted(code, "<string>", "exec")
        env = {"__builtins__": safe_builtins}
        exec(byte_code, env)
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed safe code:\n{code}"

# print("WORKING_DIRECTORY:", WORKING_DIRECTORY)
# print("Files:", list(WORKING_DIRECTORY.glob("*")))