import papermill as pm
from pathlib import Path
import subprocess
import typer

import sys
sys.path.append("utils/")
from messaging import send_message

def remove_cell_outputs(notebook_path: str):
    command = f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {notebook_path}"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    
def run_notebook(notebook_path, verbose=True):
    if verbose:
        send_message(f"Starting execution of: {notebook_path}.")
    notebook_path = Path(notebook_path)
    workdir = notebook_path.parent
    output_path = notebook_path.parent / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / notebook_path.name
    
    remove_cell_outputs(str(notebook_path))
    try:
        pm.execute_notebook(
            input_path=str(notebook_path), 
            output_path=str(output_path),
            cwd=str(workdir),
        )
        if verbose:
            send_message(f"Execution of {str(notebook_path)} finished successfully.")
    except:
        if verbose:
            send_message(f"Execution of {str(notebook_path)} finished with errors.")
            
if __name__ == "__main__":
    typer.run(run_notebook)
