"""
Execute Lab4_Image_Denoising_Autoencoders.ipynb and embed all outputs into the notebook.
"""

import sys
import time
import nbformat
from nbclient import NotebookClient

notebook_path = "Lab4_Image_Denoising_Autoencoders.ipynb"

print(f"Reading notebook: {notebook_path}")
nb = nbformat.read(notebook_path, as_version=4)

print(f"Total cells to execute: {len(nb.cells)}")
print("Launching NotebookClient with kernel 'py312'...")

client = NotebookClient(nb, timeout=600, kernel_name='py312')

start_time = time.time()
try:
    client.execute()
    total_duration = time.time() - start_time
    print(f"\nExecution completed successfully in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)!")
except Exception as e:
    print(f"\nExecution encountered an error: {e}", file=sys.stderr)
    raise

# Save the executed notebook with all outputs embedded
print(f"Saving executed notebook to {notebook_path}...")
nbformat.write(nb, notebook_path)

# Verify outputs
total_code = sum(1 for c in nb.cells if c.cell_type == 'code')
with_outputs = sum(1 for c in nb.cells if c.cell_type == 'code' and len(c.get('outputs', [])) > 0)
print(f"\nVerification:")
print(f"  Code cells: {total_code}")
print(f"  Cells with outputs: {with_outputs}")
print("Done!")
