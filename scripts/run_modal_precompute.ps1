param(
    [int]$maxItems = 200
)

Write-Host "Ensure you have activated the venv and run 'modal login' first."
Write-Host "Starting Modal precompute (will run modal_run.py precompute_max_items)"

Write-Host "Running: modal run modal_run.py::precompute_max_items --max-items $maxItems"
modal run modal_run.py::precompute_max_items --max-items $maxItems

if ($LASTEXITCODE -ne 0) {
    Write-Host "Modal precompute returned non-zero exit code: $LASTEXITCODE"
} else {
    Write-Host "Modal precompute finished. Check ./checkpoints/teacher_activations for outputs."
}
