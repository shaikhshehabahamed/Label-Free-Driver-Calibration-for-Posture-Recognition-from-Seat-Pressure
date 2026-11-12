# Reproducible, laptop-safe run order for the calibration study (v2: 22-driver primary cohort, columns 64-160, order-controlled).
# usage: powershell -File run_chain.ps1 all|audit|lda|heavy|protocols|final|finalnow
# ONE process at a time, below-normal priority (guard.py), 2 BLAS threads, GPU thermal pause (guard.cooldown); every GPU experiment is resumable.
# Interpreter: shared venv with CUDA torch + xgboost-GPU.  sklearn/torch imports intermittently crash with exit code -1073740022 when the
# machine is loaded, so each step is retried on that code only.  A failing step is logged and the chain continues.
param([string]$chain = "all")
$py = "C:\Users\sheha\Downloads\CNC Turning (Roughness, Forces, Tool Wear)\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"

function Run-Py([string]$script, [string[]]$pyargs = @(), [hashtable]$envv = @{}) {
    foreach ($k in $envv.Keys) { Set-Item -Path "Env:$k" -Value $envv[$k] }
    for ($i = 1; $i -le 6; $i++) {
        Write-Output ("[{0}] {1} {2} {3} (attempt {4})" -f (Get-Date -Format "HH:mm:ss"), ($envv.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }), $script, ($pyargs -join " "), $i)
        & $py -W ignore $script @pyargs
        if ($LASTEXITCODE -ne -1073740022) { break }
        Start-Sleep -Seconds 3
    }
    foreach ($k in $envv.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
    if ($LASTEXITCODE -ne 0) { Write-Output ("!! {0} failed with exit code {1}" -f $script, $LASTEXITCODE) }
}

function Wait-Chain([string]$name) {   # never overlap heavy jobs: wait until another run_chain.ps1 <name> process has ended
    while ($true) {
        $other = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match "run_chain\.ps1\s+$name" -and $_.ProcessId -ne $PID }
        if (-not $other) { break }
        Start-Sleep -Seconds 20
    }
}

function Run-Audit {   # data audit, geometry, label-repair figure, segment-level view, time-only floor (CPU-light)
    Run-Py "03_eda.py"
    Run-Py "05_posture_contrast.py"
    Run-Py "47_geometry_audit.py"
    Run-Py "46_repair_figure.py"
    Run-Py "10_segment_variants.py"
    Run-Py "23_time_baseline.py"
}

function Run-Lda {   # every GPU-LDA experiment (about 10 s per configuration)
    foreach ($e in "ladder", "order", "window", "op", "region", "trim", "res", "sens") { Run-Py "31_gpu_run.py" @($e) }
    Run-Py "31_gpu_run.py" @("sens") @{ DATASET = "all" }        # sensitivity: 30 drivers, repaired labels
    Run-Py "31_gpu_run.py" @("sens") @{ DATASET = "orig" }       # sensitivity: 30 drivers, first cleaning
    Run-Py "31_gpu_run.py" @("sens", "8")                        # sensitivity: rows >= 8 s into a segment
    Run-Py "31_gpu_run.py" @("sens") @{ KEEP_MARKER_FRAMES = "1" }   # sensitivity: marker presses left in the labelled data
    Run-Py "42_per_posture.py"
    Run-Py "44_order_audit.py"
    Run-Py "33_learning_curve.py"
    Run-Py "34_nested_tuning.py" @("none")
    Run-Py "34_nested_tuning.py" @("run")
    Run-Py "36_descriptors.py"
    Run-Py "38_identifiability.py"
}

function Run-Heavy {   # boosted trees on GPU, CNN, anthropometrics (needs the learner results)
    Run-Py "31_gpu_run.py" @("learners")
    Run-Py "39_anthro_final.py"
    Run-Py "37_cnn.py"
}

function Run-Protocols {   # scikit-learn protocol audit (LDA, LogReg, kNN, RF) with 2 workers - the longest CPU job, run last
    Run-Py "04_protocols.py"
}

function Run-Final {   # figures and the proposal document (CPU-light); needs python-docx -> the CWRU venv
    Run-Py "22_figures.py"
    $pyd = "C:\Users\sheha\Downloads\CWRU_Bearing_Fault_Diagnosis\.venv\Scripts\python.exe"
    $out = Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) "PhD_Proposal_Seat_Pressure_Driver_Calibration.docx"
    Write-Output ("[{0}] building {1}" -f (Get-Date -Format "HH:mm:ss"), $out)
    & $pyd -W ignore 40_build_docx.py $out
}

switch ($chain) {
    "all" { Run-Audit; Run-Lda; Run-Heavy; Run-Protocols }
    "audit" { Run-Audit }
    "lda" { Run-Lda }
    "heavy" { Run-Heavy }
    "protocols" { Run-Protocols }
    "final" { Wait-Chain "all"; Run-Final }   # starts automatically when the computing chain has finished
    "finalnow" { Run-Final }
}
Write-Output ("[{0}] chain '{1}' finished" -f (Get-Date -Format "HH:mm:ss"), $chain)
