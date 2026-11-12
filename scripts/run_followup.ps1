# Reproducible, laptop-safe run order for the independent follow-up study (E1-E4, E7) that builds on the pilot's cache (run_chain.ps1 / 00-49 must have been run).
# Uses ONLY the DPR dataset (no external data).
# usage: powershell -File run_followup.ps1 [-steps e1,refs,e2,e3,e7,e4,ranktol,e2sens,sum]      (default: all, in the order below)
# ONE process at a time, below-normal priority (guard.py), 2 BLAS threads, GPU thermal pause (guard.cooldown); every experiment is resumable (results appended per configuration).
# e1  50_independent_replication.py   clean-room scikit-learn re-implementation of the headline contrast (unit-tested against the pilot's features)
# refs 51_verify_refs.py              Crossref verification of every reference of the evidence base
# ops 55_verify_operators.py <pyriemann folder>   unit tests of the whitening / EA / CORAL operators against pyRiemann and the official CORAL formula
# e2  52_operators.py                 adaptation operators (sub / cmvn / CORAL-type whitening / EA) on identical random-order streams, 24 and 96 cells
# e2sens 52_operators.py sens         hyperparameter sensitivity of the operators (cmvn variance floor 0.03 / 0.3, whitening shrinkage 0.25)
# e3  53_composition.py               seconds vs distinct postures; 10-shuffle ladder at 384 cells
# e7  54_time_adjusted.py             does pressure add information beyond the protocol clock? (boosted trees)
# e4  56_cnn_adabn.py                 CNN: driver-relative input vs AdaBN, 2 seeds
# ranktol 59_rank_tol_check.py        diagnostic behind the numerical-rank tolerance of the LDA
# sum 52b / 53b / 54b summaries -> results/*_summary.csv
param([string[]]$steps = @("e1", "refs", "e2", "e3", "e7", "e4", "ranktol", "e2sens", "sum"), [string]$pyriemann = "", [string]$after = "")
# -after "<text>": wait until another run_followup.ps1 process whose command line contains <text> has ended (chains never overlap)
$py = "C:\Users\sheha\Downloads\CNC Turning (Roughness, Forces, Tool Wear)\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"

function Run-Py([string]$script, [string[]]$pyargs = @()) {
    for ($i = 1; $i -le 6; $i++) {              # sklearn / torch imports intermittently crash (0xC000070A) while the machine is loaded: retry on that code only
        Write-Output ("[{0}] {1} {2} (attempt {3})" -f (Get-Date -Format "HH:mm:ss"), $script, ($pyargs -join " "), $i)
        & $py -W ignore $script @pyargs
        if ($LASTEXITCODE -ne -1073740022) { break }
        Start-Sleep -Seconds 3
    }
    if ($LASTEXITCODE -ne 0) { Write-Output ("!! {0} failed with exit code {1}" -f $script, $LASTEXITCODE) }
}

function Wait-Python([string]$pattern) {          # never overlap heavy jobs: wait until a python process running $pattern has ended
    while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }) { Start-Sleep -Seconds 20 }
}

$steps = @($steps | ForEach-Object { $_ -split "," } | Where-Object { $_ })          # accepts -steps e7,e4 from powershell -File as well
if ($after) {
    while (Get-CimInstance Win32_Process -Filter "Name='pwsh.exe'" | Where-Object { $_.CommandLine -match "run_followup\.ps1" -and $_.CommandLine -match [regex]::Escape($after) -and $_.ProcessId -ne $PID }) { Start-Sleep -Seconds 20 }
}
Wait-Python "53_composition|52_operators|54_time|56_cnn|59_rank"
foreach ($s in $steps) {
    switch ($s) {
        "e1"   { Run-Py "50_independent_replication.py" }
        "refs" { Run-Py "51_verify_refs.py" }
        "ops"  { Run-Py "55_verify_operators.py" @($pyriemann) }
        "e2"   { Run-Py "52_operators.py" }
        "e2sens" { Run-Py "52_operators.py" @("sens") }
        "e3"   { Run-Py "53_composition.py" }
        "e7"   { Run-Py "54_time_adjusted.py" }
        "e4"   { Run-Py "56_cnn_adabn.py" }
        "ranktol" { Run-Py "59_rank_tol_check.py" }
        "sum"  { foreach ($x in "52b_summarize.py", "53b_summarize.py", "54b_summarize.py") { Run-Py $x } }
    }
}
Write-Output ("[{0}] follow-up chain finished ({1})" -f (Get-Date -Format "HH:mm:ss"), ($steps -join ","))
