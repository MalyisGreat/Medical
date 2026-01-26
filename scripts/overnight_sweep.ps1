param(
  [string]$Device = $env:DEVICE
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projDir = Resolve-Path (Join-Path $scriptDir "..")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $projDir "research\\logs\\$timestamp"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not $Device -or $Device.Trim().Length -eq 0) {
  $Device = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
}

$steps = if ($Device -eq "cuda") { 200 } else { 40 }
$batch = if ($Device -eq "cuda") { 8 } else { 2 }
$compile = @()
if ($Device -eq "cuda") {
  $compile = @("--compile")
}

function Run {
  param(
    [string]$Name,
    [string[]]$Args
  )

  $logPath = Join-Path $logDir "$Name.log"
  Write-Host "RUN: $Name"
  Write-Host "ARGS: $($Args -join ' ')"
  & python train.py @Args 2>&1 | Tee-Object -FilePath $logPath
}

Push-Location $projDir

# Smoke checks
& python model.py 2>&1 | Tee-Object -FilePath (Join-Path $logDir "model_smoke.log")
& python train.py --test_only --datasets wikitext 2>&1 | Tee-Object -FilePath (Join-Path $logDir "data_smoke.log")

# Cheap prep test (small shards)
& python prepare_data.py --max_tokens 10000 --datasets wikitext --output ./data_shards_test --seq_length 128 --shard_size 10000 2>&1 | Tee-Object -FilePath (Join-Path $logDir "prepare_smoke.log")

# Ablations: optimizer + seq length
Run "wikitext_adamw_seq128" (@("--model","tiny","--datasets","wikitext","--max_seq_length","128","--max_steps",$steps,"--batch_size",$batch,"--optimizer","adamw","--device",$Device) + $compile)
Run "wikitext_muon_seq128"  (@("--model","tiny","--datasets","wikitext","--max_seq_length","128","--max_steps",$steps,"--batch_size",$batch,"--optimizer","muon","--device",$Device) + $compile)
Run "wikitext_adamw_seq256" (@("--model","tiny","--datasets","wikitext","--max_seq_length","256","--max_steps",$steps,"--batch_size",$batch,"--optimizer","adamw","--device",$Device) + $compile)
Run "wikitext_muon_seq256"  (@("--model","tiny","--datasets","wikitext","--max_seq_length","256","--max_steps",$steps,"--batch_size",$batch,"--optimizer","muon","--device",$Device) + $compile)

# Streaming sanity (small)
Run "stream_synth_muon" (@("--model","tiny","--streaming","--datasets","synth","--max_seq_length","256","--max_steps",$steps,"--batch_size",$batch,"--optimizer","muon","--device",$Device) + $compile)

Pop-Location

Write-Host ""
Write-Host "Logs written to: $logDir"
