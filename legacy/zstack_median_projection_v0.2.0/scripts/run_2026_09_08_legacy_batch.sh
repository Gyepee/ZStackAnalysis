#!/usr/bin/env bash
set -uo pipefail

pipeline_root="/home/jisooj/Workspace/ZStackAnalysis/legacy/zstack_median_projection_v0.2.0"
job_root="$pipeline_root/background_jobs/2026-09-08"
status_file="$job_root/status.tsv"
mkdir -p "$job_root"
printf 'scan_id\tstarted_utc\tfinished_utc\trun_status\tvalidation_status\n' > "$status_file"

sources=(
  "/datajoint-data/data/jisooj/JJ_ROS-2338_2026-09-08_scan9G23WFYJ_sess9G23WFYJ/scan9G23WFYJ_JJ_ROS-2338_02027.tif"
  "/datajoint-data/data/jisooj/JJ_ROS-2338_2026-09-08_scan9G23WITP_sess9G23WITP/scan9G23WITP_JJ_ROS-2338_02028.tif"
  "/datajoint-data/data/jisooj/JJ_ROS-2338_2026-09-08_scan9G23WO79_sess9G23WO79/scan9G23WO79_JJ_ROS-2338_02030.tif"
  "/datajoint-data/data/jisooj/JJ_ROS-2338_2026-09-08_scan9G23WWL9_sess9G23WWL9/scan9G23WWL9_JJ_ROS-2338_02031.tif"
  "/datajoint-data/data/jisooj/JJ_ROS-2338_2026-09-08_scan9G23WZPO_sess9G23WZPO/scan9G23WZPO_JJ_ROS-2338_02032.tif"
  "/datajoint-data/data/jisooj/JJ_ROS-2338_2026-09-08_scan9G23X3A9_sess9G23X3A9/scan9G23X3A9_JJ_ROS-2338_02034.tif"
)

cd "$pipeline_root" || exit 1
for source in "${sources[@]}"; do
  filename="${source##*/}"
  scan_id="${filename%%_*}"
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log="$job_root/${scan_id}.log"
  if PYTHONPATH=src python scripts/reconstruct_median_stack.py \
      --source-tiff "$source" --channel 3 > "$log" 2>&1; then
    run_status="pass"
    run_dir="$(find analysis_runs -mindepth 1 -maxdepth 1 -type d -name "*_${scan_id}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "$run_dir" ]] && PYTHONPATH=src python scripts/validate_run.py "$run_dir" > "$job_root/${scan_id}.validation.json" 2>&1; then
      validation_status="pass"
    else
      validation_status="fail"
    fi
  else
    run_status="fail"
    validation_status="not_run"
  fi
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$scan_id" "$started" "$finished" "$run_status" "$validation_status" >> "$status_file"
done
