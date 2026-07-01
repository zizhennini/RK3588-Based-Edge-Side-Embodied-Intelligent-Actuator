#!/usr/bin/env bash
set -eu

device="/dev/video21"
width="1920"
height="1080"
pixfmt="MJPG"
skip="5"
out_dir="/home/elf/work/Qwen-Chat-Assistant/photos"
prefix="camera"
timestamp=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-dir) out_dir="${2:?}"; shift 2 ;;
    --prefix)  prefix="${2:?}"; shift 2 ;;
    --timestamp) timestamp="${2:?}"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "$out_dir"
[ -z "$timestamp" ] && timestamp="$(date +%Y%m%d_%H%M%S)"
jpg="${out_dir%/}/${prefix}_${timestamp}.jpg"

v4l2-ctl -d "$device" \
  --set-fmt-video="width=${width},height=${height},pixelformat=${pixfmt}" \
  --stream-mmap=4 \
  --stream-skip="$skip" \
  --stream-count=1 \
  --stream-to="$jpg" 2>/dev/null

printf 'jpg=%s\n' "$jpg"
printf 'device=%s\n' "$device"
printf 'format=%sx%s %s\n' "$width" "$height" "$pixfmt"
