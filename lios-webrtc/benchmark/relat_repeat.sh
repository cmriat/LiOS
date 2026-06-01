#!/usr/bin/env bash
# Repeat a bidirectional (clock-offset-cancelled) latency measurement N times and
# collect the true one-way d (ms) of each run, for building error-bar charts.
#
# Usage:
#   N=5 LABEL=p2p     SCRIPT=benchmark/cross_p2p_latency_bidir.sh  benchmark/relat_repeat.sh
#   N=5 LABEL=lk_self SCRIPT=benchmark/cross_lk_latency_bidir.sh   benchmark/relat_repeat.sh
#   N=5 LABEL=lk_cloud SCRIPT=benchmark/run_lat.sh \
#       LK_URL=wss://xxx.livekit.cloud LK_KEY=... LK_SECRET=...     benchmark/relat_repeat.sh
#
# Output: appends one float per run to benchmark/relat_results/<LABEL>.txt
set -u
cd "$(cd "$(dirname "$0")/../" && pwd)"
N=${N:-5}
LABEL=${LABEL:?need LABEL}
SCRIPT=${SCRIPT:?need SCRIPT path}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-220}
OUTDIR=benchmark/relat_results
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${LABEL}.txt"
: > "$OUT"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=RELAY_HOST,your-host.example.com,.example.com,.livekit.cloud,127.0.0.1,localhost

echo "=== [$LABEL] repeat x$N  script=$SCRIPT ==="
ok=0
for i in $(seq 1 "$N"); do
  echo "--- [$LABEL] run $i/$N ---"
  out=$(timeout "$PER_RUN_TIMEOUT" bash "$SCRIPT" 2>&1)
  d=$(printf '%s\n' "$out" | grep -aoE 'RESULT_D=[0-9.]+' | tail -1 | cut -d= -f2)
  if [ -n "$d" ]; then
    echo "$d" >> "$OUT"
    ok=$((ok+1))
    echo "    -> d=${d}ms  (ok=$ok)"
  else
    echo "    -> FAILED (no RESULT_D); tail:"
    printf '%s\n' "$out" | tail -4 | sed 's/^/        /'
  fi
  sleep 2
done

echo "=== [$LABEL] collected $ok/$N runs into $OUT ==="
python3 - "$OUT" "$LABEL" <<'PY'
import sys, statistics as st
vals=[float(x) for x in open(sys.argv[1]) if x.strip()]
if not vals:
    print(f"[{sys.argv[2]}] no values"); raise SystemExit
med=st.median(vals)
print(f"[{sys.argv[2]}] n={len(vals)} median={med:.2f} min={min(vals):.2f} max={max(vals):.2f} "
      f"mean={st.mean(vals):.2f}" + (f" std={st.pstdev(vals):.2f}" if len(vals)>1 else ""))
print("  values:", ", ".join(f"{v:.1f}" for v in vals))
PY
