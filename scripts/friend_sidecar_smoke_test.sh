#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
RUN_ID="${RUN_ID:-$(date +%s)-$$}"
SIDECAR_ID="friend-smoke-${RUN_ID}"
PACKAGE="friend-smoke-package-${RUN_ID}"
EXPERIENCE="${EXPERIENCE:-The room costs five gold pieces.}"

require() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'FAIL missing command: %s\n' "$1" >&2
        exit 1
    }
}

api() {
    local method="$1"
    local path="$2"
    local body="${3:-}"

    printf 'API %s %s\n' "$method" "$path" >&2
    if [[ -n "$body" ]]; then
        curl -fsS -X "$method" "$BASE_URL$path" \
            -H 'Content-Type: application/json' \
            -d "$body"
    else
        curl -fsS -X "$method" "$BASE_URL$path"
    fi
}

cleanup() {
    api DELETE "/v1/sidecar/$SIDECAR_ID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

require curl
require jq

printf 'TEST health\n'
api GET /health >/dev/null
printf 'PASS health\n'

printf 'TEST create sidecar %s\n' "$SIDECAR_ID"
api POST /v1/sidecar/create "$(jq -cn --arg id "$SIDECAR_ID" \
    '{id:$id,learning_enabled:true}')" >/dev/null

BEFORE="$(api GET "/v1/sidecar/status?id=$SIDECAR_ID")"
SHA_BEFORE="$(jq -er '.sha256' <<<"$BEFORE")"
printf 'initial SHA %s\n' "$SHA_BEFORE"

printf 'TEST batch multi-token observation\n'
OBSERVE="$(api POST /v1/sidecar/observe_sequence "$(jq -cn \
    --arg id "$SIDECAR_ID" --arg text "$EXPERIENCE" \
    '{id:$id,text:$text,mode:"batch"}')")"
TOKENS="$(jq -er '.tokens_observed' <<<"$OBSERVE")"
WRITES="$(jq -er '.writes_realized' <<<"$OBSERVE")"
SHA_AFTER="$(jq -er '.sha256_after' <<<"$OBSERVE")"
RATE="$(jq -er '.observed_tokens_per_second' <<<"$OBSERVE")"

if (( TOKENS <= 1 )); then
    printf 'FAIL expected multi-token input, got %s token(s)\n' "$TOKENS" >&2
    exit 1
fi
if (( WRITES > 0 )) && [[ "$SHA_BEFORE" == "$SHA_AFTER" ]]; then
    printf 'FAIL %s writes occurred but SHA did not change\n' "$WRITES" >&2
    exit 1
fi
printf 'PASS observation tokens=%s writes=%s rate=%s tok/s\n' "$TOKENS" "$WRITES" "$RATE"
printf 'learned SHA %s\n' "$SHA_AFTER"

printf 'TEST save, unload, reload\n'
api POST /v1/sidecar/save "$(jq -cn --arg id "$SIDECAR_ID" --arg path "$PACKAGE" \
    '{id:$id,path:$path}')" >/dev/null
api POST /v1/sidecar/unload "$(jq -cn --arg id "$SIDECAR_ID" '{id:$id}')" >/dev/null
api POST /v1/sidecar/load "$(jq -cn --arg id "$SIDECAR_ID" --arg path "$PACKAGE" \
    '{id:$id,path:$path,learning_enabled:true}')" >/dev/null
SHA_RELOADED="$(api GET "/v1/sidecar/status?id=$SIDECAR_ID" | jq -er '.sha256')"

if [[ "$SHA_AFTER" != "$SHA_RELOADED" ]]; then
    printf 'FAIL learned SHA %s != reloaded SHA %s\n' "$SHA_AFTER" "$SHA_RELOADED" >&2
    exit 1
fi
printf 'PASS save/reload SHA %s\n' "$SHA_RELOADED"
printf 'PASS friend sidecar smoke test\n'
