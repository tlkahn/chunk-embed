#!/usr/bin/env bash
#
# Build text-chunker and sentenza as universal macOS binaries
# and place them in resources/bin/ for Briefcase packaging.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$PROJECT_ROOT/resources/bin"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Rust crates to build (name = cargo package name, repo = git URL)
CRATES=(
    "text-chunker|https://github.com/tlkahn/text-chunker"
    "sentenza|https://github.com/tlkahn/sentenza"
)

TARGETS=(aarch64-apple-darwin)

# ── Pre-flight checks ────────────────────────────────────────────────────

command -v rustup >/dev/null 2>&1 || { echo "Error: rustup not found"; exit 1; }
command -v cargo  >/dev/null 2>&1 || { echo "Error: cargo not found"; exit 1; }
if [ "${#TARGETS[@]}" -gt 1 ]; then
    command -v lipo >/dev/null 2>&1 || { echo "Error: lipo not found (Xcode CLI tools required)"; exit 1; }
fi

echo "Adding Rust targets…"
for target in "${TARGETS[@]}"; do
    rustup target add "$target" 2>/dev/null || true
done

mkdir -p "$OUT_DIR"

# ── Build each crate ─────────────────────────────────────────────────────

for entry in "${CRATES[@]}"; do
    IFS='|' read -r crate repo <<< "$entry"
    echo ""
    echo "━━━ Building $crate ━━━"

    for target in "${TARGETS[@]}"; do
        echo "  ▸ $target"
        cargo install \
            --git "$repo" \
            --target "$target" \
            --root "$TMP_DIR/$crate-$target" \
            --force
    done

    # Merge into universal binary (or just copy if single target)
    if [ "${#TARGETS[@]}" -gt 1 ]; then
        echo "  ▸ Creating universal binary…"
        lipo_args=()
        for target in "${TARGETS[@]}"; do
            lipo_args+=("$TMP_DIR/$crate-$target/bin/$crate")
        done
        lipo -create "${lipo_args[@]}" -output "$OUT_DIR/$crate"
    else
        echo "  ▸ Copying single-arch binary…"
        cp "$TMP_DIR/$crate-${TARGETS[0]}/bin/$crate" "$OUT_DIR/$crate"
    fi

    chmod +x "$OUT_DIR/$crate"
    echo "  ✓ $OUT_DIR/$crate"
    file "$OUT_DIR/$crate"
done

echo ""
echo "Done — universal binaries are in $OUT_DIR/"
ls -lh "$OUT_DIR/"
