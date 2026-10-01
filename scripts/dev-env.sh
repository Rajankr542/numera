# Source before building: `source scripts/dev-env.sh` (D-056).
# Pins the SDK the main checkout was configured with; the newest Command Line
# Tools SDK on this machine has .tbd files the linker rejects.
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_main="$(git -C "$_root" worktree list --porcelain | awk '/^worktree /{print $2; exit}')"
_sdk=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
[ -d "$_sdk" ] && export SDKROOT="$_sdk"
export PATH="$_main/.venv/bin:$PATH"
unset _root _main _sdk
