#!/bin/sh
# Rebuilds every fixture here. Needs `minisign` (brew install minisign); no test calls it.
# The key pair is a TEST key: only a debug build accepts it (ROLEPOD_BRAIN_UPDATE_PUBKEY).
set -eu
cd "$(dirname "$0")"
rm -f test.key test.pub *.minisig bin-*
minisign -G -f -W -p test.pub -s test.key >/dev/null
script() { printf '#!/bin/sh\nif [ "${1:-}" = "--version" ]; then echo "brain %s"; exit 0; fi\nif [ "${1:-}" = "self-test" ]; then exit %s; fi\nexit 0\n' "$2" "$3" > "bin-$1"; }
script good 9.0.0 0
script old 0.0.1 0
script badver 9.0.0 0
script stfail 9.2.0 1
TARGETS="aarch64-apple-darwin x86_64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu"
sign() { minisign -S -s test.key -x "$1.$2.minisig" -t "brain-$2 $3" -m "bin-$1" >/dev/null; }
for t in $TARGETS; do
  sign good "$t" 9.0.0
  sign old "$t" 0.0.1
  sign badver "$t" 9.1.0
  sign stfail "$t" 9.2.0
  # the good bytes, validly signed, but for another target / another version
  other=x86_64-unknown-linux-gnu; [ "$t" = "$other" ] && other=aarch64-apple-darwin
  minisign -S -s test.key -x "wrongtarget.$t.minisig" -t "brain-$other 9.0.0" -m bin-good >/dev/null
  minisign -S -s test.key -x "wrongversion.$t.minisig" -t "brain-$t 9.0.1" -m bin-good >/dev/null
done
