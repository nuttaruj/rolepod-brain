Test fixtures for the updater (src/update.rs, tests/update_e2e.rs).

- `test.key` / `test.pub`: a throwaway minisign key pair, passwordless. A release build never reads it; only a debug build takes it, through `ROLEPOD_BRAIN_UPDATE_PUBKEY`.
- `bin-*`: shell scripts standing in for `brain` (they answer `--version` and `self-test`).
- `<name>.<target>.minisig`: signatures over them, one per unix target, trusted comment `brain-<target> <version>`.

Rebuild with `./make.sh` (needs `brew install minisign`; no test calls minisign).
