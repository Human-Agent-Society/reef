# Git to Zig

Reimplement git in Zig as a drop-in replacement for the `git` binary. The C
source for git v2.47.0 is at `/home/workspace/git-src/` — read it, understand it, and
rewrite it. Your binary must behave identically to the real `git` — same CLI
interface, same output formats, same exit codes.

Your workspace is `/home/workspace/zig-port/` with a build scaffold that already compiles
and links zlib. `zig build` produces `zig-out/bin/git`. The system `git` is
installed — use it to test your implementation as you go.

No internet. Do not compile or link the C source or wrap around the existing git binary — write Zig.
Work autonomously, do not ask user for input.
