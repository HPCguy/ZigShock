# ZigShock
1D and 3D Zig shock physics examples

To try the 1D:
```
% zig build-exe -OReleaseFast 1D/shock.zig
% ./shock > ultra
% gnuplot
gnuplot> plot "ultra"
gnuplot> exit
%
```

These are perfect test cases to resolve the following Zig issue, since all numerics are centralized to a few lines.

https://github.com/ziglang/zig/issues/23173

==============================================

I will accept pull requests that:

(1) Make better use of Zig language features while keeping the source code approximately the same size or smaller.

** AND **

(2) Have performance that is exactly as fast or faster than the version being replaced, definitely for the ReleaseFast build of the code, and maybe also for the Debug build.

