# ORCA/RVO2 runner

`orca_runner.cpp` uses the official RVO2 core in `RVO2/` and adds:
- double-integrator velocity tracking (`|a| <= 2.0`, `|v| <= 1.5`);
- random pause hold;
- endpoint hold so the robot reaches the actual final waypoint instead of
  stopping at RVO2's 0.12 m target acceptance radius.

Built with the portable MinGW toolchain:

```powershell
g++ -O2 -std=c++17 -DRVO_STATIC_DEFINE `
  -I third_party/RVO2 `
  third_party/orca_runner.cpp third_party/RVO2/*.cc `
  -o third_party/orca_runner.exe
```

The built binary is committed for reproducibility on this Windows environment;
`src/fhd/adapters/orca_adapter.py` uses `third_party/orca_runner.exe`.
