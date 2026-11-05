# 00 现有成果冻结与可恢复性报告

**冻结时间**：2026-11-05 23:20–23:22（Asia/Shanghai）  
**仓库**：https://github.com/xieyuanZZZ/No02  
**冻结分支**：`experiment/online-temporal-evaluation-v2`  
**冻结提交**：`7019d54208c0892a1132a9fddd69af421ef8aa38`  
**冻结 Tag**：`freeze-online-temporal-evaluation-v2-20261105`  
**Tag 对象**：`bbfc44d81ffce74f666bc99156f67a2a810e06b8`（已推送到 `origin`）

---

## 1. 开工核对

```text
git fetch origin --prune --tags
远端 experiment/online-temporal-evaluation-v2 = 7019d54
本地 HEAD = 7019d54
工作区无未提交修改
```

未发现远端更新提交，未做任何回退或历史重写。

---

## 2. 冻结 Tag

```powershell
git tag -a freeze-online-temporal-evaluation-v2-20261105 `
  -m "Freeze: online temporal evaluation v2, 20-task scale results, commit 7019d54" `
  7019d54208c0892a1132a9fddd69af421ef8aa38
git push origin freeze-online-temporal-evaluation-v2-20261105
```

远端校验：

```text
bbfc44d81ffce74f666bc99156f67a2a810e06b8  refs/tags/freeze-online-temporal-evaluation-v2-20261105
```

---

## 3. 仓库外完整备份

**备份根目录**：

```text
D:\No02_freezes\No02_online_temporal_v2_20261105
```

| 内容 | 路径/大小 | 说明 |
|---|---|---|
| Git 全历史 | `repo_git_all.bundle`，71.2 MB | `git bundle create --all`，含所有分支与冻结 Tag |
| 工作树 | `worktree/`，约 6.243 GB / 2,173 文件 | 代码、配置、数据、results（含 Diffusion checkpoints）、CSV/JSON、图、文档 |
| 冻结说明 | `FREEZE_INFO.txt` | commit/tag/备份范围 |
| SHA256 清单 | `SHA256_MANIFEST.txt`，2,175 行 | bundle + FREEZE_INFO + 2,173 个 worktree 文件；UTF-8 保留中文文件名 |
| 恢复验证 | `RESTORE_TEST/` | 从 bundle clone + 从 worktree 恢复 + 哈希核验，不参与清单 |

**排除项**（不属于需备份的实验产物/虚拟环境）：

```text
.git（由 bundle 保护）
.venv
.pytest_cache
torchinductor_xuan
pytest-of-xuan
__pycache__
.mypy_cache
node_modules
```

**关键 SHA256**：

```text
repo_git_all.bundle
7d5f904a5c4e17bb76ef488f19f141a783be809cb3ba6cb1ee27160f6588c708

FREEZE_INFO.txt
96d8c5ddf12e72c865ae9c3fe4d7ebebd8cf019b3ab75084bf193ff65df917b9
```

---

## 4. 恢复验证记录

验证脚本与结果保存在：

```text
D:\No02_freezes\No02_online_temporal_v2_20261105\RESTORE_TEST\report.txt
```

实际执行：

1. `git bundle verify`：**complete history**，Tag 存在，exit 0；
2. `git clone repo_git_all.bundle RESTORE_TEST/repo_from_bundle`：exit 0；
3. 恢复 clone 的 `HEAD = 7019d54208c0892a1132a9fddd69af421ef8aa38`，Tag 存在；
4. `robocopy worktree -> RESTORE_TEST/worktree_restored`：2,173 文件，0 失败；
5. 按 UTF-8 `SHA256_MANIFEST.txt` 逐文件核对：**2,174 个哈希检查，0 不匹配**，
   `FREEZE_INFO` 单独核对相等；
6. 在恢复后的 worktree 中运行：

```powershell
$env:PYTHONPATH="...\RESTORE_TEST\worktree_restored\src"
python -m pytest tests\test_online_baseline.py -q -p no:cacheprovider
```

结果：**9 passed**，exit 0；导入路径确认来自恢复目录，不是原工作区。

> 说明：首次在恢复目录运行 pytest 时，Windows 临时目录环境不可用导致 pytest 启动失败；
> 设置 `TEMP/TMP=D:\paper2\pytest-of-xuan` 后 9 项全部通过。这是测试环境问题，
> 不是恢复内容损坏；恢复文件的 SHA256 已全部一致。

---

## 5. 新工作分支与 worktree

```text
worktree: D:\No02_external_baseline_v1
branch:   experiment/external-baseline-comparison-v1
base:     7019d54208c0892a1132a9fddd69af421ef8aa38
```

该 worktree 只用于新增外部基线适配、评测脚本和新结果；不修改冻结分支与 Tag。

---

## 6. 只读冻结范围

- First-Hit 风险算法与在线协调实现；
- 离线整数搜索等待优化；
- MPC 控制与执行仿真核心；
- Diffusion 模型与权重；
- 冻结路径/任务数据；
- `results/Online_Temporal/`、`results/Online_Temporal_v2/` 及对应报告。

后续如需修 bug，必须先证明冻结版本有误，并同时保留旧结果、记录差异；不得为了让某个基线
或本方法好看而改冻结代码/参数/评价口径。

---

## 7. 结论

- Tag 已创建并推送；
- Git 全历史有 bundle；
- 被 `.gitignore` 忽略的模型权重/大结果已通过 worktree 镜像备份；
- SHA256 清单完整且可校验；
- 已实际从备份恢复并运行测试；
- **冻结阶段通过**，可以进入外部基线复现。
