# 数据采集台账(两把爪 × 任务)

> 统计时间: **2026-10-04**。数据来源: `~/pi_data/<爪>/<任务>_task/<目录>/meta/info.json`
> 和打包账本 `~/pi_data/_pack_ledger/<repo>/pack_manifest.json`。
> 每次采完新数据、打包追加后更新本文件(条数/小时/日期)。任务协议细节见 `airbot/TASK_PROTOCOL.md`。

## 总量

| 爪 | 条数 | 小时 |
|---|---|---|
| GET | 450 | 7.05 |
| parallel(平行爪) | 300 | 4.24 |
| **合计** | **750** | **11.3** |

最后一次采集: 2026-09-13(GET 抽屉)。

## 按任务汇总

| 任务 | prompt(必须逐字一致) | 臂 | GET | parallel | 电流遥测 |
|---|---|---|---|---|---|
| 积木入碗 | `pick up blocks and place them in the bowl` | 右臂单臂 | 50 | 50 | 无 |
| 堆金字塔 | `stack the bricks into a pyramid` | 右臂单臂 | 50 | 50 | 无 |
| 擦黑板 | `wipe the blackboard` | 右臂单臂 | 50 | 50 | 无 |
| 挂马克杯 | `hang the mug on the mug tree` | 右臂单臂 | 50 | 50 | 有 |
| 扫物入铲 | `sweep the objects into the dustpan` | 双臂 | 50 | 50 | 有 |
| 抽屉分类整理 | `put the blocks in the top drawer and the other objects in the third drawer` | 右臂单臂 | 50 | 50 | 有 |
| 切橡皮泥(新握法) | `cut the play-dough with the knife` | 双臂 | 50 | 不采(评测项) | 有 |
| 切橡皮泥(旧换手, 已停采) | `regrasp the knife and cut the play-dough` | 双臂 | 100 | — | 无 |

说明:
- **切泥只采 GET 是有意的**: `PAPER_CHARTER.md` 把平行爪切泥定为"能力边界评测项", 不进训练集
  (`TASK_PROTOCOL.md` 第 16 行"平行爪同协议采集"与此矛盾, 以纲领为准)。
- 旧换手握法 100 条 prompt 不同, 仅作消融。
- **双臂爪搭配**: 切泥 = 臂0(右)GET 持刀 + 臂1(左)parallel 按压;
  扫地 GET 版 = 臂0 GET 持扫帚 + 臂1 parallel 持铲; 扫地 parallel 版 = 右臂也换平行爪持扫帚。
- **平行爪抽屉 50 条是在 J4 +90° 分支下采的**, 其中 31 条顶到 J4 上限。协议已改为 −90°;
  **GET 抽屉 50 条(2026-09-13)是按 −90° 采的**: J4 全程 [−97°, +4°], 没有一条离限位小于 5°。
  两分支末端位姿严格相同, 这 50 条仍是有效训练数据。见 `airbot/TASK_PROTOCOL.md` "J4 分支与关节限位"。
- **电流遥测**(100Hz sidecar: 关节力矩 + 夹爪力矩)从挂杯任务开始才有。积木/金字塔/擦黑板
  两爪共 300 条、加上旧握法切泥 100 条, 合计 400 条无遥测, 训练力矩头时被 `effort_mask` 掩掉;
  有遥测的是另外 350 条。

## 按采集目录明细

| 爪 | 任务 | 目录 | 条 | 小时 | 打包日期 |
|---|---|---|---|---|---|
| GET | 擦黑板 | get_wipe | 50 | 0.53 | 2026-06-28 |
| GET | 切泥(旧) | cut_dough_get | 20 | 0.51 | 2026-07-03 |
| GET | 切泥(旧) | cut_dough_get_20260708_040303 | 9 | 0.21 | 2026-07-08 |
| GET | 切泥(旧) | cut_dough_get_20260708_051752 | 9 | 0.17 | 2026-07-08 |
| GET | 切泥(旧) | cut_dough_get_20260708_065150 | 12 | 0.20 | 2026-07-08 |
| GET | 切泥(旧) | cut_dough_get_20260709_052608 | 30 | 0.50 | 2026-07-09 |
| GET | 切泥(旧) | cut_dough_get_20260709_165649 | 20 | 0.39 | 2026-07-09 |
| GET | 切泥(新) | cut_dough_get_v2 | 18 | 0.31 | 2026-07-22 |
| GET | 切泥(新) | cut_dough_get_v2_20260722_230910 | 32 | 0.53 | 2026-07-23 |
| GET | 挂杯 | get_mug_tree | 18 | 0.22 | 2026-07-26 |
| GET | 挂杯 | get_mug_tree_20260726_182735 | 32 | 0.38 | 2026-07-26 |
| GET | 积木 | get_block | 50 | 0.49 | 2026-07-01 |
| GET | 金字塔 | get_pyramid | 50 | 1.03 | 2026-06-25 |
| GET | 扫地 | sweep_objects | 50 | 0.62 | 2026-08-20 |
| GET | 抽屉 | get_drawer | 50 | 0.97 | 2026-09-13 |
| parallel | 擦黑板 | airbot_wipe_data | 27 | 0.28 | 2026-06-15 |
| parallel | 擦黑板 | airbot_wipe_data_20260616_171940 | 23 | 0.20 | 2026-06-16 |
| parallel | 抽屉 | parallel_drawer | 7 | 0.17 | 2026-08-28 |
| parallel | 抽屉 | parallel_drawer_20260828_163602 | 30 | 0.62 | 2026-08-28 |
| parallel | 抽屉 | parallel_drawer_20260829_012318 | 13 | 0.29 | 2026-08-29 |
| parallel | 挂杯 | parallel_mug_tree | 50 | 0.70 | 2026-07-28 |
| parallel | 积木 | airbot_block_data | 50 | 0.47 | 2026-06-14 |
| parallel | 金字塔 | airbot_pyramid_data_20260616_180605 | 50 | 0.90 | 2026-06-17 |
| parallel | 扫地 | sweep_objects | 8 | 0.11 | 2026-08-23 |
| parallel | 扫地 | sweep_objects_20260824_003716 | 42 | 0.49 | 2026-08-24 |

## 打包数据集(LeRobot repo)与用途

| repo | 条数 | 内容 | 用于训练 |
|---|---|---|---|
| cotrain_dualarm2 | — | 早期(无挂杯/扫地/抽屉) | region_v1/v2 |
| cotrain_dualarm3 | — | + 挂杯 | film_v1, film_drop_v1/v2 |
| cotrain_dualarm4 | 600 | + GET 扫地; arm-frame 位姿 | armframe_v1, armframe_v2 |
| cotrain_dualarm5 | 700 | + parallel 扫地 + parallel 抽屉; arm-frame; 含 14D effort 列 | effort_v1, armframe_v3 |

**GET 抽屉 50 条尚未打包**: 截至 2026-10-04 账本里 `cotrain_dualarm5` 仍是 700 条。
`cotrain_dualarm5` 有打包账本, 新数据用 `--arm-frame --append --repo-id cotrain_dualarm5` 增量追加即可
(`--repo-id` 必须写, 脚本靠它定位账本);
**数据变了必须在集群重算 norm_stats, 不能复制旧的**。

## 缺口(按优先级, 对应 PAPER_CHARTER 里程碑)

1. **把 GET 抽屉 50 条追加进 `cotrain_dualarm5`**(已采未打包) → 上传 → 重算 norm_stats → 重训。
   至此六个任务两把爪各 50 条, 跨爪配对已齐。
2. 采集脚本加深度/内参/rig.json 录制(带开关) → 第 3、4 把爪入库采集。
3. 以 mid/rear 区域为主接触部位的任务(推物/按压等), 支柱 2 证据最薄。
4. 第 5 把爪(保留爪)零样本/少样本阶梯 0→10→50 条。
