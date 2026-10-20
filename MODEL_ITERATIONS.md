# 网络设计迭代记录

> 最后更新: **2026-10-13**。每训一个新版本、每有真机/探针结论, 在这里追加一行。
> 论文主张以 `PAPER_CHARTER.md` 为准; 数据台账见 `DATA_INVENTORY.md`。
> checkpoint 路径约定: 集群 `~/ypf/Pi0.5_reproduction/checkpoints/<config>/<exp>/<step>`,
> 本地 `~/openpi/checkpoints/` 同结构。

## 总表

| 日期 | 版本(exp) | config | 数据 | 改了什么 | 结果 | commit |
|---|---|---|---|---|---|---|
| 04-19 ~ 06-15 | 基线 | (已删) | 单任务 | pi0.5 + LoRA 微调; 关节空间 → W′ 相对末端位姿(每臂 10D = pos3 + rot6d + grip1) | 管线跑通 | a77c8f9, 60e31c9 |
| 06-17 ~ 07-01 | 全局夹爪 token | pi05_cotrain_eef_grip(_aug)(已删) | 单臂 co-train | 每把爪点云 → 1 个几何 token 进 prefix; 每爪 TCP 偏移 + 开合范围标定(位姿锚在指尖); 点云增强 | 可部署 | 32e0115, 3593f62, 74c2150 |
| 07-09 | 双臂 20D | pi05_cotrain_dualarm(已删) | 双臂 | 两臂状态拼成 20D; 每臂一个 token; 单臂样本 arm1_mask=0 + 左腕黑图 | 双臂任务可训 | 5b4e605 |
| 07-23 | region_v1 / v2 | pi05_cotrain_dualarm_region | cotrain_dualarm2 | 点云按 tip/mid/rear 分 3 个区域 token(共享 PointNet + 零初始化区域嵌入) | **被注意力无视**: 离线探针 swap/全零/随机点云 ≈1.0× 噪声地板; NaN 灌注证明数据进了图 | 71532cc |
| 07-28 | film_v1 | pi05_cotrain_dualarm_film | cotrain_dualarm3 | 区域特征经零初始化投影加到动作专家 adaRMS 条件(FiLM) | **被主动压制**: 参数范数镜像下降, 探针 1.3×@10k → 1.1×@34k; 但分布内挂杯成功率最好 | c315547 |
| 07-29 | film_drop_v1 | pi05_cotrain_dualarm_film_drop | cotrain_dualarm3 | 互斥相机遮断(无全盲样本) | 10k 探针无效(环境相机仍能认爪) | 67f9cec |
| 07-29 | film_drop_v2 | pi05_cotrain_dualarm_film_drop | cotrain_dualarm3 | 遮断加全盲: 45% 只留腕 / 15% 只留环境 / 10% 全盲 | **通道被激活**: 探针 1.5×@10k → 2.4×@34k; 真机 A/B/A/B(装平行爪喂 GET 点云下探深 ~21mm = 示教差; film_v1 无此现象)。代价: 分布内精度下降 | 4f647f1 |
| 07-30 | armframe_v1 | pi05_cotrain_dualarm_armframe | cotrain_dualarm4 | **去掉 TCP 标定**: 位姿锚在机械臂报点, GET 指尖多伸的 17mm 只在点云里; 点云对齐到臂参考系并在法兰处裁剪; 保尺度增强 | 区域乱序成为探针最大效应(结构被读取); 视觉剥夺剂量-反应 swap 2.4→3.3→6.1× | ee9f04d |
| 08-20 | **armframe_v2** | pi05_cotrain_dualarm_armframe | cotrain_dualarm4(600) | 遮断降到 30/10/8; 34k 步 | 四个演示任务(积木/挂杯/扫地/切泥)的备份权重; 本地唯一现存权重 | 733a9dc |
| 08-29 | effort_v1 | pi05_cotrain_dualarm_effort | cotrain_dualarm5(700) | 力感知 Level A: 与 action_out_proj 并列的 effort_head, 预测未来 10 步 14D 原始电机力矩(臂0 6+夹爪1, 臂1 6+夹爪1), 掩码 MSE 权重 0.05; **只是训练辅助目标, 推理不用, 不是输入**; 40k 步 | 真机: 演示四任务不抽动但主观成功率下降 → 回退; 抽屉任务有"跳一下"式抽动(见下) | 9878296 |
| 08-30 | armframe_v3 | pi05_cotrain_dualarm_armframe | cotrain_dualarm5(700) | 与 effort_v1 同数据同 40k 步, **只去掉力矩头**(effort_dim=0); 训练改为每 2000 步滚动存档 | **结果待记录** | 0661626, e86e01b, 9a439bb |
| 10-12 | sweep_single_v1 | pi05_single_parallel_sweep | single_parallel_sweep(50) | 单任务/单爪对照: 只用平行爪扫地 50 条(双臂, 两爪皆 parallel), **无相机遮断**, 其余同 armframe; 20k 步, 每 2k 存档, 只留最新档 | **训练中**: 10-12 第一次跑到 16k 时家目录写满(ENOSPC)崩溃, 从 12000 续; 用途: 与 750 条共训权重在同一任务上比成功率 | (待提交) |
| 10-13 | cotrain_nodrop_v1 | pi05_cotrain_dualarm_nodrop | cotrain_dualarm5(750, 含 GET 抽屉) | 全量数据、**无相机遮断**, 其余同 armframe; 40k 步 | **已训完**(2026-10-20, 39999); 20000 档曾用于 10-14 演示; 用 train_fast_ultra.sh 见缝插针(1/2 卡 H200/H100, 锁文件协调), 中途修了跨卡数续档 | (待提交) |
| 10-20 | wipe_single_v1 | pi05_single_parallel_wipe | single_parallel_wipe(50) | 单任务/单爪对照(平行爪擦黑板), 同 sweep_single_v1 配方 | **待训**(train_single_pair.sh 先后跑两个) | (待提交) |
| 10-20 | mug_single_v1 | pi05_single_parallel_mug | single_parallel_mug(50) | 单任务/单爪对照(平行爪挂杯), 同上 | **待训** | (待提交) |

## 贯穿的机制结论(论文脊柱)

条件化通道被使用的程度取决于它是否是该信息的**唯一来源**; 改架构改不动, 改信息结构才能:
token 注入被无视 → FiLM 被压制 → 相机遮断(含全盲)激活通道 → 去标定后区域结构被读取。
根因: 每爪 TCP/开合标定在接口层抹平了两爪差异, 且视觉本就能认爪, 几何通道没有残差可学。
**交付/演示用无遮断或轻遮断权重, 机制研究用强遮断权重。**

## 待归因: effort_v1 成功率下降

effort_v1 相对 armframe_v2 同时改了三样: 力矩 loss、+100 条数据(parallel 抽屉/扫地)、34k→40k 步。
armframe_v3 是唯一变量对照(只差力矩头), 在相同任务上对比:
- v3 恢复 → 力矩 loss 拖累动作 → 下一轮降权到 0.01–0.02, 或只监督夹爪 2 维力矩;
- v3 同样差 → 数据混合或步数问题, 力矩头无辜。

## 已知部署问题

- **抽屉任务"跳一下"式抽动**(effort_v1, 平行爪): 推理延迟稳定 ~190ms、无相机超时, 已排除;
  `--chunk 5` 无改善。**2026-09-13 示教数据实测定位**: 抽屉在 J4 +90° 分支下 J4 全程 [+60°, +172°],
  31/50 条顶到 J4 上限; J4/J6 是手腕滚转关节, 限位附近笛卡尔 IK 可能换解(J4−180°, J5 取反, J6+180°)
  → 关节猛跳。其他任务 J4 都在 ±150° 内, 与"只有抽屉抽动"吻合。**处理**: 协议改 J4 −90° 分支
  (50 条全部留在行程内, 手腕余量中位 23°); 两分支末端位姿严格相同(URDF FK 差 0.002mm), 已采数据有效。
  真机验证待做: −90° 开局跑抽屉 rollout + `--log-joints`。详见 `airbot/TASK_PROTOCOL.md` "J4 分支与关节限位"。
- **关节守卫**(`airbot/joint_guard.py`, 2026-09-13): 采集脚本默认开启, 推理 `--log-joints` 开启。
  J4/J6 提示 <15°、危险 <5°; J1/J5 <10°/<3°; J2/J3 顶止挡只进小结(常规任务频繁顶 J2, 无害);
  "顶住"= 已在限位区且 lead−follow 偏差 >12° 持续 0.3s。阈值依据 700 条示教的余量/跟随误差统计。
- 同一次日志里策略在抽屉任务上 30 秒无推进(夹爪在抓/松之间循环), 与抽动是两个问题。
- 带 norm_stats 额外键(effort)的权重: `policy_config.py` 已在输出反归一化中跳过辅助目标键。
- 旧权重配新 config: config 的 repo_id 变了时, 要把 checkpoint 里 `assets/<旧repo>` 复制成
  `assets/<新repo>`(例: armframe_v2 需 `cp -r cotrain_dualarm4 cotrain_dualarm5`)。

## 现存 config(`src/openpi/training/config.py`)

消融梯: `_vanilla`(无点云) / `_region` / `_film` / `_film_drop` / `_armframe` / `_effort`。
