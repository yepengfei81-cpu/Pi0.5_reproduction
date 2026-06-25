"""每把夹爪的部署/打包参数 —— 测完只改这一个文件, collect/pack/deploy 都读它。

字段:
  close / open : 该爪装在机械臂上时 get_eef_pos 的"刚好闭合 / 完全张开"读数。
                 夹爪归一化用 g01=(g-close)/(open-close); 避免对 GET 过挤压。
  tcp_offset   : 该爪指尖相对 get_end_pose 报点的偏移(EE 系, 米), pivot 标定得到。
                 parallel=0(运动学本就标定到平行爪指尖); 用来把 W' 锚到指尖、跨爪统一。

命名/顺序须与 grippers.npz 的 names、config.gripper_names 一致。
新增爪: 在此加一行 + 加进 grippers.npz + config.gripper_names。
"""

GRIPPER_PARAMS = {
    "parallel": {"close": 0.0, "open": 0.073, "tcp_offset": (0.0, 0.0, 0.0)},
    # GET: 指尖比平行爪沿 +X(指向)多伸 17mm(安装面->指尖: GET 90.5mm vs 平行 73.5mm, 卡尺)。
    #      close/open = teach_probe 读到的 get_eef_pos: 刚闭合 8.9mm / 全开 70.1mm。
    #      (物理全开宽度卡尺量得 62mm, 仅记录; 归一化用 get_eef_pos 读数。)
    "get":      {"close": 0.0089, "open": 0.0701, "tcp_offset": (0.017, 0.0, 0.0)},
}


def get_params(name: str) -> dict:
    if name not in GRIPPER_PARAMS:
        raise KeyError(f"未知夹爪 '{name}', 请在 gripper_params.py 的 GRIPPER_PARAMS 里加上")
    return GRIPPER_PARAMS[name]
