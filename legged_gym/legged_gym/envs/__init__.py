# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal_c_config import AnymalCFlatCfg, AnymalCFlatCfgPPO, AnymalCRoughCfg, AnymalCRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .g1.g1_stand_config import G1StandCfg, G1StandCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

# --- Original legged_gym task registry ---
task_registry.register("anymal_c_flat", LeggedRobot, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
task_registry.register("anymal_c_rough", LeggedRobot, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
task_registry.register("a1_rough", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("cassie_rough", LeggedRobot, CassieRoughCfg(), CassieRoughCfgPPO())

# --- Incremental: G1 stand task ---
task_registry.register("g1_stand", LeggedRobot, G1StandCfg(), G1StandCfgPPO())
