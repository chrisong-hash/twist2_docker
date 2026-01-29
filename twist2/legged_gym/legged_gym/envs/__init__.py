# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .base.humanoid import Humanoid
from .base.humanoid_mimic import HumanoidMimic

from .g1.g1_mimic_config import G1MimicCfg, G1MimicCfgPPO
from .g1.g1_mimic import G1Mimic

# DeepMimic (for teleoperation)
from .g1.g1_mimic_distill import G1MimicDistill
from .g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO, G1MimicStuCfg, G1MimicStuCfgDAgger
from .g1.g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicStuRLCfgDAgger

# Strict termination variant for experiment
from .g1.g1_mimic_distill_config_strict import G1MimicPrivStrictCfg, G1MimicPrivStrictCfgPPO

# V3: Stability-focused reward rebalancing
from .g1.g1_mimic_distill_config_v3 import G1MimicPrivCfgV3, G1MimicPrivCfgPPOV3

# V4: Balanced penalty distribution (fix V3's action_rate dominance)
from .g1.g1_mimic_distill_config_v4 import G1MimicPrivCfgV4, G1MimicPrivCfgPPOV4

# Overfit test: Default weights on tiny dataset (diagnostic)
from .g1.g1_mimic_distill_config_overfit import G1MimicPrivCfgOverfit, G1MimicPrivCfgPPOOverfit

# V5: Longevity focused (change tracking:alive ratio from 24:1 to 7:1)
from .g1.g1_mimic_distill_config_v5 import G1MimicPrivCfgV5, G1MimicPrivCfgPPOV5

# V6: Responsive teleop (anti-stutter, remove action_rate penalty)
from .g1.g1_mimic_distill_config_v6 import G1MimicPrivCfgV6, G1MimicPrivCfgPPOV6

# V6.1: V6 params + custom weighted data (c_walk: 2.5, virtuals: 2.0, TWIST2_full: 1.0)
from .g1.g1_mimic_distill_config_v6_1 import G1MimicPrivCfgV6_1, G1MimicPrivCfgPPOV6_1

# V7: Random motion freeze (teaches stability at any pose)
from .g1.g1_mimic_distill_freeze import G1MimicDistillFreeze
from .g1.g1_mimic_distill_config_v7 import G1MimicPrivCfgV7, G1MimicPrivCfgPPOV7

from .g1.g1_mimic_future import G1MimicFuture
from .g1.g1_mimic_future_config import G1MimicStuFutureCfg, G1MimicStuFutureCfgDAgger

# V6.2 Student: 0.5s future sight + higher dagger_coef
from .g1.g1_mimic_future_config_v6_2 import G1MimicStuFutureCfgV6_2, G1MimicStuFutureCfgPPOV6_2

from legged_gym.gym_utils.task_registry import task_registry


# DeepMimic (for teleoperation)
task_registry.register("g1_mimic", G1Mimic, G1MimicCfg(), G1MimicCfgPPO())
task_registry.register("g1_stu_mimic", G1MimicDistill, G1MimicStuCfg(), G1MimicStuCfgDAgger())
task_registry.register("g1_priv_mimic", G1MimicDistill, G1MimicPrivCfg(), G1MimicPrivCfgPPO())
task_registry.register("g1_stu_rl", G1MimicDistill, G1MimicStuRLCfg(), G1MimicStuRLCfgDAgger())
task_registry.register("g1_stu_future", G1MimicFuture, G1MimicStuFutureCfg(), G1MimicStuFutureCfgDAgger())

# V6.2 Student: 0.5s future sight for real-time deployment
task_registry.register("g1_stu_future_v6_2", G1MimicFuture, G1MimicStuFutureCfgV6_2(), G1MimicStuFutureCfgPPOV6_2())

# Strict termination experiment
task_registry.register("g1_priv_mimic_strict", G1MimicDistill, G1MimicPrivStrictCfg(), G1MimicPrivStrictCfgPPO())

# V3: Stability-focused reward rebalancing
task_registry.register("g1_priv_mimic_v3", G1MimicDistill, G1MimicPrivCfgV3(), G1MimicPrivCfgPPOV3())

# V4: Balanced penalty distribution
task_registry.register("g1_priv_mimic_v4", G1MimicDistill, G1MimicPrivCfgV4(), G1MimicPrivCfgPPOV4())

# Overfit diagnostic test
task_registry.register("g1_priv_mimic_overfit", G1MimicDistill, G1MimicPrivCfgOverfit(), G1MimicPrivCfgPPOOverfit())

# V5: Longevity focused
task_registry.register("g1_priv_mimic_v5", G1MimicDistill, G1MimicPrivCfgV5(), G1MimicPrivCfgPPOV5())

# V6: Responsive teleop (anti-stutter)
task_registry.register("g1_priv_mimic_v6", G1MimicDistill, G1MimicPrivCfgV6(), G1MimicPrivCfgPPOV6())

# V6.1: V6 params + custom weighted data
task_registry.register("g1_priv_mimic_v6_1", G1MimicDistill, G1MimicPrivCfgV6_1(), G1MimicPrivCfgPPOV6_1())

# V7: Random motion freeze (uses special freeze environment)
task_registry.register("g1_priv_mimic_v7", G1MimicDistillFreeze, G1MimicPrivCfgV7(), G1MimicPrivCfgPPOV7())


