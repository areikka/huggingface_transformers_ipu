# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import poptorch
from poptorch.enums import AnchorMode
import popart
import numpy as np
import ctypes
import os
import popdist
import popdist.poptorch
import horovod.torch as hvd

def init_popdist(args):
    hvd.init()
    args.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replication_factor:
        print(f"The number of replicas is overridden by PopRun. "
              f"The new value is {popdist.getNumTotalReplicas()}.")
    args.replication_factor = int(popdist.getNumLocalReplicas())
    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()


def get_options(config, train=True):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''
    # Numpy options
    np.random.seed(config.seed)

    # Initialise PopDist
    if popdist.isPopdistEnvSet():
        init_popdist(config)
        opts = popdist.poptorch.Options(ipus_per_replica=4)
        opts.replication_factor = config.replication_factor 
        opts.popdist_rank = config.popdist_rank
        opts.popdist_size = config.popdist_size
        opts.use_popdist = config.use_popdist
    else:
        # Poptorch options
        opts = poptorch.Options()
        opts.use_popdist = False
        opts.popdist_rank = -1

    opts.autoRoundNumIPUs(True)

    # if config.compile_only:
    #     opts.connectionType(poptorch.ConnectionType.Never)
    #     opts.useOfflineIpuTarget(ipu_version=2)

    if train:
        opts.deviceIterations(int(config.batches_per_step))
        if not opts.use_popdist:
            opts.replicationFactor(int(config.replication_factor))
        opts.Training.gradientAccumulation(int(config.gradient_accumulation))
        opts.Training.accumulationAndReplicationReductionType(
            poptorch.ReductionType.Mean)
        if config.anchormode == 'final':
            anchorMode = poptorch.AnchorMode.Final
        else:
            anchorMode = poptorch.AnchorMode.All
        opts.anchorMode(anchorMode)
        opts.Precision.enableStochasticRounding(not config.OffSR)
    else:
        opts.deviceIterations(config.InferencedeviceIteration)
        if not opts.use_popdist:
            opts.replicationFactor(config.InferencereplicationFactor)
        opts.anchorMode(poptorch.AnchorMode.All)
    opts.randomSeed(config.seed)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    # import pdb
    # pdb.set_trace()
    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)
    # PopART options
    if train:
        opts._Popart.set("disableGradAccumulationTensorStreams", True)
        opts._Popart.set("enableGradientAccumulation", True)
    opts._Popart.set("enableOutlining", True)
    opts._Popart.set("outlineThreshold", 10.0)
    if train:
        opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                         int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
        opts._Popart.set(
            "accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])

    if config.enable_half_partials:
        opts.Popart.set("partialsTypeMatMuls", "half")
        opts.Popart.set("convolutionOptions", {'partialsType': "half"})

    if config.synthetic_data:
        opts.Popart.set("syntheticDataMode", int(
            popart.SyntheticDataMode.RandomNormal))

    engine_options = {"target.syncReplicasIndependently": "true"}

    opts._Popart.set("engineOptions", engine_options)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(not config.OffChipStorage))
    opts.TensorLocations.setWeightLocation(
        poptorch.TensorLocationSettings().useReplicatedTensorSharding(not config.OffRTS))
    opts._Popart.set("enableFloatingPointChecks",True)
    return opts
