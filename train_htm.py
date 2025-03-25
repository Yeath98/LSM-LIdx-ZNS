import numpy as np
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.algorithms.anomaly import Anomaly
import pickle

# 加载 SST 块的最小和最大 key 以及它们的位置
keys = np.load('sst_keys.npy')

# 提取输入特征和输出标签
inputs = keys[:, :2]  # 最小和最大 key
outputs = keys[:, 2:] # level 和 order

# 将输入特征归一化到 [0, 1] 范围
input_min = np.min(inputs, axis=0)
input_max = np.max(inputs, axis=0)
inputs = (inputs - input_min) / (input_max - input_min)

# 初始化 HTM 模型
input_size = inputs.shape[1]
column_count = 2048
sparsity = 0.02

sp = SpatialPooler(
    inputDimensions=(input_size,),
    columnDimensions=(column_count,),
    potentialRadius=16,
    potentialPct=0.5,
    globalInhibition=True,
    localAreaDensity=sparsity,
    numActiveColumnsPerInhArea=-1,
    stimulusThreshold=0,
    synPermInactiveDec=0.01,
    synPermActiveInc=0.1,
    synPermConnected=0.1,
    minPctOverlapDutyCycle=0.1,
    dutyCyclePeriod=1000,
    boostStrength=0.0,
    seed=1956
)

tm = TemporalMemory(
    columnDimensions=(column_count,),
    cellsPerColumn=4,
    activationThreshold=10,
    minThreshold=10,
    maxNewSynapseCount=30,
    permanenceIncrement=0.1,
    permanenceDecrement=0.1,
    predictedSegmentDecrement=0.0,
    maxSegmentsPerCell=256,
    maxSynapsesPerSegment=32,
    seed=1956
)

anomaly = Anomaly()

# 训练 HTM 模型
for i in range(len(inputs)):
    # Spatial Pooler
    input_vector = inputs[i]
    active_columns = np.zeros(column_count, dtype=np.uint32)
    sp.compute(input_vector, True, active_columns)

    # Temporal Memory
    active_cells = np.zeros(column_count * 4, dtype=np.uint32)
    tm.compute(active_columns, True, active_cells)

    # Anomaly Detection
    anomaly_score = anomaly.anomalyScore(tm, active_cells)
    print(f"Step {i}: Anomaly Score = {anomaly_score}")

# 保存训练好的模型
with open('htm_model.pkl', 'wb') as f:
    pickle.dump((sp, tm, anomaly), f)

print("HTM 模型训练完成并保存到 htm_model.pkl")