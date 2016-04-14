#pragma once
#include "SpatiallySparseLayer.h"

void applyBatchNormalization(SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output, BatchNormalizationMode mode,
                  cudaMemStream &memStream);

void applyBatchNormalizationBackProp(SpatiallySparseBatchInterface &input,
                          SpatiallySparseBatchInterface &output,
                          BatchNormalizationMode mode, cudaMemStream &memStream);

class BatchNormalizationLayer : public SpatiallySparseLayer {

public:
  BatchNormalizationMode mode;

  BatchNormalizationLayer(cudaMemStream &memStream, BatchNormalizationMode mode);

  void preprocess(SpatiallySparseBatch &batch,
                  SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output);

  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);

  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output, float learningRate,
                 float momentum);

  int calculateInputSpatialSize(int outputSpatialSize);
};
