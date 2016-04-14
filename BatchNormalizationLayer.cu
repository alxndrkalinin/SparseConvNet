#include <iostream>
#include "BatchNormalizationLayer.h"
#include "utilities.h"

__global__ void dBn(float *a, float *b, float *batchMean, float *batchVar, int nOut) {
  int i = blockIdx.x * nOut;
  int k = threadIdx.x;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE, k += KERNELBLOCKSIZE) {
	b[j] = (a[j] - batchMean[k]) / batchVar[k];
  }
}

void bn(float *a, float *b, float batchMean, float batchVar, int count, int nOut,
                     cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dBn << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, batchMean, batchVar, nOut);
    processed += batch;
  }
  cudaCheckError();
}

__global__ void dBnBackprop(float *a, float *b, float *da,
                                         float *db, float batchMean, float batchVar, int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
	  // TODO: finish backwards pass
	  // https://github.com/BVLC/caffe/blob/b590f1d27eb5cbd9bc7b9157d447706407c68682/src/caffe/layers/batch_norm_layer.cu#L111

//    da[j] = db[j] * b[j] * (1 - b[j]);
  }
}

void bnBackprop(float *a, float *b, float *da, float *db, float batchMean, float batchVar,
                             int count, int nOut, cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dBnBackprop << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, da + processed * nOut,
         db + processed * nOut, batchMean, batchVar, nOut);
    processed += batch;
  }
  cudaCheckError();
}

////////////////////////////////////////////////////////////////////////////////
void applyBatchNormalization(SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output, BatchNormalizationMode mode,
                  cudaMemStream &memStream) {
  switch (mode) {
  case GLOBAL:
    bn(input.sub->features.dPtr(), output.sub->features.dPtr(),
    				input.batchMean, input.batchVar,
                    output.nSpatialSites, output.featuresPresent.size(),
                    memStream);
    break;
    // TODO: implement running BN mode
//  case RUNNING:
//    bn(input.sub->features.dPtr(), output.sub->features.dPtr(),
//                output.nSpatialSites, output.featuresPresent.size(), memStream);
//    break;
  case NOBN:
    break;
  }
}

void applyBatchNormalizationBackProp(SpatiallySparseBatchInterface &input,
                          SpatiallySparseBatchInterface &output,
                          BatchNormalizationMode mode, cudaMemStream &memStream) {
  switch (mode) {
  case GLOBAL:
    bnBackprop(
        input.sub->features.dPtr(), output.sub->features.dPtr(),
        input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
        input.batchMean, input.batchVar,
        output.nSpatialSites, output.featuresPresent.size(), memStream);
    break;
    // TODO: implement running BN mode
//  case RUNNING:
//    bnBackprop(input.sub->features.dPtr(), output.sub->features.dPtr(),
//                        input.sub->dfeatures.dPtr(),
//                        output.sub->dfeatures.dPtr(), output.nSpatialSites,
//                        output.featuresPresent.size(), memStream);
//    break;
  case NOBN:
    break;
  }
}

BatchNormalizationLayer::BatchNormalizationLayer(cudaMemStream &memStream, BatchNormalizationMode mode)
    : SpatiallySparseLayer(memStream), mode(mode) {
  std::cout << BatchNormalizationMode[mode] << std::endl;
};

void BatchNormalizationLayer::preprocess(SpatiallySparseBatch &batch,
                              SpatiallySparseBatchInterface &input,
                              SpatiallySparseBatchInterface &output) {
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = input.spatialSize;
  output.nSpatialSites = input.nSpatialSites;
  output.grids = input.grids;
  output.backpropErrors = input.backpropErrors;
}

void BatchNormalizationLayer::forwards(SpatiallySparseBatch &batch,
                            SpatiallySparseBatchInterface &input,
                            SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  applyBatchNormalization(input, output, mode, memStream);
}

void BatchNormalizationLayer::backwards(SpatiallySparseBatch &batch,
                             SpatiallySparseBatchInterface &input,
                             SpatiallySparseBatchInterface &output,
                             float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    applyBatchNormalizationBackProp(input, output, mode, memStream);
  }
}

int BatchNormalizationLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}
