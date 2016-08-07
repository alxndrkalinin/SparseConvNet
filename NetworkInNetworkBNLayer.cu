#include "NetworkInNetworkBNLayer.h"
#include "utilities.h"
#include <iostream>
#include <cassert>

const float eps = 1e-6;
const float rMomentum = 0.9;

// kernel for BN, runs per-feature
__global__ void dBN(float *features, float *gamma, float *beta, float *runningMean, float *runningVar,
		float *mu, float *var, int nRows, int nColumns) {
  // each feature index
  int c = blockIdx.x * KERNELBLOCKSIZE + threadIdx.x;

  // step 1 - calculate mean
  float m = 0;
  // for sites of this feature
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    m += f;
  }
  m /= nRows;
  mu[c] = m;

  // step 2 - calculate variance
  float v = 0;
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    v += (f - m);
  }
  v /= nRows;
  var[c] = v;

  // step 3 - calculate output
  float invvar = 1.f / (sqrt(v + eps));
  float g = gamma[c];
  float b = beta[c];
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    features[i * nColumns + c] = g * (f - m) * invvar + b;
  }

  // update running values
  float rMu = runningMean[c];
  float rVar = runningVar[c];
  runningMean[c] = rMomentum * rMu + (1.0f - rMomentum) * m;
  runningVar[c] = rMomentum * rVar + (1.0f - rMomentum) * v;
}

// kernel for BN, runs per-feature
__global__ void dBNTest(float *features, float *gamma, float *beta, float *runningMean, float *runningVar,
		float *mu, float *var, int nRows, int nColumns) {
  // each feature index
  int c = blockIdx.x * KERNELBLOCKSIZE + threadIdx.x;

  float m = runningMean[c];
  float v = runningVar[c];
  float g = gamma[c];
  float b = beta[c];
  mu[c] = m;
  var[c] = v;
  // for sites of this feature
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    features[i * nColumns + c] = g * ((f - m) / sqrt(v + eps)) + b;
  }
}

void BN(float *matrix, float *gamma, float *beta, float *runningMean, float *runningVar,
		float *mu, float *var, batchType type, int nRows, int nColumns, cudaMemStream &memStream) {

	if(type == TRAINBATCH) {
		assert(nColumns % KERNELBLOCKSIZE == 0);
		dBN<<< nColumns / KERNELBLOCKSIZE, KERNELBLOCKSIZE, 0, memStream.stream>>>
				(matrix, gamma, beta, runningMean, runningVar, mu, var, nRows, nColumns);
		cudaCheckError();
	} else {
		assert(nColumns % KERNELBLOCKSIZE == 0);
		dBNTest<<< nColumns / KERNELBLOCKSIZE, KERNELBLOCKSIZE, 0, memStream.stream>>>
				(matrix, gamma, beta, runningMean, runningVar, mu, var, nRows, nColumns);
		cudaCheckError();
	}
}

__global__ void dBNBackprop(float *features, float *dfeatures, float *gamma, float *dgamma, float *beta, float *dbeta,
		float *mu, float *var, int nRows, int nColumns) {
	// each feature index
  int c = blockIdx.x * KERNELBLOCKSIZE + threadIdx.x;

  float m = mu[c];
  float v = var[c];
  float g = gamma[c];

  // step 1 - calculate dbeta
  float db = 0;
  // for sites of this feature
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    db += f;
  }
  dbeta[c] = db;

  // step 2 - calculate dgamma
  float dg = 0;
  // for sites of this feature
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    float df = dfeatures[i * nColumns + c];
    dg += (f - m) * (pow(v + eps, -1.f / 2.f)) * df;
  }
  dgamma[c] = dg;

  // step 3 - calculate dfeatures
  float dfm = 0;
  // for sites of this feature
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    float df = dfeatures[i * nColumns + c];
    dfm += df * (f - m);
  }
  // for sites of this feature
  for (int i = 0; i < nRows; i++) {
    float f = features[i * nColumns + c];
    float df = dfeatures[i * nColumns + c];
    float dfu = (1.0f / nRows) * g * (pow(v + eps, -1.f / 2.f)) * (nRows * df - db) -
    		(f - m) * (pow(v + eps, -1.f)) * dfm;
    dfeatures[i * nColumns + c] = dfu;
  }
}

void BNBackprop(float *features, float *dfeatures, float *gamma, float *dgamma, float *beta, float *dbeta,
		float *mu, float *var, int nRows, int nColumns, cudaMemStream &memStream) {
  dBNBackprop << <dim3(nColumns / KERNELBLOCKSIZE, KERNELBLOCKSIZE),
                     KERNELBLOCKSIZE, 0, memStream.stream>>>
      (features, dfeatures, gamma, dgamma, beta, dbeta, mu, var, nRows, nColumns);
  cudaCheckError();
}

__global__ void dGradientDescentShrunkVectorKeepPositiveBN(
    float *d_delta, float *d_momentum, float *d_weights, int nOut,
    int nOutDropout, int *outFeaturesPresent, float learningRate,
    float momentum) {
  for (int i = threadIdx.x; i < nOutDropout; i += NTHREADS) {
    int ii = outFeaturesPresent[i];
    // NAG light
    float w = d_weights[ii];
    float m = d_momentum[ii];
    float d = d_delta[i];
    w -= m * momentum;
    m = momentum * m - learningRate * (1 - momentum) * d;
    w += m * (1 + momentum);
    w = (w > 0.001) ? w : 0.001;
    d_weights[ii] = w;
    d_momentum[ii] = m;
  }
}

__global__ void dGradientDescentKeepPositiveBN(float *d_delta, float *d_momentum,
                                             float *d_weights, int nOut,
                                             float learningRate,
                                             float momentum) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    float w = d_weights[j];
    float m = d_momentum[j];
    float d = d_delta[j];
    w -= m * momentum;
    m = momentum * m - learningRate * (1 - momentum) * d;
    w += m * (1 + momentum);
    w = (w > 0.001) ? w : 0.001;
    d_weights[j] = w;
    d_momentum[j] = m;
  }
}

NetworkInNetworkBNLayer::NetworkInNetworkBNLayer(
    cudaMemStream &memStream, cublasHandle_t &cublasHandle, int nFeaturesIn,
    int nFeaturesOut, float dropout, ActivationFunction fn,
    float alpha // used to determine intialization weights only
    )
    : SpatiallySparseLayer(memStream), cublasHandle(cublasHandle),
      nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut), dropout(dropout),
      W(true, nFeaturesIn * nFeaturesOut), MW(true, nFeaturesIn * nFeaturesOut),
      B(true, nFeaturesOut), MB(true, nFeaturesOut), Gamma(true, nFeaturesOut),
      MGamma(true, nFeaturesOut), Beta(true, nFeaturesOut), MBeta(true, nFeaturesOut),
      RunningMean(true, nFeaturesOut), RunningVar(true, nFeaturesOut),
      Mu(true, nFeaturesOut), Var(true, nFeaturesOut) {
  float scale = pow(6.0f / (nFeaturesIn + nFeaturesOut * alpha), 0.5f);
  W.copyToCPUAsync(memStream);
  W.setUniform(-scale, scale);
  W.copyToGPUAsync(memStream);
  MW.setZero();
  B.setZero();
  MB.setZero();
  Gamma.copyToCPUAsync(memStream);
  Gamma.setConstant(1.0f);
  Gamma.copyToGPUAsync(memStream);
  MGamma.setZero();
  Beta.setZero();
  MBeta.setZero();
  RunningMean.setZero();
  RunningVar.setZero();
  Mu.setZero();
  Var.setZero();
  std::cout << "Learn " << nFeaturesIn << "->" << nFeaturesOut
            << " dropout=" << dropout << " " << sigmoidNames[PRELU]
            << std::endl;
}
void NetworkInNetworkBNLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.nFeatures == nFeaturesIn);
  output.nFeatures = nFeaturesOut;
  output.spatialSize = input.spatialSize;
  output.nSpatialSites = input.nSpatialSites;
  output.grids = input.grids;
  int o = nFeaturesOut * (batch.type == TRAINBATCH ? (1.0f - dropout) : 1.0f);
  output.featuresPresent.hVector() = rng.NchooseM(nFeaturesOut, o);
  output.backpropErrors = true;
}
void NetworkInNetworkBNLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  // std::cerr << output.nFeatures << " " << PReLU.meanAbs() << "\n";
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  if (batch.type == TRAINBATCH and
      nFeaturesIn + nFeaturesOut >
          input.featuresPresent.size() + output.featuresPresent.size()) {
    w.resize(input.featuresPresent.size() * output.featuresPresent.size());
    dShrinkMatrixForDropout << <input.featuresPresent.size(), KERNELBLOCKSIZE,
                                0, memStream.stream>>>
        (W.dPtr(), w.dPtr(), input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(), output.nFeatures,
         output.featuresPresent.size());
    cudaCheckError();
    b.resize(output.featuresPresent.size());
    dShrinkVectorForDropout << <1, NTHREADS, 0, memStream.stream>>>
        (B.dPtr(), b.dPtr(), output.featuresPresent.dPtr(), output.nFeatures,
         output.featuresPresent.size());
    gamma.resize(output.featuresPresent.size());
    dShrinkVectorForDropout << <1, NTHREADS, 0, memStream.stream>>>
        (Gamma.dPtr(), gamma.dPtr(), output.featuresPresent.dPtr(),
         output.nFeatures, output.featuresPresent.size());
    beta.resize(output.featuresPresent.size());
    dShrinkVectorForDropout << <1, NTHREADS, 0, memStream.stream>>>
        (Beta.dPtr(), beta.dPtr(), output.featuresPresent.dPtr(),
         output.nFeatures, output.featuresPresent.size());
    cudaCheckError();
    replicateArray(b.dPtr(), output.sub->features.dPtr(), output.nSpatialSites,
                   output.featuresPresent.size(), memStream);
    cudaCheckError();
    d_rowMajorSGEMM_alphaAB_betaC(
        cublasHandle, input.sub->features.dPtr(), w.dPtr(),
        output.sub->features.dPtr(), output.nSpatialSites,
        input.featuresPresent.size(), output.featuresPresent.size(), 1.0f, 1.0f,
        __FILE__, __LINE__);
    cudaCheckError();
    BN(output.sub->features.dPtr(), gamma.dPtr(), beta.dPtr(), RunningMean.dPtr(), RunningVar.dPtr(),
    		Mu.dPtr(), Var.dPtr(), batch.type, output.nSpatialSites, output.featuresPresent.size(), memStream);
  } else {
    float p = 1.0f - (batch.type != RESCALEBATCH ? dropout : 0);
    replicateArray(B.dPtr(), output.sub->features.dPtr(), output.nSpatialSites,
                   output.featuresPresent.size(), memStream);
    d_rowMajorSGEMM_alphaAB_betaC(cublasHandle, input.sub->features.dPtr(),
                                  W.dPtr(), output.sub->features.dPtr(),
                                  output.nSpatialSites, input.nFeatures,
                                  output.nFeatures, p, p, __FILE__, __LINE__);
    cudaCheckError();
    BN(output.sub->features.dPtr(), Gamma.dPtr(), Beta.dPtr(), RunningMean.dPtr(), RunningVar.dPtr(),
    		Mu.dPtr(), Var.dPtr(), batch.type, output.nSpatialSites, output.nFeatures, memStream);
  }
  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() *
                      output.featuresPresent.size();
}
void NetworkInNetworkBNLayer::scaleWeights(
    SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output,
    float &scalingUnderneath, bool topLayer) {
  assert(output.sub->features.size() > 0 && "call after forwards(...)");
  float scale = output.sub->features.meanAbs();
  std::cout << "featureScale:" << scale << std::endl;
  if (topLayer) {
    scale = 1;
  } else {
    scale = powf(scale, -0.1);
  }
  W.multiplicativeRescale(scale / scalingUnderneath);
  B.multiplicativeRescale(scale);
  MW.multiplicativeRescale(scale / scalingUnderneath);
  MB.multiplicativeRescale(scale);
  scalingUnderneath = scale;
}

void NetworkInNetworkBNLayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  dw.resize(input.featuresPresent.size() * output.featuresPresent.size());
  db.resize(output.featuresPresent.size());
  dgamma.resize(output.featuresPresent.size());
  dbeta.resize(output.featuresPresent.size());
  dgamma.setZero(memStream);
  dbeta.setZero(memStream);
  if (output.featuresPresent.size() < output.featuresPresent.size()) {
  	BNBackprop(output.sub->features.dPtr(), output.sub->dfeatures.dPtr(), gamma.dPtr(), dgamma.dPtr(),
  			beta.dPtr(), dbeta.dPtr(), Mu.dPtr(), Var.dPtr(), output.nSpatialSites, output.featuresPresent.size(), memStream);
    dGradientDescentShrunkVectorKeepPositiveBN
            << <1, NTHREADS, 0, memStream.stream>>>
        (dgamma.dPtr(), MGamma.dPtr(), Gamma.dPtr(), output.nFeatures,
         output.featuresPresent.size(), output.featuresPresent.dPtr(),
         learningRate, momentum);
    dGradientDescentShrunkVectorKeepPositiveBN
            << <1, NTHREADS, 0, memStream.stream>>>
        (dbeta.dPtr(), MBeta.dPtr(), Beta.dPtr(), output.nFeatures,
         output.featuresPresent.size(), output.featuresPresent.dPtr(),
         learningRate, momentum);
  } else {
  	BNBackprop(output.sub->features.dPtr(), output.sub->dfeatures.dPtr(),  Gamma.dPtr(), dgamma.dPtr(),
  			Beta.dPtr(), dbeta.dPtr(), Mu.dPtr(), Var.dPtr(), output.nSpatialSites, output.featuresPresent.size(), memStream);
    dGradientDescentKeepPositiveBN << <1, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dgamma.dPtr(), MGamma.dPtr(), Gamma.dPtr(), nFeaturesOut, learningRate,
         momentum);
    dGradientDescentKeepPositiveBN << <1, KERNELBLOCKSIZE, 0, memStream.stream>>>
            (dbeta.dPtr(), MBeta.dPtr(), Beta.dPtr(), nFeaturesOut, learningRate,
             momentum);
  }
  cudaCheckError();

  d_rowMajorSGEMM_alphaAtB_betaC(
      cublasHandle, input.sub->features.dPtr(), output.sub->dfeatures.dPtr(),
      dw.dPtr(), input.featuresPresent.size(), output.nSpatialSites,
      output.featuresPresent.size(), 1.0, 0.0);

  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() *
                      output.featuresPresent.size();
  cudaCheckError();
  db.setZero(memStream);
  columnSum(output.sub->dfeatures.dPtr(), db.dPtr(), output.nSpatialSites,
            output.featuresPresent.size(), memStream);

  if (nFeaturesIn + nFeaturesOut >
      input.featuresPresent.size() + output.featuresPresent.size()) {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites *
                                  input.featuresPresent.size());
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle, output.sub->dfeatures.dPtr(),
                                     w.dPtr(), input.sub->dfeatures.dPtr(),
                                     output.nSpatialSites,
                                     output.featuresPresent.size(),
                                     input.featuresPresent.size(), 1.0, 0.0);
      multiplyAddCount += (__int128_t)output.nSpatialSites *
                          input.featuresPresent.size() *
                          output.featuresPresent.size();
      cudaCheckError();
    }

    dGradientDescentShrunkMatrix << <input.featuresPresent.size(),
                                     KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(), output.nFeatures,
         output.featuresPresent.size(), input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(), learningRate, momentum);
    cudaCheckError();

    dGradientDescentShrunkVector << <1, NTHREADS, 0, memStream.stream>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), output.nFeatures,
         output.featuresPresent.size(), output.featuresPresent.dPtr(),
         learningRate, momentum);
    cudaCheckError();
  } else {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites *
                                  input.featuresPresent.size());
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle, output.sub->dfeatures.dPtr(),
                                     W.dPtr(), input.sub->dfeatures.dPtr(),
                                     output.nSpatialSites, nFeaturesOut,
                                     nFeaturesIn, 1.0, 0.0);
      multiplyAddCount += (__int128_t)output.nSpatialSites *
                          input.featuresPresent.size() *
                          output.featuresPresent.size();
      cudaCheckError();
    }
    dGradientDescent << <nFeaturesIn, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(), nFeaturesOut, learningRate, momentum);
    cudaCheckError();
    dGradientDescent << <1, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), nFeaturesOut, learningRate, momentum);
    cudaCheckError();
  }
}

// TODO: add BN params
void NetworkInNetworkBNLayer::loadWeightsFromStream(std::ifstream &f,
                                                       bool momentum) {
  W.copyToCPUAsync(memStream);
  MW.copyToCPUAsync(memStream);
  B.copyToCPUAsync(memStream);
  MB.copyToCPUAsync(memStream);
  Gamma.copyToCPUAsync(memStream);
  MGamma.copyToCPUAsync(memStream);
  Beta.copyToCPUAsync(memStream);
  MBeta.copyToCPUAsync(memStream);

  f.read((char *)&W.hVector()[0], sizeof(float) * W.size());
  f.read((char *)&B.hVector()[0], sizeof(float) * B.size());
  f.read((char *)&Gamma.hVector()[0], sizeof(float) * Gamma.size());
  f.read((char *)&Beta.hVector()[0], sizeof(float) * Beta.size());
  if (momentum) {
    f.read((char *)&MW.hVector()[0], sizeof(float) * MW.size());
    f.read((char *)&MB.hVector()[0], sizeof(float) * MB.size());
    f.read((char *)&MGamma.hVector()[0], sizeof(float) * MGamma.size());
    f.read((char *)&MBeta.hVector()[0], sizeof(float) * MBeta.size());
  } else {
    MW.setZero();
    MB.setZero();
    MGamma.setZero();
    MBeta.setZero();
  }

  W.copyToGPUAsync(memStream);
  MW.copyToGPUAsync(memStream);
  B.copyToGPUAsync(memStream);
  MB.copyToGPUAsync(memStream);
  Gamma.copyToGPUAsync(memStream);
  MGamma.copyToGPUAsync(memStream);
  Beta.copyToGPUAsync(memStream);
  MBeta.copyToGPUAsync(memStream);
}

// TODO: add BN params
void NetworkInNetworkBNLayer::putWeightsToStream(std::ofstream &f,
                                                    bool momentum) {
  W.copyToCPUAsync(memStream);
  MW.copyToCPUAsync(memStream);
  B.copyToCPUAsync(memStream);
  MB.copyToCPUAsync(memStream);
  Gamma.copyToCPUAsync(memStream);
  MGamma.copyToCPUAsync(memStream);
  Beta.copyToCPUAsync(memStream);
  MBeta.copyToCPUAsync(memStream);
  f.write((char *)&W.hVector()[0], sizeof(float) * W.size());
  f.write((char *)&B.hVector()[0], sizeof(float) * B.size());
  f.write((char *)&Gamma.hVector()[0], sizeof(float) * Gamma.size());
  f.write((char *)&Beta.hVector()[0], sizeof(float) * Beta.size());
  if (momentum) {
    f.write((char *)&MW.hVector()[0], sizeof(float) * MW.size());
    f.write((char *)&MB.hVector()[0], sizeof(float) * MB.size());
    f.write((char *)&MGamma.hVector()[0], sizeof(float) * MGamma.size());
    f.write((char *)&MBeta.hVector()[0], sizeof(float) * MBeta.size());
  }
  W.copyToGPUAsync(memStream);
  MW.copyToGPUAsync(memStream);
  B.copyToGPUAsync(memStream);
  MB.copyToGPUAsync(memStream);
  Gamma.copyToGPUAsync(memStream);
  Beta.copyToGPUAsync(memStream);
  MGamma.copyToGPUAsync(memStream);
  MBeta.copyToGPUAsync(memStream);
};
int NetworkInNetworkBNLayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  return outputSpatialSize;
}
