#pragma once
#include <fstream>
#include "SpatiallySparseLayer.h"
#include "NetworkInNetworkLayer.h"
#include "Rng.h"

class NetworkInNetworkBNLayer : public SpatiallySparseLayer {
private:
  RNG rng;
  cublasHandle_t &cublasHandle;

public:
  vectorCUDA<float> W;      // Weights
  vectorCUDA<float> MW;     // momentum
  vectorCUDA<float> w;      // shrunk versions
  vectorCUDA<float> dw;     // For backprop
  vectorCUDA<float> B;      // Weights
  vectorCUDA<float> MB;     // momentum
  vectorCUDA<float> b;      // shrunk versions
  vectorCUDA<float> db;     // For backprop
  // for BN
  vectorCUDA<float> Gamma; // scale
  vectorCUDA<float> MGamma; // momentum
  vectorCUDA<float> gamma; // shrunk
  vectorCUDA<float> dgamma; // backdrop
  vectorCUDA<float> Beta; // shift
  vectorCUDA<float> MBeta; // momentum
  vectorCUDA<float> beta; // shrunk
  vectorCUDA<float> dbeta; // backdrop
  vectorCUDA<float> RunningMean;
  vectorCUDA<float> RunningVar;
  vectorCUDA<float> Mu;
  vectorCUDA<float> Var;
  ActivationFunction fn;
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  NetworkInNetworkBNLayer(
      cudaMemStream &memStream, cublasHandle_t &cublasHandle, int nFeaturesIn,
      int nFeaturesOut, float dropout = 0, ActivationFunction fn = NOSIGMOID,
      float alpha = 1 // used to determine intialization weights only
      );
  void preprocess(SpatiallySparseBatch &batch,
                  SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output);
  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);
  void scaleWeights(SpatiallySparseBatchInterface &input,
                    SpatiallySparseBatchInterface &output,
                    float &scalingUnderneath, bool topLayer);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output, float learningRate,
                 float momentum);
  void loadWeightsFromStream(std::ifstream &f, bool momentum);
  void putWeightsToStream(std::ofstream &f, bool momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};
