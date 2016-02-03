#include "SparseConvNet.h"
#include "SpatiallySparseDatasetSHREC2015.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 10;

class DeepC2Triangular : public SparseConvTriangLeNet {
public:
  DeepC2Triangular(int dimension, int l, int k, ActivationFunction fn,
                   int nInputFeatures, int nClasses, float p = 0.0f,
                   int cudaDevice = -1, int nTop = 1);
};
DeepC2Triangular::DeepC2Triangular(int dimension, int l, int k,
                                   ActivationFunction fn, int nInputFeatures,
                                   int nClasses, float p, int cudaDevice,
                                   int nTop)
    : SparseConvTriangLeNet(dimension, nInputFeatures, nClasses, cudaDevice, nTop) {
//  for (int i = 0; i <= l; i++)
//    addLeNetLayerMP((i + 1) * k, (i == l) ? 2 : 2, 1, (i < l) ? 3 : 1,
//                              (i < l) ? 2 : 1, fn, p * i * 1.0f / l);
  
  // VGG-like
  // addTriangularLeNetLayerMP(nFeatures, filterSize, filterStride, poolSize, poolStride, activationFn, dropout, minActiveInputs);
//  addLeNetLayerMP(k, 3, 1, 1, 1, fn, 0); // no pooling
//  addLeNetLayerMP(k, 3, 1, 2, 1, fn, p * 1.0f / l);
//  addLeNetLayerMP(2 * k, 3, 1, 1, 1, fn, p * 2.0f / l); // no pooling
//  addLeNetLayerMP(2 * k, 3, 1, 2, 2, fn, p * 3.0f / l);
//  addLeNetLayerMP(3 * k, 3, 1, 1, 1, fn, p * 4.0f / l); // no pooling
//  addLeNetLayerMP(3 * k, 3, 1, 1, 1, fn, p * 5.0f / l); // no pooling
//  addLeNetLayerMP(3 * k, 3, 1, 2, 2, fn, p * 6.0f / l);
//  addLeNetLayerMP(4 * k, 3, 1, 1, 1, fn, p * 7.0f / l); // no pooling
//  addLeNetLayerMP(4 * k, 3, 1, 1, 1, fn, p * 8.0f / l); // no pooling
//  addLeNetLayerMP(4 * k, 3, 1, 2, 2, fn, p * 9.0f / l);
//  addLeNetLayerMP(4 * k, 3, 1, 1, 1, fn, p * 10.0f / l); // no pooling
//  addLeNetLayerMP(4 * k, 3, 1, 1, 1, fn, p * 11.0f / l); // no pooling
//  addLeNetLayerMP(4 * k, 1, 1, 2, 2, fn, p * 12.0f / l);
//  addLeNetLayerMP(24 * k, 1, 1, 1, 1, fn, 0.5f);
//  addLeNetLayerMP(24 * k, 1, 1, 1, 1, fn, 0.5f);
//  addLeNetLayerMP(15 * k, 1, 1, 1, 1, fn, p * 13.0f / l);

  // NiN-like
  addLeNetLayerMP(192, 5, 1, 1, 1, fn, 0); // no pooling
  addLeNetLayerMP(160, 1, 1, 1, 1, fn, 0); // no pooling
  addLeNetLayerMP(96, 1, 1, 3, 2, fn, 0.5); // max-pooling + dropout
  
  addLeNetLayerMP(192, 5, 1, 1, 1, fn, 0); // no pooling
  addLeNetLayerMP(192, 1, 1, 1, 1, fn, 0); // no pooling
  addLeNetLayerMP(192, 1, 1, 3, 2, fn, 0.5); // max-pooling + dropout
  
  addLeNetLayerMP(192, 3, 1, 1, 1, fn, 0); // no pooling
  addLeNetLayerMP(192, 1, 1, 1, 1, fn, 0); // no pooling
  addLeNetLayerMP(10, 1, 1, 8, 1, fn, 0); // max-pooling

  // Softmax output
  addSoftmaxLayer();
}

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/SHREC2015";
  int fold = 0;
  if (lenArgs > 1)
    int fold = atoi(args[1]);
  std::cout << "Fold: " << fold << std::endl;
  SpatiallySparseDataset trainSet = SHREC2015TrainSet(60, 10, fold);
  trainSet.summary();
  trainSet.repeatSamples(15);
  trainSet.summary();
  SpatiallySparseDataset testSet = SHREC2015TestSet(60, 10, fold);
  testSet.summary();

  DeepC2Triangular cnn(3, 10, 60, VLEAKYRELU, trainSet.nFeatures,
                       trainSet.nClasses, 0.0f, cudaDevice);
  
  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 100 * 5; epoch++) {
    std::cout << "epoch:" << epoch << ": " << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.01 * epoch));
    if (epoch % 1 == 0) {
//      cnn.saveWeights(baseName, epoch);
      cnn.processDatasetRepeatTest(testSet, batchSize, 1);
    }
  }
}
