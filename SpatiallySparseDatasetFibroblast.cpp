// Fibroblast: 2 classes, NORM, 494 exemplars, and SS, 505 exemplars

#include "SpatiallySparseDatasetFibroblast.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "OpenCVPicture.h"
#include "Off3DFormatPicture.h"

SpatiallySparseDataset FibroblastTrainSet(int renderSize, int kFold, int fold) {
  SpatiallySparseDataset dataset;
  dataset.name = "Fibroblast (Training subset)";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 2;
  std::ifstream cla("Data/Fibroblast/fibroblast.cla");
  std::string line;
  int nClasses, nTotal;
  getline(cla, line); // header line
  cla >> nClasses >> nTotal;
  for (int cl = 0; cl < nClasses; cl++) {
    getline(cla, line); // blank line
    std::string className;
    int parent, nExemplars;
    cla >> className >> parent >> nExemplars;
    for (int exemp = 0; exemp < nExemplars; exemp++) {
      int num;
      cla >> num;
      std::string filename =
          std::string("Data/Fibroblast/") +
          std::to_string(num) + std::string(".off");
      if (exemp % kFold != fold)
        dataset.pictures.push_back(
            new OffSurfaceModelPicture(filename, renderSize, cl));
    }
  }
  return dataset;
};

SpatiallySparseDataset FibroblastTestSet(int renderSize, int kFold, int fold) {
  SpatiallySparseDataset dataset;
  dataset.name = "Fibroblast (Validation subset)";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 2;
  std::ifstream cla("Data/Fibroblast/fibroblast.cla");
  std::string line;
  int nClasses, nTotal;
  getline(cla, line); // header line
  cla >> nClasses >> nTotal;
  for (int cl = 0; cl < nClasses; cl++) {
    getline(cla, line); // blank line
    std::string className;
    int parent, nExemplars;
    cla >> className >> parent >> nExemplars;
    for (int exemp = 0; exemp < nExemplars; exemp++) {
      int num;
      cla >> num;
      std::string filename =
          std::string("Data/Fibroblast/") +
          std::to_string(num) + std::string(".off");
      if (exemp % kFold == fold)
        dataset.pictures.push_back(
            new OffSurfaceModelPicture(filename, renderSize, cl));
    }
  }
  return dataset;
};
