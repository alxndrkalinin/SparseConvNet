#pragma once
#include "SpatiallySparseDataset.h"
#include <iostream>

SpatiallySparseDataset FibroblastTrainSet(int renderSize, int kFold, int fold);
SpatiallySparseDataset FibroblastTestSet(int renderSize, int kFold, int fold);
