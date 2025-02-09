#pragma once
#include "common.h"
#include <vector>
#include <fstream>
#include <sstream>

class IrisDataLoader {
public:
    static bool loadData(IrisData& data);
private:
    static bool loadFromFile(std::vector<float>& features, std::vector<int>& labels);
    static bool allocateMemory(IrisData& data, int n_samples);
    static void freeMemory(IrisData& data);
};
