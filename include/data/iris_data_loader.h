#pragma once
#include "common.h"

class IrisDataLoader {
public:
    IrisDataLoader();
    ~IrisDataLoader();
    
    static bool loadData(IrisData& data);
    IrisData& getData() { return data; }
    
private:
    IrisData data;
    void allocateMemory(int n_samples);
    void freeMemory();
    static bool loadFromFile(std::vector<float>& features, 
                           std::vector<int>& labels);
};
