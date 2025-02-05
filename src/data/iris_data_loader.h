#pragma once
#include "common.h"

class IrisDataLoader {
public:
    IrisDataLoader();
    ~IrisDataLoader();
    
    bool loadData(const std::string& filename);
    IrisData& getData() { return data; }
    
private:
    IrisData data;
    void allocateMemory(int n_samples);
    void freeMemory();
};
