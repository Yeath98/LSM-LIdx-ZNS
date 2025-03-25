#pragma once

#include "Table.h"
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <tuple>
cd 


using namespace std;

const string DEL = "~DELETED~";
inline int UpperNum(int i) {return (int)pow(3, i+1);}
const int Level0 = 3;

class KVStore {
private:
    SkipList *memTable;
    vector<int> Level; //记录对应层的文件数目
    vector<set<Table>> SSTable;
    MemoryManager memoryPool;
public:
	KVStore();

	~KVStore();

	void put(uint64_t key, const std::string &s);

	std::string get(uint64_t key);

	bool del(uint64_t key);

	void compactionForLevel(int level);

	void writeToFile(int level, uint64_t timeStamp, uint64_t numPair, vector<pair<uint64_t, string>> &newTable);

	void printSSTablesInfo();  // 添加打印 SSTable 信息的函数声明

	void extractSSTableKeys(vector<tuple<uint64_t, uint64_t, int, int>>& keys);//提取所有 SST 块的最小和最大 key 以及它们的位置

	std::vector<int> predictSSTableLocation(uint64_t key);//在查询时使用 HTM 模型预测 SST 块的位置
	
	//HTM-LIDX
	void loadHTMModel(const std::string& modelPath);
};
