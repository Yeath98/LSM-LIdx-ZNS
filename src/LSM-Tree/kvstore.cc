#include "kvstore.h"
//add
#include <Python.h>
#include <numpy/arrayobject.h>

//todo:注意reset()

KVStore::KVStore()
{
    memTable = new SkipList;
    Level.push_back(0);

    if(SSTable.empty())
        SSTable.emplace_back();
}

//将 MemTable 中的所有数据以 SSTable 形式写进磁盘
KVStore::~KVStore()
{
//    Table newTable(memoryPool, memTable, ++Level[0]);
//    if(Level[0] > Level0)
//        compactionForLevel(1);
    delete memTable;
}

/**
 * Insert/Update the key-value pair.
 * No return values for simplicity.
 */
void KVStore::put(uint64_t key, const string &s)
{
    string val = memTable->get(key);
    if(!val.empty())
        memTable->memory += s.size() - val.size();
    else
        memTable->memory += 4 + 8 + s.size() + 1;  //索引值 + key + value所占的内存大小 + "\0"

    if(memTable->memory > DEFAULT_SSTABLE_SIZE)
    {
        Level[0]++;
        SSTable[0].emplace(memoryPool, memTable, Level[0]);
        if(!val.empty())
            memTable->memory += s.size() - val.size();
        else
            memTable->memory += 4 + 8 + s.size() + 1;
        if(Level[0] > Level0)
        {
            compactionForLevel(1);
            Level[0] = 0;
        }
    }
    memTable->put(key, s);
}
/**
 * Returns the (string) value of the given key.
 * An empty string indicates not found.
 */

std::string KVStore::get(uint64_t key)
{
    string ans = memTable->get(key);

    if(!ans.empty())
    {
        if(ans==DEL)
            return "";
        else
            return ans;
    }

    for(const auto &tableList:SSTable)
    {
        for(auto table = tableList.rbegin(); table!=tableList.rend(); ++table)
        {
            ans = table->getValue(memoryPool, key);
            if(!ans.empty())
            {
                if(ans==DEL)
                    return "";
                else
                    return ans;
            }
        }
    }

    return ans;
}
/**
 * Delete the given key-value pair if it exists.
 * Returns false iff the key is not found.
 */
bool KVStore::del(uint64_t key)
{
    bool flag = !get(key).empty();
    if(flag)
        put(key, DEL);
    return flag;
}

void update(vector<vector<pair<uint64_t, string>>> &KVToCompact, vector<vector<pair<uint64_t, string>>::iterator> &KVToCompactIter,
        map<uint64_t,int> &minKey, int index) {
    while(KVToCompactIter[index] != KVToCompact[index].end()) {
        uint64_t select = KVToCompactIter[index]->first;
        if(minKey.count(select)==0) {
            minKey.emplace(select, index);
            break;
        }
        else if(index > minKey[select]) {
            int pos = minKey[select];
            minKey[select] = index;
            update(KVToCompact, KVToCompactIter, minKey, pos);
            break;
        }
        KVToCompactIter[index]++;
    }
}

/**
 * 递归合并各层SSTABLE，对0层将所有文件向下合并
 * 其它层则只将超出最大个数的文件向下合并
 * @param level 被合并的层
 */

bool operator<(const set<Table>::iterator &a, const set<Table>::iterator &b) {
        return (*a) > (*b);
}

void KVStore::compactionForLevel(int level)
{
    //递归中止条件，如果上层的文件数小于最大上限，则不再合并
    int SSTnum = SSTable[level-1].size();
    if(level != 1 && SSTnum <= UpperNum(level-1)) {
        memoryPool.tryReset();
        return;
    }

    if(Level.size()<=level)
    {
        Level.push_back(0);
        SSTable.emplace_back();
    }

    bool lastLevel = false;
    if(level == SSTable.size()-1)
        lastLevel = true;

    vector<set<Table>::iterator> FileToRemoveLevelminus1;                    //记录需要被删除的文件
    vector<set<Table>::iterator> FileToRemoveLevel;


    //需要被合并的SSTable
    priority_queue<set<Table>::iterator> sortTable;

    uint64_t timestamp = 0;
    uint64_t tempMin = INT64_MAX, tempMax = 0;

    //需要合并的文件数
    int compactNum = (level == 1)? Level0+1: SSTnum - UpperNum(level-1);

    //遍历Level-1中被合并的SSTable，获得时间戳和最大最小键
    auto it = SSTable[level-1].begin();
    for(int i=0 ; i<compactNum; ++i)
    {
        sortTable.push(it);
        FileToRemoveLevelminus1.push_back(it);

        timestamp = it->getTimestamp();
        if(it->getMaxKey()>tempMax)
            tempMax = it->getMaxKey();
        if(it->getMinKey()<tempMin)
            tempMin = it->getMinKey();
        it++;
    }

    //找到Level中和Level-1中键有交集的文件
    for(auto iter = SSTable[level].begin(); iter != SSTable[level].end(); ++iter)
    {
        if(iter->getMaxKey()>=tempMin && iter->getMinKey()<=tempMax)
        {
            sortTable.push(iter);
            FileToRemoveLevel.push_back(iter);
        }
    }

    vector<vector<pair<uint64_t, string>>> KVToCompact;                   //被合并的键值对，下标越大时间戳越大
    vector<vector<pair<uint64_t, string>>::iterator> KVToCompactIter;     //键值对迭代器
    uint64_t size = InitialSize;
    map<uint64_t, int> minKey;                      //minKey中只存放num个数据，分别为各个SSTable中最小键和对应的SSTable
    uint64_t tempKey;
    int index;                                      //对应的SSTable序号
    string tempValue;
    vector<pair<uint64_t, string>> newTable;                 //暂存合并后的键值对
    newTable.reserve(81920);

    while(!sortTable.empty()) {
        vector<pair<uint64_t, string>> KVPair;
        KVPair.reserve(20480);
        sortTable.top()->traverse(memoryPool, KVPair);
        sortTable.pop();
        KVToCompact.push_back(move(KVPair));
    }

    KVToCompactIter.resize(KVToCompact.size());
    //获取键值对迭代器
    for(int i=KVToCompact.size()-1; i>=0; --i)
    {
        auto iter = KVToCompact[i].begin();
        while(iter!=KVToCompact[i].end())
        {
            if(minKey.count(iter->first)==0)  //如果键相同，保留时间戳较大的
            {
                minKey[iter->first] = i;
                break;
            }
            iter++;
        }
        if(iter!=KVToCompact[i].end())
            KVToCompactIter[i] = iter;
    }

    int nums=0;
    //只要minKey不为空，minKey的第一个元素一定为所有SSTable中的最小键
    //每个循环将minKey中的最小键和对应的值加入newTable
    while(!minKey.empty()) {
        auto iter = minKey.begin();
        tempKey = iter->first;
        index = iter->second;
        tempValue = move(KVToCompactIter[index]->second);
        if(lastLevel && tempValue == DEL)    //最后一层的删除标记不写入文件
            goto next;
        size += tempValue.size() + 1 + 12;
        if(size > DEFAULT_SSTABLE_SIZE) {
            writeToFile(level, timestamp, newTable.size(), newTable);
            size = InitialSize + tempValue.size() + 1 + 12;
        }
        newTable.emplace_back(tempKey, move(tempValue));
next:   minKey.erase(tempKey);
        KVToCompactIter[index]++;
        update(KVToCompact, KVToCompactIter, minKey, index);
        if(KVToCompactIter[index]==KVToCompact[index].end())
            nums++;
    }

    //不足2MB的文件也要写入
    if(!newTable.empty())
        writeToFile(level, timestamp, newTable.size(), newTable);

    //删除level和level-1中的被合并文件
    for(auto &file:FileToRemoveLevelminus1) {
        file->clear(memoryPool);
        SSTable[level-1].erase(file);
    }

    for(auto &file:FileToRemoveLevel) {
        file->clear(memoryPool);
        SSTable[level].erase(file);
    }

    compactionForLevel(level+1);
}

void KVStore::writeToFile(int level, uint64_t timeStamp, uint64_t numPair, vector<pair<uint64_t, string>> &newTable)
{
    Level[level]++;
    SSTable[level].emplace(memoryPool, level, Level[level], timeStamp, numPair, newTable);
    newTable.clear();
}

void KVStore::printSSTablesInfo() {
    std::cout << "SSTable Information:" << std::endl;
    for (size_t level = 0; level < SSTable.size(); ++level) {
        std::cout << "Level " << level << ":" << std::endl;
        for (const auto& table : SSTable[level]) {
            std::cout << "  Table Order: " << table.getTimestamp() << std::endl;
            std::cout << "    Min Key: " << table.getMinKey() << std::endl;
            std::cout << "    Max Key: " << table.getMaxKey() << std::endl;
            std::cout << "    Number of Pairs: " << table.getNumPair() << std::endl;

            // 打印键值对
            // std::vector<std::pair<uint64_t, std::string>> keyValuePairs;
            // table.traverse(memoryPool, keyValuePairs);
            // std::cout << "    Key-Value Pairs:" << std::endl;
            // for (const auto& pair : keyValuePairs) {
            //     std::cout << "      Key: " << pair.first << ", Value: " << pair.second << std::endl;
            // }
        }
    }
}


void KVStore::extractSSTableKeys(std::vector<std::tuple<uint64_t, uint64_t, int, int>>& keys) {
    for (size_t level = 0; level < SSTable.size(); ++level) {
        for (const auto& table : SSTable[level]) {
            keys.emplace_back(
                table.getMinKey(),  // 最小 key
                table.getMaxKey(),  // 最大 key
                static_cast<int>(level),  // level
                static_cast<int>(table.getTimestamp())  // order
            );
        }
    }
}


class HTMModel {
public:
    HTMModel(const std::string& modelPath) {
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('./')");
        pName = PyUnicode_FromString("train_htm");
        pModule = PyImport_Import(pName);
        Py_DECREF(pName);
        pFunc = PyObject_GetAttrString(pModule, "loadModel");
        pModel = PyObject_CallObject(pFunc, NULL);
        Py_DECREF(pFunc);
    }

    ~HTMModel() {
        Py_DECREF(pModule);
        Py_DECREF(pModel);
        Py_Finalize();
    }

    std::vector<int> predict(uint64_t key) {
        npy_intp dims[1] = {2};
        PyObject* pArray = PyArray_SimpleNew(1, dims, NPY_UINT64);
        *((uint64_t*)PyArray_DATA(pArray)) = key;
        *((uint64_t*)PyArray_DATA(pArray) + 1) = key;
        PyObject* pResult = PyObject_CallMethod(pModel, "predict", "(O)", pArray);
        uint64_t level = PyLong_AsLong(PyTuple_GetItem(pResult, 0));
        uint64_t order = PyLong_AsLong(PyTuple_GetItem(pResult, 1));
        Py_DECREF(pResult);
        Py_DECREF(pArray);
        return {static_cast<int>(level), static_cast<int>(order)};
    }

private:
    PyObject* pName = NULL;
    PyObject* pModule = NULL;
    PyObject* pFunc = NULL;
    PyObject* pModel = NULL;
};

HTMModel* g_htmModel = NULL;

void KVStore::loadHTMModel(const std::string& modelPath) {
    g_htmModel = new HTMModel(modelPath);
}

std::vector<int> KVStore::predictSSTableLocation(uint64_t key) {
    if (g_htmModel == NULL) {
        return {};
    }
    return g_htmModel->predict(key);
}