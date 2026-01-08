#ifndef HISTORY_HPP
#define HISTORY_HPP

#include <vector>
#include <utility>
#include <string>
#include <unordered_map>

class History {
public:
    std::pair<std::vector<double>, std::vector<double>> loss_pair; // 1 train, 2 val
    std::pair<std::vector<double>, std::vector<double>> accuracy_pair;
    std::pair<std::vector<double>, std::vector<double>> precision_pair;
    // std::pair<std::vector<double>, std::vector<double>> recall_pair;
    
    History(size_t);
    static std::pair<std::vector<double>, std::vector<double>> create_pair(size_t);

    std::unordered_map<std::string, std::pair<std::vector<double>, std::vector<double>>*> vecMap;
};

#endif
