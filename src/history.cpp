#include "history.hpp"

std::pair<std::vector<double>, std::vector<double>> History::create_pair(size_t size)
{
    return std::make_pair(std::vector<double>(size), std::vector<double>(size));
}

History::History(size_t size)
{
    this->vecMap["loss"]      = create_pair(size);
    this->vecMap["accuracy"]  = create_pair(size);
    this->vecMap["precision"] = create_pair(size);
}
