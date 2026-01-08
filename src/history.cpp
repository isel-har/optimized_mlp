#include "history.hpp"

std::pair<std::vector<double>, std::vector<double>> History::create_pair(size_t size) {
    return std::make_pair(std::vector<double>(size), std::vector<double>(size));
}

History::History(size_t size) {

    this->loss_pair      = create_pair(size);
    this->accuracy_pair  = create_pair(size);
    this->precision_pair = create_pair(size);

    this->vecMap["loss"]      = &this->loss_pair;
    this->vecMap["accuracy"]  = &this->accuracy_pair;
    this->vecMap["precision"] = &this->precision_pair;
}
