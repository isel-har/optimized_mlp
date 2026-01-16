#ifndef SAVE_SPLIT_SCLAER_HPP
#define SAVE_SPLIT_SCLAER_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "scaler.hpp"
#include "csv_to_eigen.hpp"

rapidcsv::Document shuffle_rows(const rapidcsv::Document& doc);

void save_scale(rapidcsv::Document& doc);
void save_split(const rapidcsv::Document& doc, size_t val_size);
void save_split_scaler(const std::string& path, size_t val_size);

#endif
