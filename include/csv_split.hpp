#ifndef DATA_SPLITER_H
#define DATA_SPLITER_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>


std::pair<std::vector<std::string>, std::vector<std::string>> split_rows(std::vector<std::string>*,
                                                                         float);
std::vector<std::string>                                      csv_to_rawstrs(const char* h);
void shuffle_rows(std::vector<std::string>*);
void save_split_data(const std::string&,
                        std::pair<std::vector<std::string>, std::vector<std::string>>*);

#endif