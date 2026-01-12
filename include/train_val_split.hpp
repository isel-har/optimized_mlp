#ifndef DATA_PREP_HPP
#define DATA_PREP_HPP

#include "csv_to_eigen.hpp"
#include "scaler.hpp"

t_split train_val_split(const std::string &train_path, const std::string &val_path);

#endif
