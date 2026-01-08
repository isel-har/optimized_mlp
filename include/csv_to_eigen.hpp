#ifndef CSV_TO_EIGEN_HPP
#define CSV_TO_EIGEN_HPP

#include "rapidcsv.h"
#include <Eigen/Dense>
#include <utility>

using namespace Eigen;

typedef struct DatasetSplit {
    MatrixXd X_train;
    MatrixXd y_train;
    MatrixXd X_val;
    MatrixXd y_val;
} t_split;

std::pair<MatrixXd, MatrixXd> csv_to_eigen(const std::string &);

#endif