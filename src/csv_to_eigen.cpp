#include "csv_to_eigen.hpp"

std::pair<MatrixXd, MatrixXd>    csv_to_eigen(const std::string &path) {

    std::pair<MatrixXd, MatrixXd> xy;
    size_t rowsize;
    size_t colsize;

    rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1));
    rowsize = doc.GetRowCount();
    
    const std::vector<char> &yv = doc.GetColumn<char>(1);
    xy.second = MatrixXd::Zero(yv.size(), 2);

    for (size_t i = 0; i < yv.size(); ++i) {

        size_t index = (yv[i] == 'M') ? 0 : 1;
        xy.second(i, index) = 1.0;
    }
    doc.RemoveColumn(1);
    colsize = doc.GetColumnCount();
    
    xy.first = MatrixXd(rowsize, colsize);

    for (size_t i = 0; i < colsize; ++i) {
        std::vector<double> col = doc.GetColumn<double>(i);
        xy.first.col(i) = Map<VectorXd>(col.data(), col.size());
    }
    return xy;
}