#include "activations.hpp"

MatrixXd relu(const MatrixXd& m, bool derivative)
{
    if (!derivative)
        return m.cwiseMax(0.0f);

    return (m.array() > 0.0).cast<double>().matrix();
}

MatrixXd sigmoid(const MatrixXd& m, bool derivative)
{
    // σ(x)
    MatrixXd s = 1.0 / (1.0 + (-m.array()).exp());

    if (!derivative)
        return s;

    // derivative: σ(x) * (1 − σ(x))
    return s.array() * (1.0 - s.array());
}

MatrixXd softmax(const MatrixXd& x, bool derivative)
{
    MatrixXd probs(x.rows(), x.cols());

    for (int i = 0; i < x.rows(); ++i)
    {
        // Shift for numerical stability
        double   row_max = x.row(i).maxCoeff();
        VectorXd shifted = x.row(i).transpose().array() - row_max;

        // exponentiate
        VectorXd exp_row = shifted.array().exp();

        // normalize
        double sum_exp = exp_row.sum();
        probs.row(i)   = (exp_row / sum_exp).transpose();
    }

    if (!derivative)
        return probs;

    // simplified derivative (diagonal approximation)
    return probs.array() * (1.0 - probs.array());
}
