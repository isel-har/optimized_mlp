#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include <exception>
#include <algorithm>
#include <unordered_set>
#include "optimizers.hpp"
#include "metrics.hpp"
#include "history.hpp"
#include "layer.hpp"
#include "json.hpp"
#include <fstream>
#include <utility>
#include "csv_to_eigen.hpp"

using json = nlohmann::json;

class MLPClassifier {
private:
    static std::unordered_map<std::string, Metric*> metricsMap;

    unsigned int        epochs;
    unsigned int        batch_size;
    unsigned int        input_shape;
    bool                built;

    std::vector<std::pair<std::string, Metric*>> metrics;
    std::vector<Layer>                          layers;
    
    Optimizer   *optimizer = nullptr;
    const json  *confptr   = nullptr;

    MatrixXd    feed(const MatrixXd&);
    void        backward(const MatrixXd&);

public:
    MLPClassifier();
    MLPClassifier(const json&);
    ~MLPClassifier();

    void    load();
    void    save() const;
    void    build(void);
    History fit(const t_split &);

    std::vector<json>  default_layers();

    MatrixXd    predict(const MatrixXd&);
    MatrixXd    argmax(const MatrixXd &) const;

    void        train_val_metrics(unsigned int epoch, const t_split &dataset, History &history);
};

static std::unordered_map<std::string, Metric*> metricsMap;

#endif
