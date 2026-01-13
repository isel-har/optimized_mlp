#ifndef SEQUENTIAL_HPP
#define SEQUENTIAL_HPP

#include "csv_to_eigen.hpp"
#include "earlystopping.hpp"
#include "history.hpp"
#include "json.hpp"
#include "layer.hpp"
#include "metrics.hpp"
#include "optimizers.hpp"
#include "checker.tpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;

class MLPClassifier
{
  private:
  
  static std::unordered_map<std::string, Metric*> metricsMap;
  unsigned int epochs;
  unsigned int batch_size;
  unsigned int input_shape;
  bool         built;
  
  std::vector<Layer>                           layers;
  
  Optimizer*    optimizer = nullptr;
  const json*   confptr   = nullptr;
  EarlyStopping earlystopping;
  
  MatrixXd feed(const MatrixXd&);
  void     backward(const MatrixXd&);
  
  public:
    std::vector<std::pair<std::string, Metric*>> metrics;
    MLPClassifier();
    MLPClassifier(const json&);
    ~MLPClassifier();

    void    load();
    void    save(const std::string &name) const;
    void    build(unsigned int);
    History fit(const t_split&);

    std::vector<json> default_layers();

    MatrixXd predict(const MatrixXd&);
    MatrixXd argmax(const MatrixXd&) const;

    void train_val_metrics(unsigned int epoch, const t_split& dataset, History& history);
};

static std::unordered_map<std::string, Metric*> metricsMap;

#endif
