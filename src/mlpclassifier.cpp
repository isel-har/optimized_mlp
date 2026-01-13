#include "mlpclassifier.hpp"

std::unordered_map<std::string, Metric*> MLPClassifier::metricsMap = {
    {"accuracy", new Accuracy()},
    {"precision", new Precision()},
    {"loss", new BinarycrossEntropy()}
};

MLPClassifier::MLPClassifier(const json& conf) : built(false), earlystopping(false)
{
    this->confptr = &conf;
}

MLPClassifier::~MLPClassifier()
{
    for (size_t i = 0; i < this->metrics.size(); ++i)
    {
        this->metrics[i].second = nullptr;
    }
}

void MLPClassifier::train_val_metrics(unsigned int epoch, const t_split& dataset, History& history)
{
    MatrixXd ypred_train = this->feed(dataset.X_train);
    MatrixXd ypred_val   = this->feed(dataset.X_val);

    MatrixXd     ypmax_train = this->argmax(ypred_train);
    MatrixXd     ypmax_val   = this->argmax(ypred_val);
    unsigned int index       = epoch - 1;

    std::cout << "epoch " << epoch << '/' << this->epochs;

    double loss     = this->metricsMap["loss"]->compute(ypred_train, dataset.y_train);
    double loss_val = this->metricsMap["loss"]->compute(ypred_val, dataset.y_val);
    history.vecMap["loss"].first[index]  = loss;
    history.vecMap["loss"].second[index] = loss_val;

    for (auto& [name, vec] : history.vecMap)
    {
        if (name != "loss")
        {
            double metric_train     = this->metricsMap[name]->compute(ypmax_train, dataset.y_train);
            double metric_val       = this->metricsMap[name]->compute(ypmax_val, dataset.y_val);
            vec.first.at(index) = metric_train;
            vec.second.at(index) = metric_val;
        }
    }

    std::cout << "- loss:" << loss;
    for (const auto& metric : this->metrics)
    {
        double metric_ = history.vecMap[metric.first].first.at(index);
        std::cout << " - " << metric.first << ':' << metric_;
    }

    std::cout << " | val metric:";
    std::cout << "- loss:" << loss_val;
    for (const auto& metric : this->metrics)
    {
        double metric_ = history.vecMap[metric.first].second.at(index);
        std::cout << " - " << metric.first << ':' << metric_;
    }
    std::cout << std::endl;
}

std::vector<json> MLPClassifier::default_layers()
{
    std::vector<json> jlayers;
    json              hidden;

    hidden["size"]       = 16;
    hidden["activation"] = "relu";
    jlayers.push_back(hidden);
    hidden["size"] = 8;
    jlayers.push_back(hidden);
    hidden["activation"] = "softmax";
    hidden["size"] = 2;
    jlayers.push_back(hidden);
    return jlayers;
}

void MLPClassifier::build(unsigned int shape)
{
    if (this->confptr == nullptr)
        throw std::runtime_error("config object required to build.");

    const json conf = *this->confptr;
    this->input_shape            = shape;// to fix!

    double      learning_rate    = checked_range(conf.value("learning_rate", 0.01), 0.01, 0.1, "learning_rate");
    this->epochs                 = checked_range(conf.value("epochs", 10), 1, 100, "epochs");
    this->batch_size             = checked_range(conf.value("batch_size", 32), 1, 256, "batch_size"); 
    this->earlystopping._enabled = conf.value("early_stopping", false);
    this->optimizer              = conf.value("adam_optimizer", false)? new GradientDescent(learning_rate):new GradientDescent(learning_rate);

    std::vector<std::string> metrics = conf.value("metrics", std::vector<std::string>({}));
    checked_range(metrics.size(), (size_t)0, (size_t)4, "metrics_size");

    auto unique_metrics = std::unordered_set<std::string>(metrics.begin(), metrics.end());
    
    for (const auto& metric : unique_metrics)
    {
        if (MLPClassifier::metricsMap.find(metric) != MLPClassifier::metricsMap.end())
        {
            this->metrics.push_back(std::make_pair(metric, MLPClassifier::metricsMap[metric]));
        }
    }
    std::vector<json> layers_json = conf.value("layers", this->default_layers()); // default two hidden layers + output layer
    checked_range(layers_json.size(), (size_t)2, (size_t)11, "layers_stack_size");
    checked_layers(layers_json);
    
    this->layers.emplace_back(shape, layers_json[0]["size"],
                              layers_json[0]["activation"]);
    for (size_t i = 1; i < layers_json.size(); ++i)
    {
        unsigned int shape = layers_json[i - 1]["size"];
        this->layers.emplace_back(shape, layers_json[i]["size"], layers_json[i]["activation"]);
    }
    this->built = true;
}

MatrixXd MLPClassifier::feed(const MatrixXd& x)
{
    MatrixXd feed = layers[0].forward(x);
    for (size_t i = 1; i < this->layers.size(); ++i)
    {
        feed = layers[i].forward(feed);
    }
    return feed;
}

void MLPClassifier::backward(const MatrixXd& dl_out)
{
    int      last  = (int) this->layers.size() - 1;
    MatrixXd dloss = dl_out;
    for (; last >= 0; --last)
    {
        dloss = this->layers[last].backward(dloss);
    }
}

MatrixXd MLPClassifier::argmax(const MatrixXd& y_probs) const
{
    MatrixXd result = MatrixXd::Zero(y_probs.rows(), y_probs.cols());
    for (size_t i = 0; i < (size_t) y_probs.rows(); ++i)
    {
        size_t index     = (y_probs(i, 0) > y_probs(i, 1)) ? 0 : 1;
        result(i, index) = 1.0f;
    }
    return result;
}

History MLPClassifier::fit(const t_split& dataset)
{
    if (!this->built)
        throw std::runtime_error("build required before training phase.");

    if (this->input_shape != (size_t) dataset.X_train.cols())
        throw std::runtime_error("input shape must be equal to given input cols");

    History history(this->epochs);
    unsigned int e = 1;
    double      loss_e = 0.0;

    while (e <= epochs && !earlystopping(loss_e))
    {
        for (unsigned int i = 0; i < (unsigned int) dataset.X_train.rows(); i += batch_size)
        {
            unsigned int end = std::min(i + batch_size, (unsigned int) dataset.X_train.rows());
    
            MatrixXd xbatch = dataset.X_train.middleRows(i, end - i);
            MatrixXd ybatch = dataset.y_train.middleRows(i, end - i);
    
            MatrixXd probs = this->feed(xbatch);
            MatrixXd loss  = (probs.array() - ybatch.array()).matrix();
    
            this->backward(loss);
            this->optimizer->update(this->layers);
        }
        this->train_val_metrics(e, dataset, history);
        loss_e = history.vecMap["loss"].second[e - 1];
        ++e;
    }
    return history;
}

void MLPClassifier::save(const std::string &name) const
{
    std::ofstream file(name, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open file for saving model.");

    size_t total_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&total_layers), sizeof(total_layers));

    for (const auto& layer : layers)
    {
        unsigned int size = layer.size; // number of neurons in this layer
        unsigned int rows = (unsigned int) layer.weights.rows();
        unsigned int cols = (unsigned int) layer.weights.cols();
        // Write per-layer metadata
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

        size_t activation_length = layer.activation__.length();
        file.write(reinterpret_cast<const char*>(&activation_length), sizeof(activation_length));
        file.write(layer.activation__.c_str(), activation_length);

        for (unsigned int i = 0; i < cols; ++i)
        {
            for (unsigned int j = 0; j < rows; ++j)
            {
                file.write(reinterpret_cast<const char*>(&layer.weights(j, i)), sizeof(double));
            }
        }
        for (unsigned int i = 0; i < size; ++i)
        {
            file.write(reinterpret_cast<const char*>(&layer.biases(0, i)), sizeof(double));
        }
    }
    std::cout << '\'' << name + "\' saved\n";
}
