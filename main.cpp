#include "csv_to_eigen.hpp"
#include "json_loader.hpp"
#include "data_spliter.h"
#include "scaler.hpp"
#include "mlpclassifier.hpp"
#include "visualizer.hpp"


int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout<<"usage: "<<argv[0]<<" [operations (train, test, split)]\n"; 
        return 0;
    }
    try
    {
        std::string op(argv[1]);
        if (op == "split") {
            auto records = csv_to_rawstrs("data/data.csv");
            auto train_test_pair = split_rows(&records, 0.2f);
            save_splitted_data("data/", &train_test_pair);
            return 0;
        }
        if (op != "train" && op != "test") {
           std::cerr << "usage: "<< argv[0] <<" [operations (train, test, split)]\n"; 
           return 1;
        }
        if (!argv[2]) {
            std::cerr << "config file required for (train/test) operations\n"; 
            return 1;
        }
        if (op == "train") {
            json conf = load_json(argv[2]);
            t_split datasplit;
            {
                std::pair<MatrixXd, MatrixXd> train_dataset = csv_to_eigen(conf["data"]["train"]);
                std::pair<MatrixXd, MatrixXd> val_dataset   = csv_to_eigen(conf["data"]["val"]);

                datasplit.X_train = StandardScaler(train_dataset.first);
                datasplit.y_train = train_dataset.second;
                datasplit.X_val   = StandardScaler(val_dataset.first);
                datasplit.y_val   = val_dataset.second;
            }
            std::vector<MLPClassifier>  models;
            std::vector<History>        histories;

            for (auto& jmodel:conf["models"]) {
                models.emplace_back(jmodel);
            }
            for (size_t i = 0; i < models.size(); ++i) {
                models[i].build();
                std::cout << "model ______________["<< i + 1 <<"]______________\n";
                histories.push_back(models[i].fit(datasplit));
            }
            // std::vector<std::vector<PlotData>>      figures;
            // std::vector<std::vector<std::string>>   ylabels;

            // for (const auto &history:histories) {

            //     // for (const auto&[name, vptr]:history.vecMap) { // iterate over the map!

            //     // }
            // }
            // Visualizer::multi_figures(figures, ylabels);
            // Visualizer::show();
        }
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "error :" << e.what() << '\n';
    }
    return 1;
}
