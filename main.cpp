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
            json conf = load_json(argv[2])["training"];
            t_split datasplit;
            {
                std::pair<MatrixXd, MatrixXd> train_dataset = csv_to_eigen(conf["data"]["train"]);
                std::pair<MatrixXd, MatrixXd> val_dataset   = csv_to_eigen(conf["data"]["val"]);

                datasplit.X_train = StandardScaler(train_dataset.first);
                datasplit.y_train = train_dataset.second;
                datasplit.X_val   = StandardScaler(val_dataset.first);
                datasplit.y_val   = val_dataset.second;
            }
            /*
                allocate sizes
            */
            std::vector<MLPClassifier>  models;
            std::vector<History>        hisotries;

            for (const auto &mobj: conf["models"]) {
                models.emplace_back(mobj);
            }
            
            hisotries.reserve(models.size());
            for (auto&model:models){
                model.build();
                hisotries.push_back(model.fit(datasplit));
            }

            // Visualizer::double_plot_metric("train & validation loss per epoch", history.loss_pair, "loss", {"red", "black"});
            // Visualizer::double_plot_metric("train & validation accuracy per epoch", history.accuracy_pair, "accuracy", {"yellow", "blue"});
            // Visualizer::show();

        }
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "error :" << e.what() << '\n';
    }
    return 1;
}