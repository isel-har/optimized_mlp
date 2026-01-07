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
        json conf = load_json(argv[2]);
        MatrixXd X_scaled;
        MatrixXd Y_encoded;
        {
            xy_eigen xy = csv_to_eigen(conf[op]["data"]);
            X_scaled    = StandardScaler(xy.X);
            Y_encoded   = xy.Y;
        }
        if (op == "train") {
            auto model = MLPClassifier(conf["train"]);
            model.build();
            History history = model.fit(X_scaled, Y_encoded);
            Visualizer::plot_metric("Loss per epoch", history.loss, "loss", "red");
            Visualizer::show();
        }
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "error :" << e.what() << '\n';
    }
    return 1;
}