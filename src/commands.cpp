#include "commands.hpp"

void    print_usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " split\n"
        << "  " << prog << " train <config.json>\n"
        << "  " << prog << " test  <config.json>\n";
}

int cmd_split()
{
    auto records        = csv_to_rawstrs("data/data.csv");
    auto train_val_pair = split_rows(&records, 0.2f);
    save_split_data("data/", &train_val_pair);
    return EXIT_SUCCESS;
}

int cmd_train(const char* config_path)
{
    json conf = load_json(config_path);

    t_split datasplit =
        train_val_split(conf["data"]["train"], conf["data"]["val"]);

    std::vector<MLPClassifier> models;
    std::vector<History> histories;

    if (!conf.contains("models") || conf["models"].empty())
        models.emplace_back(conf);
    else
        for (const auto& jmodel : conf["models"])
            models.emplace_back(jmodel);

    const unsigned int input_shape =
        static_cast<unsigned int>(datasplit.X_train.cols());

    for (size_t i = 0; i < models.size(); ++i)
    {
        std::cout << "model ______________[" << i + 1 << "]______________\n";

        models[i].build(input_shape);
        histories.push_back(models[i].fit(datasplit));
        models[i].save("model_" + std::to_string(i + 1) + ".bin");
    }

    return EXIT_SUCCESS;
}
// int cmd_predict(const char *model_path)
// {
//     return 1;
// }