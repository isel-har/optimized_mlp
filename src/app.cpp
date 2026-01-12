#include "app.hpp"

int run(int argc, char** argv)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string op = argv[1];

    if (op == "split")
        return cmd_split();

    if (op == "train" || op == "predict")
    {
        if (argc < 3)
        {
            std::cerr << "Error: config file required\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        if (op == "train")
            return cmd_train(argv[2]);
        // else
            // return cmd_predict(argv[2]);

        std::cerr << "Error: test not implemented yet\n";
        return EXIT_FAILURE;
    }

    std::cerr << "Error: unknown operation '" << op << "'\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
