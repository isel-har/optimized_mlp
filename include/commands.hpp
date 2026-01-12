#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <iostream>
#include <string>
#include <vector>

#include "train_val_split.hpp"
#include "mlpclassifier.hpp"
#include "json_loader.hpp"
#include "visualizer.hpp"
#include "csv_split.hpp"


int     cmd_split();
int     cmd_train(const char* config_path);
// int     cmd_predict(const char *model_path);
void    print_usage(const char* prog);

#endif
