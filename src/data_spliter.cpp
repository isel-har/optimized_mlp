#include "data_spliter.h"


std::vector<std::string> csv_to_rawstrs(const char *csvpath)
{
    std::string                 row;
    std::ifstream               file(csvpath);
    std::vector<std::string>    rawdata;

    if (!file.is_open())
        throw std::exception();// file error exception

    //can reserve here!
    while (getline(file, row)) 
        rawdata.push_back(row);

    file.close();
    return rawdata;
}

void    shuffle_rows(std::vector<std::string> *rowsptr) {

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(rowsptr->begin(), rowsptr->end(), g);
}


std::pair<std::vector<std::string>, std::vector<std::string>> split_rows(std::vector<std::string> *rowsptr, float test_size) {
    
    std::pair<std::vector<std::string>, std::vector<std::string>> splited_pair;

    // if (randomize == true)
    shuffle_rows(rowsptr);

    size_t train_size_ = static_cast<size_t>(rowsptr->size() * test_size);
    size_t i = 0;
    while (i < rowsptr->size() - train_size_)  {
        splited_pair.first.push_back(rowsptr->at(i));
        ++i;
    }
    size_t j = 0;
    while (j < train_size_) {
        splited_pair.second.push_back(rowsptr->at(i));
        ++j;
        ++i;
    }
    return splited_pair;
}

void    save_splitted_data(const std::string &path, std::pair<std::vector<std::string>, std::vector<std::string>> *splitted_data)
{
    std::ofstream   trainf(path + "data_train.csv");
    std::ofstream   valf(path + "data_val.csv");
    
    size_t i = 0;
    while (i < splitted_data->first.size())
    {
        trainf << splitted_data->first.at(i) + "\n";
        ++i;
    }
    i = 0;
    while (i < splitted_data->second.size())
    {
        valf << splitted_data->second.at(i) + "\n";
        ++i;
    }
    std::cout <<"data_train.csv and data_val.csv are saved.\n";
}
