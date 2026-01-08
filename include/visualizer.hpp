#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "matplotlibcpp.h"
#include <vector>
#include <string>
#include <numeric>
#include <utility>

namespace plt = matplotlibcpp;

class Visualizer {
public:
    // Helper to generate the X-axis (epochs) based on data size
    static std::vector<double> get_epochs(size_t size);

    // Generic method to plot any metric from your History map
    static void plot_metric(const std::string& title, 
                            const std::vector<double>& data, 
                            const std::string& ylabel, 
                            const std::string& color);

    static void double_plot_metric(const std::string& title, 
    const std::pair<std::vector<double>, std::vector<double>> &data, 
    const std::string& ylabel,
    std::pair<std::string, std::string>colors);

    static void show();
};

#endif