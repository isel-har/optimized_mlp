#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "matplotlibcpp.h"
#include <vector>
#include <string>
#include <numeric>
#include <utility>
#include "history.hpp"

namespace plt = matplotlibcpp;

typedef struct {
    std::string         title;
    std::vector<double> data;
    std::string         color;
    std::string         linestyle;
} PlotData;


class Visualizer {
public:
    static std::vector<double>  get_epochs(size_t size);

    static void multi_plots(const std::vector<PlotData>&plots, std::string ylabel); // multi curves 1 figure
    static void multi_figures(const std::vector<std::vector<PlotData>>&figures, std::vector<std::string>ylabels);
    static void show();
};

#endif