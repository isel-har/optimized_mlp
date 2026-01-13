#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "history.hpp"
#include "matplotlibcpp.h"

#include <numeric>
#include <string>
#include <iomanip>
#include <sstream>
#include <random>

namespace plt = matplotlibcpp;

class PlotData
{
  public:
    std::string               title;
    std::vector<double>       data;
    std::string               linestyle;

    static std::string randomHexColor();
    PlotData(const std::string&, const std::vector<double>&, const std::string&);
  };
  
class Visualizer
{
  public:
  static std::vector<double> get_epochs(size_t size);

    static void multi_plots(const std::vector<PlotData>& plots,
                            std::string                  ylabel); // multi curves 1 figure
    static void multi_figures(const std::vector<std::vector<PlotData>>& figures,
                              std::vector<std::string>                  ylabels);
    
    // static std::string rgb_hex(const PlotData::Color &color);
    static void show();
};

#endif