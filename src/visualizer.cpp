#include "visualizer.hpp"

std::vector<double> Visualizer::get_epochs(size_t size)
{
    std::vector<double> epochs(size);
    std::iota(epochs.begin(), epochs.end(), 1.0);
    return epochs;
}

void    Visualizer::multi_plots(const std::vector<PlotData>&plots, std::string ylabel)
{
    if (!plots.size() || !plots[0].data.size())
        throw std::runtime_error("size of plots and data cannot be 0.");

    std::vector<double> epochs{get_epochs(plots[0].data.size())};
    plt::figure();
    for (const auto &plot:plots)
        plt::plot(epochs, plot.data, {{"color", plot.color}, {"linestyle", plot.linestyle}});

    plt::xlabel("epoch");
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

void    Visualizer::multi_figures(const std::vector<std::vector<PlotData>>&figures, std::vector<std::string>ylabels) {

    if (figures.size() != ylabels.size())
        throw std::runtime_error("number of y labels should be equal to figures.");

    for (size_t i = 0; i < figures.size(); ++i) {
        Visualizer::multi_plots(figures[i], ylabels[i]);
    }
}

void Visualizer::show() {
    plt::show();
}
