#include "visualizer.hpp"

    // Helper to generate the X-axis (epochs) based on data size
std::vector<double> Visualizer::get_epochs(size_t size) {
    std::vector<double> epochs(size);
    std::iota(epochs.begin(), epochs.end(), 1.0); // 1.0, 2.0, ...
    return epochs;
}

void Visualizer::plot_metric(const std::string& title, const std::vector<double>& data, 
                            const std::string& ylabel, const std::string& color)
{  
    std::vector<double> epochs = get_epochs(data.size());

    plt::figure();
    plt::named_plot(title, epochs, data, color);
    plt::title(title);
    plt::xlabel("Epoch");
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

void Visualizer::double_plot_metric(const std::string& title, 
    const std::pair<std::vector<double>, std::vector<double>> &data, 
    const std::string& ylabel,
    std::pair<std::string, std::string>colors)
{
    std::vector<double> epochs = get_epochs(data.first.size());
    plt::figure();
    plt::plot(epochs, data.first,  {{"color", colors.first}});
    plt::plot(epochs, data.second, {{"color", colors.second}, {"linestyle","--"}});
    plt::title(title);
    plt::xlabel("Epoch");
    plt::ylabel(ylabel);
    plt::grid(true);
    // plt::legend();
}

void Visualizer::show() {
    plt::show();
}
