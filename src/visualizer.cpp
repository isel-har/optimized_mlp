#include "visualizer.hpp"

 PlotData::PlotData(const std::string& title, const std::vector<double>& data, const std::string& linestyle):
    title(title), data(data), linestyle(linestyle)
{}

std::vector<double> Visualizer::get_epochs(size_t size)
{
    std::vector<double> epochs(size);
    std::iota(epochs.begin(), epochs.end(), 1.0);
    return epochs;
}

std::string PlotData::randomHexColor() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0, 255);

    std::stringstream ss;
    ss << '#'
       << std::hex << std::setw(2) << std::setfill('0') << dist(rng)
       << std::setw(2) << dist(rng)
       << std::setw(2) << dist(rng);

    return ss.str();
}

void Visualizer::multi_plots(const std::vector<PlotData>& plots, std::string ylabel)
{
    if (plots.empty())
        throw std::runtime_error("Size of plots cannot be 0.");

    plt::figure();
    
    for (const auto& plot : plots)
    {
        if (plot.data.empty()) continue;

        std::vector<double> current_epochs = get_epochs(plot.data.size());
        std::string color = PlotData::randomHexColor();
        plt::plot(current_epochs, plot.data, {
            {"color", color}, 
            {"linestyle", plot.linestyle}, 
            {"label", plot.title}
        });
    }

    plt::xlabel("epoch");
    plt::ylabel(ylabel);
    plt::grid(true);
    plt::legend();
}

void Visualizer::multi_figures(const std::vector<std::vector<PlotData>>& figures,
                               std::vector<std::string>                  ylabels)
{
    if (figures.size() != ylabels.size())
        throw std::runtime_error("number of y labels should be equal to figures.");

    for (size_t i = 0; i < figures.size(); ++i)
    {
        Visualizer::multi_plots(figures[i], ylabels[i]);
    }
}

void Visualizer::show()
{
    plt::show();
}
