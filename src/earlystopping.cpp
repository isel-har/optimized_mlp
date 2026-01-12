#include "earlystopping.hpp"

EarlyStopping::EarlyStopping(bool enabled)
    : _enabled(enabled), _patience(6), optimal_loss(std::numeric_limits<double>::max())
{
}
EarlyStopping::EarlyStopping(char patience, bool enabled)
    : _enabled(enabled), _patience(patience), optimal_loss(std::numeric_limits<double>::max())
{
}

bool EarlyStopping::operator()(double loss)
{
    if (!_enabled)
        return false;
    if (loss < optimal_loss)
    {
        this->optimal_loss = loss;
        this->times        = 0;
    }
    else
        ++this->times;
    if (this->times >= this->_patience)
        return true;
    return false;
}