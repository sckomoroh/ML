/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <string>
#include <vector>

namespace gnuplot {

using VectorF = std::vector<float>;
using PointsSet = std::vector<VectorF>;

class GnuPlot final {
public:
private:
    struct PointsData {
        PointsData(const std::string& gnuplotCommand, const PointsSet& pointSet)
            : gnuplotCommand{gnuplotCommand}
            , pointSet{pointSet}
        {
        }

        std::string gnuplotCommand;
        PointsSet pointSet;
    };

private:
    FILE* mPipe;
    std::vector<PointsData> mPoints;

public:
    enum class EPlotType { Lines, Points };

public:
    GnuPlot();
    ~GnuPlot();

public:
    void setXAxisName(const char* name);

    void setYAxisName(const char* name);

    void plot(const PointsSet& points,
              EPlotType plotType,
              const std::string& color,
              const std::string& name = "");

private:
    std::string buildGnuPlotCommand(EPlotType plotType,
                                    const std::string& color,
                                    const std::string& name);

    void performPlot();
};

}  // namespace gnuplot
