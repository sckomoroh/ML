/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "GnuPlot.h"

#include <numeric>
#include <sstream>

namespace gnuplot {

GnuPlot::GnuPlot()
{
    mPipe = popen("gnuplot -persist", "w");
    fflush(mPipe);
}

GnuPlot::~GnuPlot() { fclose(mPipe); }

void GnuPlot::setXAxisName(const char* name)
{
    fprintf(mPipe, "set xlabel '%s'\n", name);
    fprintf(mPipe, "replot\n");
}

void GnuPlot::setYAxisName(const char* name)
{
    fprintf(mPipe, "set ylabel '%s'\n", name);
    fprintf(mPipe, "replot\n");
}

void GnuPlot::plot(const PointsSet& points,
                   EPlotType plotType,
                   const std::string& color,
                   const std::string& name)
{
    auto plotCommand = buildGnuPlotCommand(plotType, color, name);

    mPoints.emplace_back(plotCommand, points);

    performPlot();
}

std::string GnuPlot::buildGnuPlotCommand(EPlotType plotType,
                                         const std::string& color,
                                         const std::string& name)
{
    std::stringstream ss;
    std::string type;
    switch (plotType) {
    case EPlotType::Lines:
        type = "lines";
        break;
    case EPlotType::Points:
        type = "points";
        break;
    }

    std::string title = "notitle";
    if (!name.empty()) {
        title = "title '" + name + "'";
    }

    ss << "'-' with " << type << " pt 7 lc rgb '" << color << "' " << title;

    return ss.str();
}

void GnuPlot::performPlot()
{
    std::string plotCommands = "plot ";
    plotCommands += std::accumulate(mPoints.begin(), mPoints.end(), std::string(),
                                    [](const std::string& acc, const PointsData& data) {
                                        return acc.empty() ? data.gnuplotCommand
                                                           : acc + ", " + data.gnuplotCommand;
                                    }) +
                    "\n";
    fprintf(mPipe, "%s", plotCommands.c_str());

    for (auto& pointsSet : mPoints) {
        for (auto point : pointsSet.pointSet) {
            fprintf(mPipe, "%f %f\n", point[0], point[1]);
        }
        fprintf(mPipe, "e\n");
    }

    fflush(mPipe);
}

}  // namespace gnuplot
