/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include <cmath>

#include <iostream>
#include <memory>

#include <QApplication>

#include "ui/MainWindow.h"

#include "regressions/LinearRegression.h"
#include "regressions/Logistic2ParamRegression.h"
#include "regressions/LogisticRegression.h"
#include "regressions/PolynomialRegression.h"

using namespace regression;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    std::shared_ptr<IRegression> regression = std::make_shared<Logistic2ParamRegression>();

    std::vector<std::vector<IRegression::PointF>> points;
    points = regression->generateData();
    constexpr int LEFT_INDEX = 0;
    constexpr int RIGHT_INDEX = 1;
    std::cerr << "Train 2D logistical model" << std::endl;
    std::vector<float> coeffs = regression->train({points[LEFT_INDEX], points[RIGHT_INDEX]}, false);

    MainWindow::Data data1;
    data1.mColor = Qt::red;
    for (auto i = 0; i < points[LEFT_INDEX].size(); i++) {
        data1.mData.emplace_back(points[LEFT_INDEX][i].x, points[LEFT_INDEX][i].y);
    }

    MainWindow::Data data2;
    data2.mColor = Qt::green;
    for (auto i = 0; i < points[RIGHT_INDEX].size(); i++) {
        data2.mData.emplace_back(points[RIGHT_INDEX][i].x, points[RIGHT_INDEX][i].y);
    }


    MainWindow::Data data3;
    data3.mColor = Qt::blue;
    std::cerr << "Check the points" << std::endl;
    std::vector<IRegression::PointF> pointsA;
    for (auto test1 = 0.0f; test1 < 10.0f; test1 += 0.1f) {
        for (auto test2 = 0.0f; test2 < 10.0f; test2 += 0.1f) {
            //auto z = sigmoid(-test2 * coeffs[2] - test1 * coeffs[1] - coeffs[0]);
            auto z = sigmoid(test2 * coeffs[2] + test1 * coeffs[1] + coeffs[0]);
            if (abs(z - 0.5) < 0.01) {
                IRegression::PointF point;
                QPoint qpoint;
                point.x = test1;
                point.y = test2;
                qpoint.setX(test1);
                qpoint.setY(test2);
                pointsA.push_back(point);
                data3.mData.push_back(qpoint);
            }
        }
    }

    std::cerr << "Train linear regression for " << pointsA.size() << " points" << std::endl;
    std::shared_ptr<IRegression> regression2 = std::make_shared<LinearRegression>();
    auto coeffs2 = regression2->train({pointsA});
    MainWindow::Function function;
    function.mColor = Qt::blue;
    function.mFunction = [&coeffs2, &regression2](double x) -> double {
        return regression2->function(coeffs2, x);
    };

    std::cerr << "Output result" << std::endl;

    w.appendData(data1);
    w.appendData(data2);
    //w.appendData(data3);
    w.appendFunction(function);
    w.adjust();

    w.resize(640, 480);

    w.show();
    return a.exec();
}
