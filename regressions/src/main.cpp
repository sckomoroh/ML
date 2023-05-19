/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include <cmath>

#include <iostream>
#include <memory>

#include <QApplication>

#include "ui/MainWindow.h"

// #define LINEAR
// #define POLY
// #define LOGISTIC
#define LOGISTIC2D

#ifdef LINEAR
#include "regressions/LinearRegression.h"
#endif

#ifdef POLY
#include "regressions/PolynomialRegression.h"
#endif

#ifdef LOGISTIC
#include "regressions/LogisticRegression.h"
#endif

#ifdef LOGISTIC2D
#include "regressions/LinearRegression.h"
#include "regressions/Logistic2ParamRegression.h"
#endif

using namespace regression;

#ifdef LOGISTIC
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
#endif

#ifdef LOGISTIC2D
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
#endif

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow win;
#ifdef LINEAR
    linear::LinearRegression regression;
    MainWindow::Data inData{Qt::red};

    auto inputData = linear::generateData();
    for (auto i = 0; i < inputData.cols(); i++) {
        inData.mData.emplace_back(inputData(0, i), inputData(1, i));
    }

    auto weights = regression.train(inputData);
    MainWindow::Function function(Qt::blue, [&k = weights](double x) { return k(0) * x + k(1); });

    win.appendData(inData);
    win.appendFunction(function);
#endif

#ifdef POLY
    polynomial::PolynomialRegression regression;
    MainWindow::Data inData{Qt::red};
    auto inputData = polynomial::generateData();
    for (auto i = 0; i < inputData.cols(); i++) {
        inData.mData.emplace_back(inputData(0, i), inputData(1, i));
    }

    auto weights = regression.train(inputData);
    MainWindow::Function function(Qt::blue, [&k = weights](double x) {
        double res = 0;
        for (auto i = 0; i < k.rows(); i++) {
            res += k(i) * pow(x, i);
        }

        return res;
    });

    win.appendData(inData);
    win.appendFunction(function);
#endif

#ifdef LOGISTIC
    logistic::LogisticRegression regression;
    MainWindow::Data inData{Qt::red};

    auto inputData = logistic::generateData();
    for (auto i = 0; i < inputData.cols(); i++) {
        inData.mData.emplace_back(inputData(0, i), inputData(1, i));
    }

    auto weights = regression.train(inputData);
    MainWindow::Function function(Qt::blue,
                                  [&k = weights](double x) { return sigmoid(k(0) + x * k(1)); });

    win.appendData(inData);
    win.appendFunction(function);

#endif

#ifdef LOGISTIC2D

#define USE_REGRESSION
    logistic2d::Logistic2ParamRegression regression;
    MainWindow::Data inData1{Qt::red};
    MainWindow::Data inData2{Qt::green};

    auto inputData = logistic2d::generateData();

    for (auto i = 0; i < inputData.cols() / 2; i++) {
        inData1.mData.emplace_back(inputData(0, i), inputData(1, i));
    }

    for (auto i = inputData.cols() / 2; i < inputData.cols(); i++) {
        inData2.mData.emplace_back(inputData(0, i), inputData(1, i));
    }

    auto k = regression.train(inputData);

    Eigen::Vector<float, linear::POINTS_COUNT> x1 = Eigen::VectorXf::LinSpaced(linear::POINTS_COUNT, -6.0f, 6.0f);
    Eigen::Vector<float, linear::POINTS_COUNT> x2 = Eigen::VectorXf::LinSpaced(linear::POINTS_COUNT, -6.0f, 6.0f);

#ifdef USE_REGRESSION
    linear::InputMatrix lineMatrix;

    for (auto i = 0; i < x1.rows(); i++) {
        for (auto j = 0; j < x2.rows(); j++) {
            auto z = sigmoid(k[0] + x1(i) * k[1] + x2(j) * k[2]);
            if (abs(z - 0.5f) < 0.01f) {
                lineMatrix.conservativeResize(lineMatrix.rows(), lineMatrix.cols() + 1);
                lineMatrix(0, lineMatrix.cols() - 1) = x1(i);
                lineMatrix(1, lineMatrix.cols() - 1) = x2(j);
            }
        }
    }

    linear::LinearRegression lineRegression;
    auto lineK = lineRegression.train(lineMatrix);
    MainWindow::Function function(Qt::blue, [&k = lineK](double x) { return x * k(0) + k(1); });
    win.appendFunction(function);

#else
    MainWindow::Data inData3{Qt::blue};

    for (auto i = 0; i < x1.rows(); i++) {
        for (auto j = 0; j < x2.rows(); j++) {
            auto z = sigmoid(k[0] + x1(i) * k[1] + x2(j) * k[2]);
            if (abs(z - 0.5f) < 0.01f) {
                inData3.mData.emplace_back(x1(i), x2(j));
            }
        }
    }

    win.appendData(inData3);
#endif

    win.appendData(inData1);
    win.appendData(inData2);

#endif

    win.adjust();
    win.resize(640, 480);

    win.show();
    return a.exec();
}

#ifdef CLUSTERING
// Classification classification;
// auto data = classification.generateData();
// auto input = classification.prepareData(data);
// auto res = classification.trainig(input);

// auto test = classification.generateData();
// auto testInput = classification.prepareData(test);
// classification.verify(testInput, res);

// int index = 0;
// for (auto& item : data) {
//     for (auto j=0; j<item.second.size(); j++) {
//         outData[index].mData.emplace_back(item.second[j], index);
//     }
//     index++;
// }

// outData[0].mColor = Qt::red;
// outData[1].mColor = Qt::blue;
// outData[2].mColor = Qt::green;
// // MainWindow::Data data1;
// // data1.mColor = Qt::red;
// // for (auto i = 0; i < points[0].size(); i++) {
// //     data1.mData.emplace_back(points[0][i].x, points[0][i].y);
// // }

// std::cerr << "Output result" << std::endl;

// w.appendData(outData[0]);
// w.appendData(outData[1]);
// w.appendData(outData[2]);

#endif
