/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include <cmath>

#include <iostream>
#include <memory>

#include <QApplication>

#include "ui/MainWindow.h"

#include "classification/Classification.h"

using namespace classification;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    std::vector<MainWindow::Data> outData(3);

    Classification classification;
    auto data = classification.generateData();
    auto input = classification.prepareData(data);
    auto res = classification.trainig(input);

    auto test = classification.generateData();
    auto testInput = classification.prepareData(test);
    classification.verify(testInput, res);

    int index = 0;
    for (auto& item : data) {
        for (auto j=0; j<item.second.size(); j++) {
            outData[index].mData.emplace_back(item.second[j], index);
        }
        index++;
    }

    outData[0].mColor = Qt::red;
    outData[1].mColor = Qt::blue;
    outData[2].mColor = Qt::green;
    // MainWindow::Data data1;
    // data1.mColor = Qt::red;
    // for (auto i = 0; i < points[0].size(); i++) {
    //     data1.mData.emplace_back(points[0][i].x, points[0][i].y);
    // }

    std::cerr << "Output result" << std::endl;

    w.appendData(outData[0]);
    w.appendData(outData[1]);
    w.appendData(outData[2]);
    w.adjust();

    w.resize(640, 480);

    w.show();
    return a.exec();
}
