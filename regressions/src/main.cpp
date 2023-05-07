/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include <memory>

#include <QApplication>

#include "ui/MainWindow.h"

#include "regressions/LinearRegression.h"
#include "regressions/PolynomialRegression.h"
#include "regressions/LogisticRegression.h"

using namespace regression;

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    std::shared_ptr<IRegression> regression = std::make_shared<PolynomialRegression>();

    std::vector<float> X;
    std::vector<float> Y;
    regression->generateData(X, Y);
    std::vector<float> coeffs = regression->train(X, Y, false);

    QList<QPointF> points;
    for (auto i = 0; i < X.size(); i++) {
        points.emplace_back(X[i], Y[i]);
    }

    Function function;
    function.mColor = QColor("blue");

    function.mFunction = std::function<double(double)>{
        [&coeffs, &regression](double x) { return regression->function(coeffs, x); }};

    w.appendFunction(function);
    w.setData(points);

    w.resize(640, 480);

    w.show();
    return a.exec();
}
