#include "MainWindow.h"

#include <QApplication>

#include "logic_linear.h"
#include "logic_polynominal.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    std::vector<float> X;
    std::vector<float> Y;
    polynominal::generateData(X, Y);
    std::vector<float> coeffs = polynominal::calculate(X, Y);
    //float k = linear::calculate(X, Y);

    w.setBounds(-1.5, 20, 1.5, -15);

    QList<QPointF> points;
    for (auto i = 0; i < X.size(); i++) {
        points.emplace_back(X[i], Y[i]);
    }

    Function function;
    function.mColor = QColor("blue");
    function.mFunction = std::function<double(double)>{[&coeffs](double x) {
        float value = 0.0f;
        for (int j = 0; j < coeffs.size(); ++j) {
            value += coeffs[j] * std::pow(x, j);
        }

        return value;
    }};

    //function.mFunction = std::function<double(double)>{[&k](double x) { return k * x; }};

    w.appendFunction(function);
    w.setData(points);

    w.resize(640, 480);

    w.show();
    return a.exec();
}
