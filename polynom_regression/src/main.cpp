#include "MainWindow.h"

#include <QApplication>
#include "logic.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    std::vector<float> X;
    std::vector<float> Y;
    generateInputData(X, Y);
    calculate(X, Y);

    w.setBounds(-1.5, 20, 1.5, -5);

    QList<QPointF> points;
    for (auto i=0; i<X.size(); i++) {
        points.emplace_back(X[i], Y[i]);
    }

    w.setData(points);

    //w.setFunction([](double x) { return x * x * x; });

    w.resize(640, 480);

    w.show();
    return a.exec();
}
