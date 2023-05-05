#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <functional>

#include <QMainWindow>
#include <QPainter>
#include <QWidget>

class MainWindow : public QMainWindow {
    Q_OBJECT
private:
    double mTop;
    double mBottom;
    double mLeft;
    double mRight;

    double mHorizontalStep;
    double mVerticalStep;

    QList<QPointF> mData;
    QList<QPointF> mOriginalData;
    std::function<double(double)> mFunction;

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

public:
    void setBounds(double left, double top, double right, double bottom);
    void setSteps(double horizontal, double vertical);
    void setData(QList<QPointF> data);
    void setFunction(std::function<double(double)> function);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void calculate();    
};
#endif  // MAINWINDOW_H
