/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <functional>
#include <list>

#include <QMainWindow>
#include <QPainter>
#include <QWidget>

struct Function {
    std::function<double(double)> mFunction;
    QList<QPointF> mData;
    QColor mColor;
};

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
    std::list<Function> mFunctions;

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

public:
    void setData(QList<QPointF> data);
    void appendFunction(Function function);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void setBounds(double left, double top, double right, double bottom);
    void calculate();    
    void setSteps(double horizontal, double vertical);
};
#endif  // MAINWINDOW_H