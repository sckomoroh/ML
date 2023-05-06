#include "MainWindow.h"

#include <QDebug>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
}

MainWindow::~MainWindow() {}

void MainWindow::setBounds(double left, double top, double right, double bottom)
{
    mTop = top;
    mBottom = bottom;
    mLeft = left;
    mRight = right;
}

void MainWindow::setSteps(double horizontal, double vertical)
{
    mHorizontalStep = horizontal;
    mVerticalStep = vertical;
}

void MainWindow::setData(QList<QPointF> data) { mOriginalData = data; }

void MainWindow::appendFunction(Function function) { mFunctions.push_back(function); }

void MainWindow::calculate()
{
    mData.clear();

    double hk = ((double)QMainWindow::width()) / (mRight - mLeft);
    double vk = ((double)QMainWindow::height()) / (mTop - mBottom);
    double step = (mRight - mLeft) / (double)(QMainWindow::height());

    for (auto& function : mFunctions) {
        function.mData.clear();
        for (double i = mLeft; i <= mRight; i = i + step) {
            auto value = -function.mFunction(i);
            auto point = QPointF{(i - mLeft) * hk, (value + mTop) * vk};
            function.mData.push_back(point);
        }
    }

    for (auto point : mOriginalData) {
        auto pt = QPointF{(point.x() - mLeft) * hk, (-point.y() + mTop) * vk};
        mData.push_back(pt);
    }
}

void MainWindow::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.setPen(Qt::black);
    double hk = ((double)QMainWindow::width()) / (mRight - mLeft);
    double vk = ((double)QMainWindow::height()) / (mTop - mBottom);
    if (mLeft < 0 && mRight > 0) {
        painter.drawLine(-mLeft * hk, 0, -mLeft * hk, height());
    }
    else {
        qDebug() << "Horizontal axis is out of range";
    }

    if (mBottom < 0 && mTop > 0) {
        painter.drawLine(0, mTop * vk, width(), mTop * vk);
    }
    else {
        qDebug() << "Vertical axis is out of range";
    }

    QPen pen = const_cast<QPen&>(painter.pen());
    pen.setStyle(Qt::DashLine);
    pen.setColor(Qt::gray);
    painter.setPen(pen);

    if (mHorizontalStep > 0.01) {
        for (double i = mHorizontalStep; i < mRight; i = i + mHorizontalStep) {
            auto pos = i * hk - mLeft * hk;
            painter.drawLine(pos, 0, pos, height());
        }

        for (double i = -mHorizontalStep; i > mLeft; i = i - mHorizontalStep) {
            auto pos = i * hk - mLeft * hk;
            painter.drawLine(pos, 0, pos, height());
        }
    }
    else {
        qDebug() << "Incorrect horizontal step";
    }

    if (mVerticalStep > 0.01) {
        for (double i = mTop - mVerticalStep; i > 0; i = i - mVerticalStep) {
            auto pos = i * vk;
            painter.drawLine(0, pos, width(), pos);
        }

        for (double i = mTop + mVerticalStep; i < -mBottom + mTop; i = i + mVerticalStep) {
            auto pos = i * vk;
            painter.drawLine(0, pos, width(), pos);
        }
    }
    else {
        qDebug() << "Incorrect vertical step";
    }

    pen = const_cast<QPen&>(painter.pen());
    pen.setStyle(Qt::SolidLine);
    pen.setColor(Qt::red);
    pen.setWidth(2);
    painter.setPen(pen);

    calculate();

    painter.drawPoints(QPolygonF{mData});

    pen = const_cast<QPen&>(painter.pen());
    pen.setStyle(Qt::SolidLine);
    pen.setColor(Qt::red);
    pen.setWidth(2);
    painter.setPen(pen);

    for (auto function : mFunctions) {
        pen = const_cast<QPen&>(painter.pen());
        pen.setStyle(Qt::SolidLine);
        pen.setColor(function.mColor);
        pen.setWidth(1);
        painter.setPen(pen);

        painter.drawPolyline(function.mData.data(), function.mData.size());
    }
}
