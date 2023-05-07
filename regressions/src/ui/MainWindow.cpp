/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "MainWindow.h"

#include <algorithm>
#include <sstream>

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

void MainWindow::setData(QList<QPointF> data)
{
    mOriginalData = data;
    auto minX =
        std::min_element(data.begin(), data.end(), [](const QPointF& left, const QPointF& right) {
            return left.x() < right.x();
        });
    auto minY =
        std::min_element(data.begin(), data.end(), [](const QPointF& left, const QPointF& right) {
            return left.y() < right.y();
        });
    auto maxX =
        std::max_element(data.begin(), data.end(), [](const QPointF& left, const QPointF& right) {
            return left.x() < right.x();
        });
    auto maxY =
        std::max_element(data.begin(), data.end(), [](const QPointF& left, const QPointF& right) {
            return left.y() < right.y();
        });

    auto paddingX = abs(maxX->x() - minX->x()) / 10.0f;
    paddingX = (abs(maxX->x() - minX->x()) - (paddingX / 2)) / 10.0f;
    auto paddingY = abs(maxY->y() - minY->y()) / 10.0f;
    paddingY = (abs(maxY->y() - minY->y()) - (paddingY / 2)) / 10.0f;
    setBounds(minX->x() - paddingX, maxY->y() + paddingY, maxX->x() + paddingX,
              minY->y() - paddingY);

    setSteps(paddingX, paddingY);
}

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
        // qDebug() << "Horizontal axis is out of range";
    }

    if (mBottom < 0 && mTop > 0) {
        painter.drawLine(0, mTop * vk, width(), mTop * vk);
    }
    else {
        qDebug() << "Vertical axis is out of range";
    }

    QPen pen = QPen(painter.pen());
    pen.setStyle(Qt::DashLine);
    pen.setColor(Qt::gray);

    QPen textPen = QPen(pen);
    textPen.setColor(Qt::darkGreen);

    QFont font = const_cast<QFont&>(painter.font());
    QFontMetrics metrics{font};
    if (mHorizontalStep > 0.01) {
        for (double i = mHorizontalStep; i < mRight; i = i + mHorizontalStep) {
            painter.setPen(pen);
            auto pos = i * hk - mLeft * hk;
            painter.drawLine(pos, 0, pos, height());

            std::stringstream ss;
            ss.precision(2);
            painter.setPen(textPen);
            ss << std::fixed << i;
            auto offset = metrics.horizontalAdvance(ss.str().c_str()) / 2;
            painter.drawText(pos - offset, 15, ss.str().c_str());
            painter.drawText(pos - offset, height() - 10, ss.str().c_str());
        }

        for (double i = -mHorizontalStep; i > mLeft; i = i - mHorizontalStep) {
            painter.setPen(pen);
            auto pos = i * hk - mLeft * hk;
            painter.drawLine(pos, 0, pos, height());

            std::stringstream ss;
            ss.precision(2);
            painter.setPen(textPen);
            ss << std::fixed << i;
            auto offset = metrics.horizontalAdvance(ss.str().c_str()) / 2;
            painter.drawText(pos - offset, 15, ss.str().c_str());
            painter.drawText(pos - offset, height() - 10, ss.str().c_str());
        }
    }
    else {
        // qDebug() << "Incorrect horizontal step";
    }

    if (mVerticalStep > 0.01) {
        for (double i = mTop - mVerticalStep; i > 0; i = i - mVerticalStep) {
            painter.setPen(pen);
            auto pos = i * vk;
            painter.drawLine(0, pos, width(), pos);

            std::stringstream ss;
            ss.precision(2);
            painter.setPen(textPen);
            ss << std::fixed << -(i - mTop);
            auto offset = metrics.height() / 2 - metrics.ascent();
            auto margin = metrics.horizontalAdvance(ss.str().c_str()) + 5;
            painter.drawText(10, pos - offset, ss.str().c_str());
            painter.drawText(width() - margin, pos - offset, ss.str().c_str());
        }

        for (double i = mTop + mVerticalStep; i < -mBottom + mTop; i = i + mVerticalStep) {
            painter.setPen(pen);
            auto pos = i * vk;
            painter.drawLine(0, pos, width(), pos);

            std::stringstream ss;
            ss.precision(2);
            painter.setPen(textPen);
            ss << std::fixed << -(i - mTop);
            auto offset = metrics.height() / 2 - metrics.ascent();
            auto margin = metrics.horizontalAdvance(ss.str().c_str()) + 5;
            painter.drawText(10, pos - offset, ss.str().c_str());
            painter.drawText(width() - margin, pos - offset, ss.str().c_str());
        }
    }
    else {
        // qDebug() << "Incorrect vertical step";
    }

    pen = const_cast<QPen&>(painter.pen());
    pen.setStyle(Qt::SolidLine);
    pen.setColor(Qt::red);
    pen.setWidth(2);
    painter.setPen(pen);

    calculate();

    auto oldBrush = painter.brush();
    painter.setBrush(QBrush(Qt::red, Qt::SolidPattern));
    for (auto iter = mData.begin(); iter != mData.end(); iter++) {
        painter.drawEllipse(*iter, 2, 2);
    }

    painter.setBrush(oldBrush);

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
