#pragma once

#include <functional>
#include <random>
#include <vector>

namespace common {

class Utils {
public:
    Utils() = delete;

public:
    static std::vector<float> generateRange(float min, float max, int valuesCount)
    {
        std::vector<float> values(valuesCount);

        for (int i = 0; i < valuesCount; ++i) {
            values[i] = min + i * abs(min - max) / valuesCount;
        }

        return values;
    }

    template <class T>
    static std::vector<T> generateRange(float min,
                                            float max,
                                            int valuesCount,
                                            std::function<void(T&, float)> predicate)
    {
        std::vector<T> values(valuesCount);

        for (int i = 0; i < valuesCount; ++i) {
            float value = min + i * abs(min - max) / valuesCount;
            predicate(values[i], value);
        }

        return values;
    }

    static void randomizeData(std::vector<float>& data, float mean, float distance)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, distance);
        for (auto& item : data) {
            item += dist(gen);
        }
    }

    template <class T>
    static void randomizeData(std::vector<T>& data,
                                            float mean,
                                            float distance,
                                            std::function<void(T&, float)> distortFunction)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, distance);
        for (auto& item : data) {
            distortFunction(item, dist(gen));
        }
    }
};

}  // namespace common