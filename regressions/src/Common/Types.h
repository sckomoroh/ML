#pragma once

namespace common {

struct PointF {
    PointF(float xVal = 0.0f, float yVal = 0.0f)
        : x{xVal}
        , y{yVal}
    {
    }

    float x = 0.0f;
    float y = 0.0f;
};

}  // namespace common