#include "regressions/Logistic2ParamRegression.h"
#include "regressions/LinearRegression.h"

using namespace regression;

int main()
{
    LinearRegression lineRegression;
    Logistic2ParamRegression::demonstrate(&lineRegression);

    return 0;
}
