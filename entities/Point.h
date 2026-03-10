#ifndef POINT_H
#define POINT_H
#include <vector>
#include <string.h>
#include <string>
using namespace std;
class Point
{

public:

    float index;
    float x;       // SourceID
    float y;       // Hop1_ID
    float z;       // Hop2_ID
    long long x_i;
    long long y_i;

    float temp_dist = 0.0;

    Point(float, float, float);
    Point();
    bool operator == (const Point& point);
    float cal_dist(Point);
    void print();
    static vector<Point> get_points(vector<Point>, int);
    static vector<Point> get_inserted_points(long long);

    string get_self();
};

#endif
