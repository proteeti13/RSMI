#include "Mbr.h"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <limits>
#include <algorithm>
#include <math.h>
using namespace std;

Mbr::Mbr()
{
}

Mbr::Mbr(float x1, float y1, float z1, float x2, float y2, float z2)
{
    this->x1 = x1;
    this->y1 = y1;
    this->z1 = z1;
    this->x2 = x2;
    this->y2 = y2;
    this->z2 = z2;
}

void Mbr::update(Point point)
{
    update(point.x, point.y, point.z);
}

void Mbr::update(float x, float y, float z)
{
    if (x < x1) x1 = x;
    if (x > x2) x2 = x;
    if (y < y1) y1 = y;
    if (y > y2) y2 = y;
    if (z < z1) z1 = z;
    if (z > z2) z2 = z;
}

void Mbr::update(Mbr mbr)
{
    if (mbr.x1 < x1) x1 = mbr.x1;
    if (mbr.x2 > x2) x2 = mbr.x2;
    if (mbr.y1 < y1) y1 = mbr.y1;
    if (mbr.y2 > y2) y2 = mbr.y2;
    if (mbr.z1 < z1) z1 = mbr.z1;
    if (mbr.z2 > z2) z2 = mbr.z2;
}

bool Mbr::contains(Point point)
{
    if (x1 > point.x || point.x > x2 ||
        y1 > point.y || point.y > y2 ||
        z1 > point.z || point.z > z2)
    {
        return false;
    }
    return true;
}

bool Mbr::strict_contains(Point point)
{
    if (x1 < point.x && point.x < x2 &&
        y1 < point.y && point.y < y2 &&
        z1 < point.z && point.z < z2)
    {
        return true;
    }
    return false;
}

bool Mbr::interact(Mbr mbr)
{
    if (x2 < mbr.x1 || mbr.x2 < x1) return false;
    if (y2 < mbr.y1 || mbr.y2 < y1) return false;
    if (z2 < mbr.z1 || mbr.z2 < z1) return false;
    return true;
}

vector<Mbr> Mbr::get_mbrs(vector<Point> dataset, float area, int num, float ratio)
{
    vector<Mbr> mbrs;
    srand(time(0));
    float sx = sqrt(area * ratio);
    float sy = sqrt(area / ratio);
    float sz = sx;
    int i = 0;
    int length = dataset.size();
    while (i < num)
    {
        int index = rand() % length;
        Point point = dataset[index];
        if (point.x + sx <= 1 && point.y + sy <= 1 && point.z + sz <= 1)
        {
            Mbr mbr(point.x, point.y, point.z, point.x + sx, point.y + sy, point.z + sz);
            mbrs.push_back(mbr);
            i++;
        }
    }
    return mbrs;
}

float Mbr::cal_dist(Point point)
{
    if (this->contains(point))
    {
        return 0;
    }
    float dx = (point.x < x1) ? (x1 - point.x) : (point.x > x2) ? (point.x - x2) : 0.0f;
    float dy = (point.y < y1) ? (y1 - point.y) : (point.y > y2) ? (point.y - y2) : 0.0f;
    float dz = (point.z < z1) ? (z1 - point.z) : (point.z > z2) ? (point.z - z2) : 0.0f;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

void Mbr::print()
{
    cout << "(x1=" << x1 << " y1=" << y1 << " z1=" << z1
         << " x2=" << x2 << " y2=" << y2 << " z2=" << z2 << ")" << endl;
}

// Returns 8 corners of the 3D bounding box
vector<Point> Mbr::get_corner_points()
{
    vector<Point> result;
    float xs[2] = {x1, x2};
    float ys[2] = {y1, y2};
    float zs[2] = {z1, z2};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                result.push_back(Point(xs[i], ys[j], zs[k]));
    return result;
}

Mbr Mbr::get_mbr(Point point, float knnquerySide)
{
    float ax1 = point.x - knnquerySide;
    float ax2 = point.x + knnquerySide;
    float ay1 = point.y - knnquerySide;
    float ay2 = point.y + knnquerySide;
    float az1 = point.z - knnquerySide;
    float az2 = point.z + knnquerySide;

    ax1 = ax1 < 0 ? 0 : ax1;
    ay1 = ay1 < 0 ? 0 : ay1;
    az1 = az1 < 0 ? 0 : az1;
    ax2 = ax2 > 1 ? 1 : ax2;
    ay2 = ay2 > 1 ? 1 : ay2;
    az2 = az2 > 1 ? 1 : az2;

    Mbr mbr(ax1, ay1, az1, ax2, ay2, az2);
    return mbr;
}

void Mbr::clean()
{
    x1 = 0; x2 = 0;
    y1 = 0; y2 = 0;
    z1 = 0; z2 = 0;
}

string Mbr::get_self()
{
    return to_string(x1) + " " + to_string(y1) + " " + to_string(z1) + " "
         + to_string(x2) + " " + to_string(y2) + " " + to_string(z2) + "\n";
}
