#include "image_process.h"

#include <stdio.h>


void rgb_to_gray(const unsigned char *rgb, unsigned char *gray, int width, int height)
{
    int pixel_count = width * height;
    for (int i = 0; i < pixel_count; ++i)
    {
        int r = rgb[i * 3 + 0];
        int g = rgb[i * 3 + 1];
        int b = rgb[i * 3 + 2];

        // 使用加权平均公式
        gray[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}
