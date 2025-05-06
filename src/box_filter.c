#include "box_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_neon.h>

/*
 * box_filter_origin
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  最简单的盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 */
void box_filter_origin(float *src, float *dst, int width, int height, int radius)
{
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            // 计算窗口的起始和结束位置
            int start_y = y - radius;
            if(start_y < 0) start_y = 0;
            int end_y = y + radius;
            if(end_y > height - 1) end_y = height - 1;
            int start_x = x - radius;
            if(start_x < 0) start_x = 0;
            int end_x = x + radius;
            if(end_x > width - 1) end_x = width - 1;

            // 计算窗口内像素值的和
            float sum = 0;
            for(int ty = start_y; ty <= end_y; ty++)
            {
                for(int tx = start_x; tx <= end_x; tx++)
                {
                    sum += src[ty * width + tx];
                }
            }

            // 求和滤波
            dst[y * width + x] = sum;
        }
    }
}

/*
 * box_filter_opencv
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 */
void box_filter_opencv(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            // 计算窗口的起始和结束位置
            int start_x = x - radius;
            if (start_x < 0) start_x = 0;
            int end_x = x + radius;
            if (end_x > width - 1) end_x = width - 1;

            float sum = 0;
            // 计算窗口内行像素值的和
            for (int tx = start_x; tx <= end_x; tx++)
            {
                sum += src[y * width + tx];
            }

            temp_x[y * width + x] = sum;
        }
    }

    // 列方向滤波
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            // 计算窗口的起始和结束位置
            int start_y = y - radius;
            if (start_y < 0) start_y = 0;
            int end_y = y + radius;
            if (end_y > height - 1) end_y = height - 1;

            float sum = 0;
            // 计算窗口内列像素值的和
            for (int ty = start_y; ty <= end_y; ty++)
            {
                sum += temp_x[ty * width + x];
            }

            dst[y * width + x] = sum;
        }
    }

    free(temp_x);
}

/*
 * box_filter_opencv_v2
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 */
void box_filter_opencv_v2(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        float sum = 0;
        int stride = y * width;     // 某一行的起始偏移量

        // 处理一行的左边缘
        for (int x = 0; x < radius; x++)
        {
            sum += src[stride + x];
        }

        for (int x = 0; x <= radius; x++)
        {
            sum += src[stride + x + radius];
            temp_x[stride + x] = sum;
        }

        // 处理一行的中间值
        for (int x = radius + 1; x < width - radius - 1; x++)
        {
            sum += src[stride + x + radius] - src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }

        // 处理一行的右边缘
        for (int x = width - radius; x < width; x++)
        {
            sum -= src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }
    }

    // 列方向滤波
    for (int x = 0; x < width; x++)
    {
        float sum = 0;

        // 处理一列的上边缘
        for (int y = 0; y < radius; y++)
        {
            sum += temp_x[x + y * width];
        }

        for (int y = 0; y <= radius; y++)
        {
            sum += temp_x[x + (y + radius) * width];
            dst[x + y * width] = sum;
        }

        // 处理一列的中间值
        for (int y = radius + 1; y < height - radius - 1; y++)
        {
            sum += temp_x[x + (y + radius) * width] - temp_x[x + (y - radius - 1) * width];
            dst[x + y * width] = sum;
        }

        // 处理一列的下边缘
        for (int y = height - radius; y < height; y++)
        {
            sum -= temp_x[x + (y - radius - 1) * width];
            dst[x + y * width] = sum;
        }
    }

    free(temp_x);
}


/*
 * box_filter_opencv_v2
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 *  减小cache miss思路：通过列缓存减小cache miss
 */
void box_filter_reduce_cache_miss(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 列缓存
    float *colsum_ptr = (float *)malloc(width * sizeof(float));
    memset(colsum_ptr, 0, width * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        float sum = 0;
        int stride = y * width;     // 某一行的起始偏移量

        // 处理一行的左边缘
        for (int x = 0; x < radius; x++)
        {
            sum += src[stride + x];
        }

        for (int x = 0; x <= radius; x++)
        {
            sum += src[stride + x + radius];
            temp_x[stride + x] = sum;
        }

        // 处理一行的中间值
        for (int x = radius + 1; x < width - radius - 1; x++)
        {
            sum += src[stride + x + radius] - src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }

        // 处理一行的右边缘
        for (int x = width - radius; x < width; x++)
        {
            sum -= src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }
    }

    // 列方向滤波
    for (int y = 0; y < radius; y++)
    {
        int stride = y * width;
        for (int x = 0; x < width; x++)
        {
            colsum_ptr[x] += temp_x[stride + x];
        }
    }

    // 处理顶部的几行
    for (int y = 0; y <= radius; y++)
    {
        int stride = y * width;
        for (int x = 0; x < width; x++)
        {
            colsum_ptr[x] += temp_x[x + (y + radius) * width];
            dst[x + stride] = colsum_ptr[x];
        }
    }

    // 处理中间的行数
    for (int y = radius + 1; y < height - radius - 1; y++)
    {
        int stride = y * width;
        for (int x = 0; x < width; x++)
        {
            colsum_ptr[x] += temp_x[x + (y + radius) * width] - temp_x[x + (y - radius - 1) * width];
            dst[x + stride] = colsum_ptr[x];
        }
    }

    // 处理底部的几行
    for (int y = height - radius; y < height; y++)
    {
        int stride = y * width;
        for (int x = 0; x < width; x++)
        {
            colsum_ptr[x] -= temp_x[x + (y - radius - 1) * width];
            dst[x + stride] = colsum_ptr[x];
        }
    }

    free(temp_x);
    free(colsum_ptr);
}


/*
 * box_filter_neon_intrinsics
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 *  减小cache miss思路：通过列缓存减小cache miss
 *  NEON优化思路：使用NEON Intrinsics优化(在行方向上由于相邻元素有依赖关系，因此是无法并行的，
 *           所以我们可以在列方向上使用 Neon Intrinsics 来并行处理数据。)
 */
void box_filter_neon_intrinsics(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 列缓存
    float *colsum_ptr = (float *)malloc(width * sizeof(float));
    memset(colsum_ptr, 0, width * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        float sum = 0;
        int stride = y * width;     // 某一行的起始偏移量

        // 处理一行的左边缘
        for (int x = 0; x < radius; x++)
        {
            sum += src[stride + x];
        }

        for (int x = 0; x <= radius; x++)
        {
            sum += src[stride + x + radius];
            temp_x[stride + x] = sum;
        }

        // 处理一行的中间值
        for (int x = radius + 1; x < width - radius - 1; x++)
        {
            sum += src[stride + x + radius] - src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }

        // 处理一行的右边缘
        for (int x = width - radius; x < width; x++)
        {
            sum -= src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }
    }

    // 列方向滤波
    // 计算列数据的向量化处理边界问题
    // 128-bit向量寄存器，浮点数据（32-bit）,所以按照向量并行一次处理4个数据
    int block = width >> 2;
    int remain = width - (block << 2);

    for (int y = 0; y < radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_temp_x_ptr = temp_x + stride;

        int n = block;
        int re = remain;

        for (; n > 0; n--)
        {
            float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

            v_colsum = vaddq_f32(v_colsum, v_temp_x);

            vst1q_f32(tmp_colsum_ptr, v_colsum);

            tmp_colsum_ptr += 4;
            tmp_temp_x_ptr += 4;
        }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            tmp_colsum_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理顶部的几行
    for (int y = 0; y <= radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_ptr = temp_x + (y + radius) * width;

        int n = block;
        int re = remain;

        for (; n > 0; n--)
        {
            float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

            v_colsum = vaddq_f32(v_colsum, v_temp_x);

            vst1q_f32(tmp_colsum_ptr, v_colsum);
            vst1q_f32(tmp_dst_ptr, v_colsum);

            tmp_colsum_ptr += 4;
            tmp_dst_ptr += 4;
            tmp_temp_x_ptr += 4;
        }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理中间的行数
    for (int y = radius + 1; y < height - radius - 1; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_add_ptr = temp_x + (y + radius) * width;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        for (; n > 0; n--)
        {
            float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t v_temp_x_add = vld1q_f32(tmp_temp_x_add_ptr);
            float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

            v_colsum = vaddq_f32(v_colsum, v_temp_x_add);
            v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

            vst1q_f32(tmp_colsum_ptr, v_colsum);
            vst1q_f32(tmp_dst_ptr, v_colsum);

            tmp_colsum_ptr += 4;
            tmp_dst_ptr += 4;
            tmp_temp_x_add_ptr += 4;
            tmp_temp_x_sub_ptr += 4;
        }
        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_add_ptr - *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_add_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    // 处理底部的几行
    for (int y = height - radius; y < height; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        for (; n > 0; n--)
        {
            float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
            float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

            v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

            vst1q_f32(tmp_colsum_ptr, v_colsum);
            vst1q_f32(tmp_dst_ptr, v_colsum);

            tmp_colsum_ptr += 4;
            tmp_dst_ptr += 4;
            tmp_temp_x_sub_ptr += 4;
        }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr -= *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    free(temp_x);
    free(colsum_ptr);
}

/*
 * box_filter_neon_assembly
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 *  减小cache miss思路：通过列缓存减小cache miss
 *  NEON优化思路：使用 NEON assembly优化
 */
void box_filter_neon_assembly(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 列缓存
    float *colsum_ptr = (float *)malloc(width * sizeof(float));
    memset(colsum_ptr, 0, width * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        float sum = 0;
        int stride = y * width;     // 某一行的起始偏移量

        // 处理一行的左边缘
        for (int x = 0; x < radius; x++)
        {
            sum += src[stride + x];
        }

        for (int x = 0; x <= radius; x++)
        {
            sum += src[stride + x + radius];
            temp_x[stride + x] = sum;
        }

        // 处理一行的中间值
        for (int x = radius + 1; x < width - radius - 1; x++)
        {
            sum += src[stride + x + radius] - src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }

        // 处理一行的右边缘
        for (int x = width - radius; x < width; x++)
        {
            sum -= src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }
    }

    // 列方向滤波
    // 计算列数据的向量化处理边界问题
    // 128-bit向量寄存器，浮点数据（32-bit）,所以按照向量并行一次处理4个数据
    int block = width >> 2;
    int remain = width - (block << 2);

    for (int y = 0; y < radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_temp_x_ptr = temp_x + stride;

        int n = block;
        int re = remain;

        // 写内联汇编的时候，一般按照输出-输入-汇编代码- Clobbers 的顺序来写
        asm volatile(
            "0:                                     \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vadd.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "subs %2, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_ptr), "2"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_temp_x_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            tmp_colsum_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理顶部的几行
    for (int y = 0; y <= radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_ptr = temp_x + (y + radius) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vadd.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "vst1.32 {d0-d1}, [%2]!                 \n"
            "subs %3, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_ptr), "2"(tmp_dst_ptr), "3"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理中间的行数
    for (int y = radius + 1; y < height - radius - 1; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_add_ptr = temp_x + (y + radius) * width;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n" // 标签,用于循环
            "vld1.32 {d0-d1}, [%2]                  \n" // 加载输入参数2，即 tmp_colsum_ptr
            "vld1.32 {d2-d3}, [%0]!                 \n" // 加载输入参数0，即 tmp_temp_x_add_ptr，并递增
            "vld1.32 {d4-d5}, [%1]!                 \n" // 加载输入参数1，即 tmp_temp_x_sub_ptr，并递增
            "vadd.f32 q0, q0, q1                    \n" // 将 q0 和 q1 相加，结果存储在 q0 中
            "vsub.f32 q0, q0, q2                    \n" // 将 q0 和 q2 相减，结果存储在 q0 中
            "vst1.32 {d0-d1}, [%2]!                 \n" // 将 q0 存储到输出参数2，即 tmp_colsum_ptr，并递增
            "vst1.32 {d0-d1}, [%3]!                 \n" // 将 q0 存储到输出参数3，即 tmp_dst_ptr，并递增
            "subs %4, #1                            \n" // 将输入参数4，即 n 减1
            "bne 0b                                 \n" // 如果 n 不等于0，则跳转到标签0处
            // output operands
            : "=r"(tmp_temp_x_add_ptr), "=r"(tmp_temp_x_sub_ptr), "=r"(tmp_colsum_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            // input operands
            : "0"(tmp_temp_x_add_ptr), "1"(tmp_temp_x_sub_ptr), "2"(tmp_colsum_ptr), "3"(tmp_dst_ptr), "4"(n)
            // Clobbers 这里用到了q0,q1,q2这三个向量寄存器
            : "cc", "memory", "q0", "q1", "q2");

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x_add = vld1q_f32(tmp_temp_x_add_ptr);
        //     float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x_add);
        //     v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_add_ptr += 4;
        //     tmp_temp_x_sub_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_add_ptr - *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_add_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    // 处理底部的几行
    for (int y = height - radius; y < height; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vsub.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "vst1.32 {d0-d1}, [%2]!                 \n"
            "subs %3, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_sub_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_sub_ptr), "2"(tmp_dst_ptr), "3"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

        //     v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_sub_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr -= *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    free(temp_x);
    free(colsum_ptr);
}

/*
 * box_filter_neon_assembly_pld
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 *  减小cache miss思路：通过列缓存减小cache miss
 *  NEON优化思路：使用 NEON assembly优化
 *  pld 指令：预取指令，可以减少 cache miss
 */
void box_filter_neon_assembly_pld(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 列缓存
    float *colsum_ptr = (float *)malloc(width * sizeof(float));
    memset(colsum_ptr, 0, width * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        float sum = 0;
        int stride = y * width;     // 某一行的起始偏移量

        // 处理一行的左边缘
        for (int x = 0; x < radius; x++)
        {
            sum += src[stride + x];
        }

        for (int x = 0; x <= radius; x++)
        {
            sum += src[stride + x + radius];
            temp_x[stride + x] = sum;
        }

        // 处理一行的中间值
        for (int x = radius + 1; x < width - radius - 1; x++)
        {
            sum += src[stride + x + radius] - src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }

        // 处理一行的右边缘
        for (int x = width - radius; x < width; x++)
        {
            sum -= src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }
    }

    // 列方向滤波
    // 计算列数据的向量化处理边界问题
    // 128-bit向量寄存器，浮点数据（32-bit）,所以按照向量并行一次处理4个数据
    int block = width >> 2;
    int remain = width - (block << 2);

    for (int y = 0; y < radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_temp_x_ptr = temp_x + stride;

        int n = block;
        int re = remain;

        // 写内联汇编的时候，一般按照输出-输入-汇编代码- Clobbers 的顺序来写
        asm volatile(
            "0:                                     \n"
            "pld    [%0, #128]                      \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "pld    [%1, #128]                      \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vadd.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "subs %2, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_ptr), "2"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_temp_x_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            tmp_colsum_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理顶部的几行
    for (int y = 0; y <= radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_ptr = temp_x + (y + radius) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n"
            "pld    [%0, #128]                      \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "pld    [%1, #128]                      \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vadd.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "pld    [%2, #128]                      \n"
            "vst1.32 {d0-d1}, [%2]!                 \n"
            "subs %3, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_ptr), "2"(tmp_dst_ptr), "3"(n)
            : "cc", "memory", "q0", "q1");

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理中间的行数
    for (int y = radius + 1; y < height - radius - 1; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_add_ptr = temp_x + (y + radius) * width;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n" // 标签,用于循环
            "pld    [%2, #128]                      \n" // 预取数据，提高缓存命中率
            "vld1.32 {d0-d1}, [%2]                  \n" // 加载输入参数2，即 tmp_colsum_ptr
            "pld    [%0, #128]                      \n" // 预取数据，提高缓存命中率
            "vld1.32 {d2-d3}, [%0]!                 \n" // 加载输入参数0，即 tmp_temp_x_add_ptr，并递增
            "pld    [%1, #128]                      \n" // 预取数据，提高缓存命中率
            "vld1.32 {d4-d5}, [%1]!                 \n" // 加载输入参数1，即 tmp_temp_x_sub_ptr，并递增
            "vadd.f32 q0, q0, q1                    \n" // 将 q0 和 q1 相加，结果存储在 q0 中
            "vsub.f32 q0, q0, q2                    \n" // 将 q0 和 q2 相减，结果存储在 q0 中
            "vst1.32 {d0-d1}, [%2]!                 \n" // 将 q0 存储到输出参数2，即 tmp_colsum_ptr，并递增
            "pld    [%3, #128]                      \n" // 预取数据，提高缓存命中率
            "vst1.32 {d0-d1}, [%3]!                 \n" // 将 q0 存储到输出参数3，即 tmp_dst_ptr，并递增
            "subs %4, #1                            \n" // 将输入参数4，即 n 减1
            "bne 0b                                 \n" // 如果 n 不等于0，则跳转到标签0处
            // output operands
            : "=r"(tmp_temp_x_add_ptr), "=r"(tmp_temp_x_sub_ptr), "=r"(tmp_colsum_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            // input operands
            : "0"(tmp_temp_x_add_ptr), "1"(tmp_temp_x_sub_ptr), "2"(tmp_colsum_ptr), "3"(tmp_dst_ptr), "4"(n)
            // Clobbers 这里用到了q0,q1,q2这三个向量寄存器
            : "cc", "memory", "q0", "q1", "q2");

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x_add = vld1q_f32(tmp_temp_x_add_ptr);
        //     float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x_add);
        //     v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_add_ptr += 4;
        //     tmp_temp_x_sub_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_add_ptr - *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_add_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    // 处理底部的几行
    for (int y = height - radius; y < height; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n"
            "pld    [%0, #128]                      \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "pld    [%1, #128]                      \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vsub.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "pld    [%2, #128]                      \n"
            "vst1.32 {d0-d1}, [%2]!                 \n"
            "subs %3, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_sub_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_sub_ptr), "2"(tmp_dst_ptr), "3"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

        //     v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_sub_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr -= *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    free(temp_x);
    free(colsum_ptr);
}

/*
 * box_filter_neon_assembly_pld_dual_lanch
 * Input: src, dst, width, height, radius
 * Output: None
 * Description:
 *  借鉴openCV的行列分离盒子滤波处理函数
 *  radius: 滤波半径,1表示3x3窗口,2表示5x5窗口,以此类推
 *
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 *  减小cache miss思路：通过列缓存减小cache miss
 *  NEON优化思路：使用 NEON assembly优化
 *  pld 指令：预取指令，可以减少 cache miss
 *  双发射流水线：
 */
void box_filter_neon_assembly_pld_dual_lanch(float *src, float *dst, int width, int height, int radius)
{
    // 临时存储行方向的滤波结果
    float *temp_x = (float *)malloc(width * height * sizeof(float));

    // 列缓存
    float *colsum_ptr = (float *)malloc(width * sizeof(float));
    memset(colsum_ptr, 0, width * sizeof(float));

    // 行方向滤波
    for(int y = 0; y < height; y++)
    {
        float sum = 0;
        int stride = y * width;     // 某一行的起始偏移量

        // 处理一行的左边缘
        for (int x = 0; x < radius; x++)
        {
            sum += src[stride + x];
        }

        for (int x = 0; x <= radius; x++)
        {
            sum += src[stride + x + radius];
            temp_x[stride + x] = sum;
        }

        // 处理一行的中间值
        for (int x = radius + 1; x < width - radius - 1; x++)
        {
            sum += src[stride + x + radius] - src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }

        // 处理一行的右边缘
        for (int x = width - radius; x < width; x++)
        {
            sum -= src[stride + x - radius - 1];
            temp_x[stride + x] = sum;
        }
    }

    // 列方向滤波
    // 计算列数据的向量化处理边界问题
    // 128-bit向量寄存器，浮点数据（32-bit）,所以按照向量并行一次处理4个数据
    int block = width >> 2;
    int remain = width - (block << 2);

    for (int y = 0; y < radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_temp_x_ptr = temp_x + stride;

        int n = block;
        int re = remain;

        // 写内联汇编的时候，一般按照输出-输入-汇编代码- Clobbers 的顺序来写
        asm volatile(
            "0:                                     \n"
            "pld    [%0, #128]                      \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "pld    [%1, #128]                      \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vadd.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "subs %2, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_ptr), "2"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_temp_x_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            tmp_colsum_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理顶部的几行
    for (int y = 0; y <= radius; y++)
    {
        int stride = y * width;

        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_ptr = temp_x + (y + radius) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n"
            "pld    [%0, #128]                      \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "pld    [%1, #128]                      \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vadd.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "pld    [%2, #128]                      \n"
            "vst1.32 {d0-d1}, [%2]!                 \n"
            "subs %3, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_ptr), "2"(tmp_dst_ptr), "3"(n)
            : "cc", "memory", "q0", "q1");

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x = vld1q_f32(tmp_temp_x_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_ptr++;
        }
    }

    // 处理中间的行数
    for (int y = radius + 1; y < height - radius - 1; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_add_ptr = temp_x + (y + radius) * width;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        // 双发射流水线处理，就是将两个指令同时发射，提高效率，所以一次处理8个数据
        int n = width >> 3;
        int re = width - (n << 3);

        asm volatile(
            "0:                                     \n" // 标签,用于循环

            "pld    [%2, #256]                      \n" // 预取数据，提高缓存命中率
            "vld1.32 {d0-d3}, [%2]                  \n" // 加载输入参数2，即 tmp_colsum_ptr

            "pld    [%0, #256]                      \n" // 预取数据，提高缓存命中率
            "vld1.32 {d4-d7}, [%0]!                 \n" // 加载输入参数0，即 tmp_temp_x_add_ptr，并递增

            "vadd.f32 q0, q0, q2                    \n"

            // 穿插用于双发射
            "pld    [%1, #256]                      \n" // 预取数据，提高缓存命中率
            "vld1.32 {d8-d11}, [%1]!                \n" // 加载输入参数1，即 tmp_temp_x_sub_ptr，并递增

            "vadd.f32 q1, q1, q3                    \n" // 将 q1 和 q3 相加，结果存储在 q1 中

            "vsub.f32 q0, q0, q4                    \n" // 将 q0 和 q4 相减，结果存储在 q0 中
            "vsub.f32 q1, q1, q5                    \n" // 将 q1 和 q5 相减，结果存储在 q1 中

            // 存储结果
            "vst1.32 {d0-d3}, [%2]!                 \n" // 将 q0 存储到输出参数2，即 tmp_colsum_ptr，并递增

            "pld    [%3, #256]                      \n" // 预取数据，提高缓存命中率
            "vst1.32 {d0-d3}, [%3]!                 \n" // 将 q0 存储到输出参数3，即 tmp_dst_ptr，并递增

            "subs %4, #1                            \n" // 将输入参数4，即 n 减 1
            "bne 0b                                 \n" // 如果 n 不等于0，则跳转到标签0处
            // output operands
            : "=r"(tmp_temp_x_add_ptr), "=r"(tmp_temp_x_sub_ptr), "=r"(tmp_colsum_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            // input operands
            : "0"(tmp_temp_x_add_ptr), "1"(tmp_temp_x_sub_ptr), "2"(tmp_colsum_ptr), "3"(tmp_dst_ptr), "4"(n)
            // Clobbers 这里用到了q0,q1,q2,q3,q4,q5这六个向量寄存器
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x_add = vld1q_f32(tmp_temp_x_add_ptr);
        //     float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

        //     v_colsum = vaddq_f32(v_colsum, v_temp_x_add);
        //     v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_add_ptr += 4;
        //     tmp_temp_x_sub_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr += *tmp_temp_x_add_ptr - *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_add_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    // 处理底部的几行
    for (int y = height - radius; y < height; y++)
    {
        int stride = y * width;
        float *tmp_colsum_ptr = colsum_ptr;
        float *tmp_dst_ptr = dst + stride;
        float *tmp_temp_x_sub_ptr = temp_x + (y - radius - 1) * width;

        int n = block;
        int re = remain;

        asm volatile(
            "0:                                     \n"
            "pld    [%0, #128]                      \n"
            "vld1.32 {d0-d1}, [%0]                  \n"
            "pld    [%1, #128]                      \n"
            "vld1.32 {d2-d3}, [%1]!                 \n"
            "vsub.f32 q0, q0, q1                    \n"
            "vst1.32 {d0-d1}, [%0]!                 \n"
            "pld    [%2, #128]                      \n"
            "vst1.32 {d0-d1}, [%2]!                 \n"
            "subs %3, #1                            \n"
            "bne 0b                                 \n"
            : "=r"(tmp_colsum_ptr), "=r"(tmp_temp_x_sub_ptr), "=r"(tmp_dst_ptr), "=r"(n)
            : "0"(tmp_colsum_ptr), "1"(tmp_temp_x_sub_ptr), "2"(tmp_dst_ptr), "3"(n)
            : "cc", "memory", "q0", "q1"
        );

        // for (; n > 0; n--)
        // {
        //     float32x4_t v_colsum = vld1q_f32(tmp_colsum_ptr);
        //     float32x4_t v_temp_x_sub = vld1q_f32(tmp_temp_x_sub_ptr);

        //     v_colsum = vsubq_f32(v_colsum, v_temp_x_sub);

        //     vst1q_f32(tmp_colsum_ptr, v_colsum);
        //     vst1q_f32(tmp_dst_ptr, v_colsum);

        //     tmp_colsum_ptr += 4;
        //     tmp_dst_ptr += 4;
        //     tmp_temp_x_sub_ptr += 4;
        // }

        for (; re > 0; re--)
        {
            *tmp_colsum_ptr -= *tmp_temp_x_sub_ptr;
            *tmp_dst_ptr = *tmp_colsum_ptr;
            tmp_colsum_ptr++;
            tmp_dst_ptr++;
            tmp_temp_x_sub_ptr++;
        }
    }

    free(temp_x);
    free(colsum_ptr);
}
