# 盒子滤波算法优化-基于Arm平台

> 下面的内容是基于 [一份朴实无华的移动端盒子滤波算法优化笔记](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96/%E4%B8%80%E4%BB%BD%E6%9C%B4%E5%AE%9E%E6%97%A0%E5%8D%8E%E7%9A%84%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%9B%92%E5%AD%90%E6%BB%A4%E6%B3%A2%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96%E7%AC%94%E8%AE%B0/)的跟练
> 跟练的相关源码，见对应的[github仓库](https://github.com/hhy0317/box_filter_optimizer)

# 编译器选项
---
笔者下面的验证环节使用编译器优化等级为`-O2`

# 盒子滤波原理介绍
---
今天要介绍的是在 Arm 端一步步优化盒子滤波算法，盒子滤波是最经典的滤波算法之一，常见的均值滤波算法就是盒子滤波归一化后获得结果。在原理上盒子滤波和均值滤波类似，用一个内核和图像进行卷积：
![[盒子滤波算法优化_image_20250427_1.png]]
其中：
![[盒子滤波算法优化_image_20250427_2.png]]
如果选择归一化，那么盒子滤波就是标准的均值滤波。

# 原始实现
---
按照上面的原理，我们实现一个简单的盒子滤波（求和盒子滤波）
```C
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
```

可以通过打印查看输入输出数据前$10\times10$的数据对比，判断计算是否正确

![[盒子滤波算法优化-基于Arm平台_image_20250427_1.png]]

输入数据是$6000\times4000$的单通道灰度数据，我们在树莓派4B的32位系统平台上运行，测试耗时数据如下:

| 图像大小      | 优化算法 | 滤波半径 | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | ---- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现 | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
其中
- 滤波半径=3，代表滤波的盒子为$7\times7$
- 通过timer模块的`clock()`函数计算时间差来计算耗时
# 第一版优化
---
首先我们借鉴一下 OpenCV 在实现盒子滤波时的 Trick，即利用盒子滤波是一种行列可分离的滤波，所以先进行行方向的滤波，得到中间结果，然后再对中间结果进行列方向的处理，得到最终的结果。代码实现如下：
```C
/**
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
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
```

测试耗时数据如下：

| 图像大小      | 滤波半径 | 优化算法     | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | -------- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现     | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现 | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |

# 第二版~~优化~~
---
在上一版算法中，虽然使用行列分离技巧后降低了一些重复计算，但是并没有完全解决重复计算的问题，在行或列的单独方向上仍然存在重复计算。并且此算法的复杂度仍然和半径有关，大概复杂度为O(n×m×(2r+1)) +  O(n×m×(2r+1));其中n为矩阵的宽度，m 为矩阵的高度，r 为滤波半径。实际上我们在这里再加一个 Trick，就可以让算法的复杂度和半径无关了。例如对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。这样维护这个行或者列的和就和半径无关了。
代码实现如下：
```C
/*
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
```

理论上来说，这版优化的速度比第一版要快很多吧？我们来测一下速：

| 图像大小      | 滤波半径 | 优化算法            | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | --------------- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现            | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现        | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗 | 10   | 35036.28ms  | RaspberryPi 4B <br>ARM Cortex-A72 |

现在发生了一个非常奇怪的事情，虽然计算量变少了，但是耗时比原版还变多了？这是为什么呢？
# 第三版优化-减少 Cache Miss
---
在上一版优化中，我们在列方向上的处理方式和行方向上完全相同。但由于 CPU 架构的一些原因（**主要是 Cache Miss 的增加**），同样的算法沿着列方向处理总是会比按照行方向慢一个档次，因此在上面的第二版优化中由于频繁的在列方向进行操作（这个操作的跨度还比较大，比如第 $Y$ 行会访问第$Y−Radius−1$ 行的元素）使得算法速度被严重拖慢，解决这个问题的方法有很多，例如：先对中间结果进行转置，然后再按照行方向的规则进行处理，处理完后在将数据转置回去，但是矩阵转置相对比较麻烦。还有一种方法是 OpenCV 代码实现中展示的，即用了一个大小是$width$ 的向量`colSum`，来存储某一行对应点的列半径区域内的和，然后遍历的时候还是按照行来遍历，中间部分对`colSum`的更新也是减去遍历跑出去的一行，加上进来的一行，这样就可以减少 Cache Miss。

这部分的代码实现如下：
```C
/*
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

```

我们来测一下速：

| 图像大小      | 滤波半径 | 优化算法                                | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | ----------------------------------- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现                                | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现                            | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗                     | 10   | 35036.28ms  | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss | 10   | 2682.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |

可以看到速度相比于第一版又优化了 接近3 倍，所以我们这个减少 Cache Miss 的技巧是行之有效的。

# 第四版优化-Neon Intrinsics
---
可以看下下面的笔记学习NEON指令集
[[ARM-NEON数据并行技术]]

接下来我们试一下使用 Neon Intrinsics 来优化一下这个算法，关于 Neon 指令集的编写入门也可以看这篇文章：[【AI 移动端算法优化】五，移动端 arm cpu 优化学习笔记之内联汇编入门](https://mp.weixin.qq.com/s/6RoQAPsenD7Pu9enTqGeoQ) 。

因为在行方向上由于相邻元素有依赖关系，因此是无法并行的，所以我们可以在列方向上使用 Neon Intrinsics 来并行处理数据。详细代码如下：

```C
/*
 * 优化思路，将行方向和列方向的滤波分开，减少重复计算
 *  理解：比如将7x7的盒子数据，经过行方向滤波后编程了1x7,再进行列方向滤波，就变成了1x1,这样减少了重复计算
 *  优化思路2:滑动窗口优化，对于某一行来讲，我们首先计算第一个点开头的半径范围内的和，然后对于接下来遍历到的点不需要重复计算半径区域内的和，
 *          只需要把前一个元素半径内的和，按半径窗口右 / 下偏移之后，减去左边移出去的点并且加上右边新增的一个点即可。
 *          这样维护这个行或者列的和就和半径无关了。
 *  减小cache miss思路：通过列缓存减小cache miss
 *  NEON优化思路：使用NEON指令集优化(在行方向上由于相邻元素有依赖关系，因此是无法并行的，
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
```

我们来测一下速：

| 图像大小      | 滤波半径 | 优化算法                                                    | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | ------------------------------------------------------- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现                                                    | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现                                                | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗                                         | 10   | 35036.28ms  | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss                     | 10   | 2682.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Intrinsics | 10   | 2550.66ms   | RaspberryPi 4B <br>ARM Cortex-A72 |

可以看到使用 Neon Intrinsics 之后，速度又有了进一步提升，接下来的一步需要向 Neon Assembly 以及调度板子上拥有的多核角度去分析了。
# 第五版优化-NEON 内联汇编（Assembly）
---
没有基础的先学习下NEON的内联汇编是什么
[[ARM-NEON内联汇编]]

代码过长，就不贴全部的代码了，可以到对应的代码仓查看，修改后的部分代码如下
> 注释的部分用的是 NEON Intrinsics 实现的
```C
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
```

我们来测一下速：

| 图像大小      | 滤波半径 | 优化算法                                                    | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | ------------------------------------------------------- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现                                                    | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现                                                | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗                                         | 10   | 35036.28ms  | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss                     | 10   | 2682.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Intrinsics | 10   | 2550.66ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Assembly   | 10   | 2533.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |

可以看到改写了内联汇编之后速度并没有加快很多，这是因为编译器帮我们做了优化（-O2），导致NEON Intrinsics和NEON Assembly效果差不多。

# 第六版优化-ARM 中的预取命令 pld 的使用
---
[[ARM-pld 数据预读取指令]]
了解到pld的基本概念后，我们尝试使用pld在第五版优化的基础上进行优化，看看效果如何，只贴出部分代码如下，详细的请看代码仓
```C
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

```
我们来测一下速：

| 图像大小      | 滤波半径 | 优化算法                                                               | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | ------------------------------------------------------------------ | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现                                                               | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现                                                           | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗                                                    | 10   | 35036.28ms  | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss                                | 10   | 2682.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Intrinsics            | 10   | 2550.66ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Assembly              | 10   | 2533.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Assembly<br>+pld 预取指令 | 10   | 2532.51ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
在笔者这里优化效果并不明显。
# 第七版~~优化~~
---
按照[原版笔记](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96/%E4%B8%80%E4%BB%BD%E6%9C%B4%E5%AE%9E%E6%97%A0%E5%8D%8E%E7%9A%84%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%9B%92%E5%AD%90%E6%BB%A4%E6%B3%A2%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96%E7%AC%94%E8%AE%B0/)的思路：
>将上面的差值先处理一遍，然后再叠加到结果上，即将下面这个代码中的 add-sub 先计算出来存成 diff(diff=add-sub)，然后将结果加上这个 diff 即可：

但笔者没看出来原版笔记这么做的思路是怎样的？笔者的计算已经是寄存器最少的做法了，比如计算的结果直接存放到原始的位置，相较于原版笔记的使用六个寄存器，笔者一直都是使用的三个寄存器，故没法再尝试优化啦。

# 第八版优化-双发射流水线
---
- 可以参考[[程序性能优化理论与方法]]第七章单核优化中关于指令级并行的概念理解
- 可以参考这篇[博客](https://blog.csdn.net/qq_41154905/article/details/105163718)的介绍理解下相关的概念

> 双发射最重要的是：一边load一边算，中间穿插交错，不要堆一起

代码编写的基本思路就是多加载一组数据（4个浮点数），然后将加载和计算处理放到一起，这样就有一个关键的问题，那就是
```asm
"pld    [%1, #256]                      \n" // 预取数据，提高缓存命中率
"vld1.32 {d8-d11}, [%1]!                \n" // 加载输入参数1，即 tmp_temp_x_sub_ptr，并递增
```
这行指令应该放在哪里？我们先放在两个`vadd.fd32`后面测一下速度，这部分代码为：

```C
asm volatile(
	"0:                                     \n" // 标签,用于循环

	"pld    [%2, #256]                      \n" // 预取数据，提高缓存命中率
	"vld1.32 {d0-d3}, [%2]                  \n" // 加载输入参数2，即 tmp_colsum_ptr

	"pld    [%0, #256]                      \n" // 预取数据，提高缓存命中率
	"vld1.32 {d4-d7}, [%0]!                 \n" // 加载输入参数0，即 tmp_temp_x_add_ptr，并递增

	"vadd.f32 q0, q0, q2                    \n"
	"vadd.f32 q1, q1, q3                    \n" // 将 q1 和 q3 相加，结果存储在 q1 中

	// 穿插用于双发射
	"pld    [%1, #256]                      \n" // 预取数据，提高缓存命中率
	"vld1.32 {d8-d11}, [%1]!                \n" // 加载输入参数1，即 tmp_temp_x_sub_ptr，并递增

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

```

| 图像大小      | 滤波半径 | 优化算法                                                                          | 循环次数 | 耗时          | 运行平台                              |
| --------- | ---- | ----------------------------------------------------------------------------- | ---- | ----------- | --------------------------------- |
| 6000x4000 | 3    | 原始实现                                                                          | 10   | 27889.78 ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现                                                                      | 10   | 9478.91ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗                                                               | 10   | 35036.28ms  | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss                                           | 10   | 2682.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Intrinsics                       | 10   | 2550.66ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Assembly                         | 10   | 2533.45ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Assembly<br>+pld 预取指令            | 10   | 2532.51ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
| 6000x4000 | 3    | 行列分离计算实现<br>+滑窗<br>+列缓存减少cache miss<br>+NEON Assembly<br>+pld 预取指令<br>+双发射流水线 | 10   | 2528.17ms   | RaspberryPi 4B <br>ARM Cortex-A72 |
在笔者这里优化效果并不明显。
笔者尝试调整了指令的顺序，将其调整到了两个add操作中间，发现差别不大

> [!注意] 
> 双发射不是一种专门的写法，而是一种有意的编程习惯，使得CPU在执行时，有可能出发双发射流水线的优化运算

# 回顾
---
从上述的验证结果来看，笔者在树莓派平台，从使用NEON Intrinsics 开始，优化的效果就几乎没有变化啦，这是为什么呢？

笔者单独写了一个验证方式，发现根本原因是编译器开的是`-O2`,是编译器提前帮我们聪明的优化了，从NEON Inrinsics 开始都有点重复编译器的优化工作啦，自然看不出有什么优化的效果。
## 编译器优化等级验证

分别使用两种方法实现了对应的计算，具体代码如下：
```C
void neon_add_sub_v1_optimized(float* colsum, float* x_add, float* x_sub, float* dst, int n)
{
    n = n >> 2;

    asm volatile(
        "0:                                     \n"

        "vld1.32 {d0-d1}, [%0]                  \n"
        "vld1.32 {d2-d3}, [%1]!                 \n"
        "vld1.32 {d4-d5}, [%2]!                 \n"

        "vadd.f32 q0, q0, q1                    \n"
        "vsub.f32 q0, q0, q2                    \n"

        "vst1.32 {d0-d1}, [%0]!                 \n" // store result to colsum
        "vst1.32 {d0-d1}, [%3]!                 \n" // store result to dst

        "subs     %4, #1                        \n"
        "bne      0b                            \n"
        : "=r"(colsum), "=r"(x_add), "=r"(x_sub), "=r"(dst), "=r"(n)
        : "0"(colsum), "1"(x_add), "2"(x_sub), "3"(dst), "4"(n)
        : "cc", "memory", "q0", "q1", "q2");
}


void neon_add_sub_v2(float* colsum, float* x_add, float* x_sub, float* dst, int n)
{
    for (int i = 0; i < n; i++) {
        colsum[i] += x_add[i];
        colsum[i] -= x_sub[i];
        dst[i] = colsum[i];
    }
}
```

其中`neon_add_sub_v1_optimized`函数是使用了NEON的内联汇编优化，`neon_add_sub_v2`函数是使用原始的计算方式，我们开启了`-O2`的优化选项，测试耗时如下：

| n        | 执行函数                        | 循环次数 | 耗时       | 运行平台                              |
| -------- | --------------------------- | ---- | -------- | --------------------------------- |
| 10240000 | `neon_add_sub_v1_optimized` | 10   | 783.63ms | RaspberryPi 4B <br>ARM Cortex-A72 |
| 10240000 | `neon_add_sub_v2`           | 10   | 778.03ms | RaspberryPi 4B <br>ARM Cortex-A72 |
看着这两个函数的执行时间居然差不多，这怎么可能呢？`neon_add_sub_v1_optimized`函数可是用了向量并行化处理的啊，不应该和最原始的写法一样的效率啊，笔者怀疑是编译器偷偷做了手脚，所以把编译优化等级取消，不进行优化，再编译运行，测下耗时:

| n        | 执行函数                        | 循环次数 | 耗时        | 运行平台                              |
| -------- | --------------------------- | ---- | --------- | --------------------------------- |
| 10240000 | `neon_add_sub_v1_optimized` | 10   | 781.72ms  | RaspberryPi 4B <br>ARM Cortex-A72 |
| 10240000 | `neon_add_sub_v2`           | 10   | 2454.61ms | RaspberryPi 4B <br>ARM Cortex-A72 |

这里看起来符合预期，两者的差别有明显的区别啦，果然是编译器帮我们优化了，那编译器是怎么做的呢？我们使用`objdump`来对优化等级为`-O2`编译出来的文件进行反汇编[[反汇编（Disassembly）]]，查看下两个函数对应的汇编代码:

```asm
000106e8 <neon_add_sub_v1_optimized>:
   106e8:	e59dc000 	ldr	ip, [sp]
   106ec:	e1a0c14c 	asr	ip, ip, #2
   106f0:	f4200a8f 	vld1.32	{d0-d1}, [r0]
   106f4:	f4212a8d 	vld1.32	{d2-d3}, [r1]!
   106f8:	f4224a8d 	vld1.32	{d4-d5}, [r2]!
   106fc:	f2000d42 	vadd.f32	q0, q0, q1
   10700:	f2200d44 	vsub.f32	q0, q0, q2
   10704:	f4000a8d 	vst1.32	{d0-d1}, [r0]!
   10708:	f4030a8d 	vst1.32	{d0-d1}, [r3]!
   1070c:	e25cc001 	subs	ip, ip, #1
   10710:	1afffff6 	bne	106f0 <neon_add_sub_v1_optimized+0x8>
   10714:	e12fff1e 	bx	lr

00010718 <neon_add_sub_v2>:
   10718:	e59dc000 	ldr	ip, [sp]
   1071c:	e35c0000 	cmp	ip, #0
   10720:	d12fff1e 	bxle	lr
   10724:	e080c10c 	add	ip, r0, ip, lsl #2
   10728:	ecb17a01 	vldmia	r1!, {s14}
   1072c:	edd07a00 	vldr	s15, [r0]
   10730:	ee777a87 	vadd.f32	s15, s15, s14
   10734:	ece07a01 	vstmia	r0!, {s15}
   10738:	ecb27a01 	vldmia	r2!, {s14}
   1073c:	e150000c 	cmp	r0, ip
   10740:	ee777ac7 	vsub.f32	s15, s15, s14
   10744:	ed407a01 	vstr	s15, [r0, #-4]
   10748:	ece37a01 	vstmia	r3!, {s15}
   1074c:	1afffff5 	bne	10728 <neon_add_sub_v2+0x10>
   10750:	e12fff1e 	bx	lr
```

可以看出来，`neon_add_sub_v2`函数被编译器优化成了流水线友好的形式（数据连续、计算顺序整齐、没有阻塞或跳转）。虽然是标量的处理方式，但是速度和`neon_add_sub_v1_optimized`差不多。