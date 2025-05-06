#include "image_process.h"
#include "box_filter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <time.h>
#include <stdio.h>

// 首次将RGB图像转为灰度图像,生成后就不需要了
#define RGB_IMAGE_TO_GRAY_ENBALE       0

// 将经过求和盒子滤波后的数据通过取均值，进行归一化处理，再转换成图片，以便检查运行结果
#define DEBUG_OUTPUT_DATA_VISIBLE      1

#define REPEAT_LOOP_TIMES              10

int main()
{
    // 加载图像
    int width, height, n;
    int radius = 3;
    clock_t start, end;

#if RGB_IMAGE_TO_GRAY_ENBALE
    unsigned char *input_data = stbi_load("res/soccer.jpg", &width, &height, &n, 0);
#else
    // 需要png格式，如果是jpg格式，stb库会默认拿到RGB三通道数据
    unsigned char *input_data = stbi_load("res/soccer_gray.png", &width, &height, &n, 1);
#endif

    if (!input_data)
    {
        printf("load image failed!\n");
        goto END_INIT;
    }

    printf("image width = %d, height = %d, n = %d \n", width, height, n);

#if RGB_IMAGE_TO_GRAY_ENBALE
    unsigned char *gray_input_data = malloc(width * height * sizeof(unsigned char));
    if (!gray_input_data)
    {
        printf("malloc gray_input_data failed!\n");
        return -1;
    }

    rgb_to_gray(input_data, gray_input_data, width, height);

    if (stbi_write_png("res/soccer_gray.png", width, height, 1, gray_input_data, width * 1))
    {
        printf("save gray image success!\n");
    }
    else
    {
        printf("save gray image failed!\n");
    }

    goto END_INIT;
#endif

    // 分配内存
    int arr_size_byte = width * height * sizeof(float);

    float *src_data = (float *)malloc(arr_size_byte);
    float *dst_data = (float *)malloc(arr_size_byte);
    if (!src_data || !dst_data)
    {
        printf("malloc failed!\n");
        goto END;
        return -1;
    }

    // 拷贝图像数据
    for (int i = 0; i < width * height; i++)
    {
        src_data[i] = (float)input_data[i];
    }

    start = clock();
    for(int i = 0; i < REPEAT_LOOP_TIMES; i++)
    {
#if (BOX_FILTER_MODE == E_BOX_FILTER_ORIGIN)
        box_filter_origin(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_BASE_OPENCV)
        box_filter_opencv(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_BASE_OPENCV_V2)
        box_filter_opencv_v2(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_REDUCE_CACHE_MISS)
        box_filter_reduce_cache_miss(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_NEON_INTRINSICS)
        box_filter_neon_intrinsics(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_NEON_ASSEMBLY)
        box_filter_neon_assembly(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_NEON_ASSEMBLY_PLD)
        box_filter_neon_assembly_pld(src_data, dst_data, width, height, radius);
#elif (BOX_FILTER_MODE == E_BOX_FILTER_DUAL_LANCH)
        box_filter_neon_assembly_pld_dual_lanch(src_data, dst_data, width, height, radius);
#endif
    }

    end = clock();

    printf("box_filter_mode = %d\n", BOX_FILTER_MODE);
    printf("box_filter_origin time = %f ms CLOCKS_PER_SEC %ld\n", (double)(end - start) / CLOCKS_PER_SEC * 1000, CLOCKS_PER_SEC);

    // debug 处理结果，打印出来处理前后的数据，只打印前10行和前10列数据
    printf("src data[10x10]:\n");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%f ", src_data[i * width + j]);
        }
        printf("\n");
    }

    printf("dst data[10x10]:\n");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            printf("%f ", dst_data[i * width + j]);
        }
        printf("\n");
    }

#if DEBUG_OUTPUT_DATA_VISIBLE
    // 拷贝图像数据,
    for (int i = 0; i < width * height; i++)
    {
        float temp_data = dst_data[i] / (((radius << 1) + 1) * ((radius << 1) + 1));
        input_data[i] = (unsigned char)temp_data;
    }

    stbi_write_png("res/soccer_gray_result.png", width, height, 1, input_data, width * 1);
#endif

END:
    if (src_data)
    {
        free(src_data);
        src_data = NULL;
    }

    if (dst_data)
    {
        free(dst_data);
        dst_data = NULL;
    }

END_INIT:
    if (input_data)
    {
        stbi_image_free(input_data);
        input_data = NULL;
    }

    return 0;
}
