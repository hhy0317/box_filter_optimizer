#ifndef BOX_FILTER_H
#define BOX_FILTER_H

#define E_BOX_FILTER_ORIGIN             0   // 最简单的盒子滤波器，直接取均值
#define E_BOX_FILTER_BASE_OPENCV        1   // 基于OpenCV思路的盒子滤波器，行列分离滤波
#define E_BOX_FILTER_BASE_OPENCV_V2     2   // 基于OpenCV思路的盒子滤波器，行列分离滤波+滑窗处理减少重复计算
#define E_BOX_FILTER_REDUCE_CACHE_MISS  3   // 基于OpenCV思路的盒子滤波器，行列分离滤波+滑窗处理+减少cache miss
#define E_BOX_FILTER_NEON_INTRINSICS    4   // 基于NEON Intrinsics的盒子滤波器
#define E_BOX_FILTER_NEON_ASSEMBLY      5   // 基于NEON ASSEMBLY的盒子滤波器(内联汇编)
#define E_BOX_FILTER_NEON_ASSEMBLY_PLD  6   // 基于NEON ASSEMBLY的盒子滤波器(内联汇编)+pld预取指令
#define E_BOX_FILTER_DUAL_LANCH         8   // 基于NEON ASSEMBLY的盒子滤波器(内联汇编)+pld预取指令+双发射流水线

#define BOX_FILTER_MODE                 E_BOX_FILTER_DUAL_LANCH


/*
 * @brief: 最简单的求和盒子滤波器
 */
void box_filter_origin(float *Src, float *Dest, int Width, int Height, int Radius);

/*
 * @brief: 基于OpenCV思路的盒子滤波器，行列分离滤波
 */
void box_filter_opencv(float *src, float *dst, int width, int height, int radius);

/*
 * @brief: 基于OpenCV思路的盒子滤波器，行列分离滤波+滑窗处理减少重复计算
 */
void box_filter_opencv_v2(float *src, float *dst, int width, int height, int radius);

/*
 * @brief: 基于OpenCV思路的盒子滤波器，行列分离滤波+滑窗处理+减少cache miss
 */
void box_filter_reduce_cache_miss(float *src, float *dst, int width, int height, int radius);

/*
 * @brief: 基于 NEON Intrinsics的盒子滤波器
 */
void box_filter_neon_intrinsics(float *src, float *dst, int width, int height, int radius);

/*
 * @brief: 基于 NEON ASSEMBLY的盒子滤波器(内联汇编)
 */
void box_filter_neon_assembly(float *src, float *dst, int width, int height, int radius);

/*
 * @brief: 基于 NEON ASSEMBLY的盒子滤波器(内联汇编)+pld预取指令
 */
void box_filter_neon_assembly_pld(float *src, float *dst, int width, int height, int radius);

/*
 * @brief: 基于 NEON ASSEMBLY的盒子滤波器(内联汇编)+pld预取指令+双发射流水线
 */
void box_filter_neon_assembly_pld_dual_lanch(float *src, float *dst, int width, int height, int radius);

#endif
