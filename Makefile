# 设置编译器
CC = gcc

# 设置编译选项
CFLAGS = -Wall -Werror

# 设置头文件
CFLAGS += -I ./include

CFLAGS += -O2 -mfpu=neon -mfloat-abi=hard

# 定义源文件
SRCS = ./src/main.c \
       ./src/box_filter.c \
	   ./src/image_process.c \

# 定义链接库
LIBS = -lm \

OBJ = $(SRCS:.c=.o)

# 定义目标程序名
TARGET = output/box_filter_program

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LIBS)


DUAL_LANCH_DEBUG_SRC = ./src/dual_lanch_debug.c

DUAL_LANCH_DEBUG_TARGET = output/dual_lanch_debug_program

DUAL_LANCH_DEBUG_LIBS = -lm

DUAL_LANCH_DEBUG_OBJ = $(DUAL_LANCH_DEBUG_SRC:.c=.o)

.PHONY: dual_lanch_debug
dual_lanch_debug: $(DUAL_LANCH_DEBUG_TARGET)

$(DUAL_LANCH_DEBUG_TARGET):$(DUAL_LANCH_DEBUG_OBJ)
	$(CC) $(DUAL_LANCH_DEBUG_OBJ) -o $(DUAL_LANCH_DEBUG_TARGET) $(DUAL_LANCH_DEBUG_LIBS)

# 编译目标：将 .c 文件编译为 .o 文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJ) $(DUAL_LANCH_DEBUG_OBJ) $(TARGET) $(DUAL_LANCH_DEBUG_TARGET)
