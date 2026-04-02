CC=g++ -O3 -std=c++17
SRCS=$(wildcard *.cpp */*.cpp)
OBJS=$(patsubst %.cpp, %.o, $(SRCS))

# for MacOs
# INCLUDE = -I/usr/local/include/libtorch/include -I/usr/local/include/libtorch/include/torch/csrc/api/include
# LIB +=-L/usr/local/include/libtorch/lib -ltorch -lc10 -lpthread
# FLAG = -Xlinker -rpath -Xlinker /usr/local/include/libtorch/lib

# ---- libtorch paths (edit LIBTORCH_CPU / LIBTORCH_GPU to match your install) ----
# CPU download:
#   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
#   unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip
#   mv libtorch ~/libtorch
#
# GPU (CUDA 12.1) download:
#   wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
#   unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu121.zip
#   mv libtorch ~/libtorch_gpu

LIBTORCH_CPU = /home/proteeti/libtorch
LIBTORCH_GPU = /home/proteeti/libtorch_gpu

TYPE = CPU

ifeq ($(TYPE), GPU)
	LIBTORCH = $(LIBTORCH_GPU)
else
	LIBTORCH = $(LIBTORCH_CPU)
endif

INCLUDE = -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include
LIB    += -L$(LIBTORCH)/lib -ltorch -ltorch_cpu -lc10 -lpthread
FLAG    = -Wl,-rpath=$(LIBTORCH)/lib

NAME=$(wildcard *.cpp)
TARGET=$(patsubst %.cpp, %, $(NAME))


$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(FLAG)
%.o:%.cpp
	$(CC) -o $@ -c $< -g $(INCLUDE)

clean:
	rm -rf $(TARGET) $(OBJS)

# # g++ -std=c++11 Exp.cpp FileReader.o -ltensorflow -o Exp_tf
