CXX := g++
NAME := mlp

NUMPY_PATH := /usr/lib/python3/dist-packages/numpy/core/include/numpy
PYTHON_PATH := /usr/include/python3.10

CXXFLAGS := -O3 -march=corei7 -mavx2 -std=c++17 \
            -Wall -Wextra \
            -Wno-deprecated-copy -Wno-deprecated-declarations

INCLUDES := -I lib -I lib/eigen -I include \
            -I $(NUMPY_PATH) -I $(PYTHON_PATH)

LIBS := -lpython3.10

SRCS := main.cpp src/activations.cpp   src/json_loader.cpp    src/optimizers.cpp \
	src/csv_to_eigen.cpp  src/layer.cpp          src/scaler.cpp \
	src/data_spliter.cpp  src/metrics.cpp        src/visualizer.cpp \
	src/history.cpp       src/mlpclassifier.cpp \

OBJS = $(SRCS:.cpp=.o)

CSVLIB := https://raw.githubusercontent.com/d99kris/rapidcsv/refs/heads/master/src/rapidcsv.h
EIGENLIB := https://gitlab.com/libeigen/eigen.git
PLOTLIB := https://raw.githubusercontent.com/lava/matplotlib-cpp/refs/heads/master/matplotlibcpp.h
JSONLIB := https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json.hpp


LIBDIR := lib

all: $(NAME)

libs:
	@echo "Downloading libraries..."
	mkdir -p $(LIBDIR)

	wget -nc -P $(LIBDIR) $(CSVLIB) $(PLOTLIB) $(JSONLIB)

	@if [ ! -d "$(LIBDIR)/eigen" ]; then \
		echo "Folder not found, cloning..."; \
		git clone $(EIGENLIB) $(LIBDIR)/eigen; \
	else \
		echo "Folder exists, skipping clone."; \
	fi


$(NAME): libs 
	$(CXX) $(CXXFLAGS) $(SRCS) $(INCLUDES) $(LIBS) -o $@


# %.o:srcs/%.c $(INC)
# 	$(CXX) $(CXXFLAGS) -c $(SRCS) $(INCLUDES) $(LIBS) $< -o $@

# # clean:
# 	rm -f $(TARGET)

.PHONY: all libs clean fclean