# Compiler
CXX = /home/savvas/opencilk-project/bin/clang++

# Compiler flags
CXXFLAGS = -I/usr/include/hdf5/serial -O2 -fopencilk

# Linker flags
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 -lopenblas -fopencilk

# Source and target files
TARGET = parallelknn
SRCS = V1CILK.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Linking the target executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compiling the source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Cleaning the build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean