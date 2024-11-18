# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -I/usr/include/hdf5/serial -O2 -fopenmp

# Linker flags
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 -lopenblas -fopenmp

# Source and target files
TARGET = parallelknn
SRCS = V1OMP.cpp
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