# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O2 -pthread

# Linker flags
LDFLAGS = -lopenblas -pthread

# Source and target files
TARGET = parallelknn
SRCS = V1PTHREADS.cpp
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