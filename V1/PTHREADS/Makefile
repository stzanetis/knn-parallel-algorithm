# Compiler
CXX = g++

# Linker flags
LDFLAGS = -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_cpp -lhdf5 -lopenblas
CXXFLAGS = -I/usr/include/hdf5/serial

# Source and target files
TARGET = parallelkNN
SRCS = V0.cpp
OBJS = $(SRCS:.cpp=.o)

# Deafault target
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
