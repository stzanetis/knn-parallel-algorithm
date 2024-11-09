# Compiler
CXX = g++

# Linker flags
LDFLAGS = -lopenblas

# Source and target files
TARGET = kNN
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
