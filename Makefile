# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O0 -g `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4` -lpthread
TARGET = integratedApp

# Source files
SRCS = integratedApp.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)

# Run the application
run: $(TARGET)
	./$(TARGET) video.mp4

# Dependencies
.PHONY: all clean run
