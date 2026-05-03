# Compiler
CXX = g++
CXXFLAGS = -O2 -std=c++11 -fopenmp

# Executable name
TARGET = yukawa_parallel

# Object files
OBJS = main_parallel.o init.o energy_p.o scattering_p.o timing.o

# Default rule
all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile rules
main_parallel.o: main_parallel.cpp
	$(CXX) $(CXXFLAGS) -c main_parallel.cpp

init.o: init.cpp
	$(CXX) $(CXXFLAGS) -c init.cpp

energy_p.o: energy_p.cpp
	$(CXX) $(CXXFLAGS) -c energy_p.cpp

scattering_p.o: scattering_p.cpp
	$(CXX) $(CXXFLAGS) -c scattering_p.cpp

timing.o: timing.cpp
	$(CXX) $(CXXFLAGS) -c timing.cpp

# Clean rule
clean:
	rm -f *.o $(TARGET)
