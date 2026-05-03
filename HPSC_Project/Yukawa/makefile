CXX = g++
CXXFLAGS = -O2 -std=c++11
OMPFLAGS = -fopenmp

SERIAL_EXE = run_serial
PARALLEL_EXE = run_parallel

SERIAL_SRC = main.cpp energy.cpp scattering.cpp timing.cpp init.cpp
PARALLEL_SRC = main_parallel.cpp energy_p.cpp scattering_p.cpp timing.cpp init.cpp
all:
	g++ -O2 -std=c++11 main.cpp energy.cpp scattering.cpp timing.cpp init.cpp -o run_serial

parallel:
	g++ -O2 -std=c++11 -fopenmp main_parallel.cpp energy_p.cpp scattering_p.cpp timing.cpp init.cpp -o run_parallel

clean:
	rm -f run_serial run_parallel
	
all: serial

serial:
	$(CXX) $(CXXFLAGS) $(SERIAL_SRC) -o $(SERIAL_EXE)

parallel:
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(PARALLEL_SRC) -o $(PARALLEL_EXE)

clean:
	rm -f $(SERIAL_EXE) $(PARALLEL_EXE)












//# Compiler
//CXX = g++
//CXXFLAGS = -O2 -std=c++11
//
//# Executable name
//TARGET = yukawa
//
//# Object files
//OBJS = main.o init.o forces.o integrator.o  energy.o  scattering.o timing.o
//
//# Default rule
//all: $(TARGET)
//
//# Link step
//$(TARGET): $(OBJS)
//	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
//
//# Compile rules
//main.o: main.cpp
//	$(CXX) $(CXXFLAGS) -c main.cpp
//
//init.o: init.cpp init.h particle.h
//	$(CXX) $(CXXFLAGS) -c init.cpp
//forces.o: forces.cpp forces.h particle.h
//	$(CXX) $(CXXFLAGS) -c forces.cpp
//
//integrator.o:integrator.cpp integrator.h particle.h
//	$(CXX) $(CXXFLAGS)  -c integrator.cpp
//
//energy.o: energy.cpp energy.h particle.h
//	$(CXX) $(CXXFLAGS) -c energy.cpp
//
//scattering.o: scattering.cpp scattering.h
//	$(CXX) $(CXXFLAGS) -c scattering.cpp
//
//timing.o: timing.cpp timing.h
//	$(CXX) $(CXXFLAGS) -c timing.cpp
//
//# Clean rule
//clean:
//	rm -f *.o $(TARGET)
