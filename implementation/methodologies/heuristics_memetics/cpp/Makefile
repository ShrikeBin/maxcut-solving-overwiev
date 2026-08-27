CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fPIC -Wall -Wextra
INCLUDES := -Iinclude -I../../../graphs/cpp

LIB_DIR   := lib
BUILD_DIR := build
TARGET    := $(LIB_DIR)/libheuristics_memetics.so

# Source files
GRAPH_SRCS := ../../../graphs/cpp/Graph.cpp \
              ../../../graphs/cpp/CutState.cpp

ALGO_SRCS  := src/LocalSearch.cpp \
              src/SimulatedAnnealing.cpp \
              src/Approximations.cpp

BIND_SRCS  := src/Bind.cpp

ALL_SRCS   := $(GRAPH_SRCS) $(ALGO_SRCS) $(BIND_SRCS)
OBJS       := $(addprefix $(BUILD_DIR)/, $(notdir $(ALL_SRCS:.cpp=.o)))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

# Object compilation rules
$(BUILD_DIR)/Graph.o: ../../../graphs/cpp/Graph.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/CutState.o: ../../../graphs/cpp/CutState.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/LocalSearch.o: src/LocalSearch.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/SimulatedAnnealing.o: src/SimulatedAnnealing.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/Approximations.o: src/Approximations.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/Bind.o: src/Bind.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(LIB_DIR) $(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)